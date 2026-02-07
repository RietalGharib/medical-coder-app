import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import json
import re
import time  # <--- NEW: Needed for waiting
from typing import Any, Dict, Optional, List, Tuple
from google import genai
from google.genai import types

# --- CONFIGURATION ---
# If you are on Windows local and Tesseract isn't in your PATH, uncomment this:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ==============================
# 0) API & CLIENT SETUP
# ==============================
# PASTE YOUR REAL KEY INSIDE THE QUOTES BELOW
API_KEY = "AIzaSyApRagBjknc_BWQttCKZyJjgM_ls1BMJYo"

def get_client():
    return genai.Client(api_key=API_KEY)

DEFAULT_MODEL_CANDIDATES = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

# ---- DEBUG / COST VISIBILITY ----
LLM_STATS = {
    "phase1_calls": 0,
    "phase1_mode_vision": 0,
    "phase1_mode_text": 0,
    "phase2_calls": 0,
}

# ==============================
# 1) ABBREVIATION UTILS (TIER A)
# ==============================
TIER_A_ABBREVIATIONS: Dict[str, str] = {
    "SOB": "shortness of breath",
    "SPO2": "oxygen saturation",
    "SpO2": "oxygen saturation",
    "GCS": "Glasgow Coma Scale",
    "ECG": "electrocardiogram",
    "EKG": "electrocardiogram",
    "IV": "intravenous",
    "NS": "normal saline",
    "PVC": "premature ventricular contractions",
    "PVCs": "premature ventricular contractions",
    "ETCO2": "end-tidal carbon dioxide",
    "RR": "respiratory rate",
    "HR": "heart rate",
    "BP": "blood pressure",
}

_ABBR_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9]{1,6}s?\b")

def expand_tier_a_shorthand(text: str) -> Tuple[str, List[Dict[str, str]]]:
    if not text:
        return text, []

    expansions: List[Dict[str, str]] = []

    def repl(match: re.Match) -> str:
        token = match.group(0)
        if token in TIER_A_ABBREVIATIONS:
            full = TIER_A_ABBREVIATIONS[token]
        elif token.upper() in TIER_A_ABBREVIATIONS:
            full = TIER_A_ABBREVIATIONS[token.upper()]
        else:
            return token

        expansions.append({"abbr": token, "expanded": full})
        return f"{token} ({full})"

    expanded = _ABBR_PATTERN.sub(repl, text)
    return expanded, expansions

# ==================================
# 2) HYBRID EXTRACTION (TEXT + OCR)
# ==================================
def extract_hybrid_content(source):
    """
    Extracts text from PDF (digital text first, then OCR for images).
    Handles both file paths (str) and Streamlit uploads (BytesIO).
    """
    try:
        # Check if source is a string (file path) or a stream (uploaded file)
        if isinstance(source, str):
            if not os.path.exists(source):
                return f"Error: File not found at {source}"
            doc = fitz.open(source)
        else:
            # It's a Streamlit uploaded file object
            # Reset pointer to start just in case
            source.seek(0)
            bytes_data = source.read()
            doc = fitz.open(stream=bytes_data, filetype="pdf")
            
    except Exception as e:
        return f"Error reading PDF: {e}"
    
    full_text = []
    print(f"[Extractor] Processing {len(doc)} pages with Hybrid OCR...")

    for page_num, page in enumerate(doc):
        # 1. HARDCODE: Extract digital text (Fast & Accurate)
        text_layer = page.get_text()
        
        # 2. OCR: Extract images
        image_list = page.get_images(full=True)
        ocr_text_list = []
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Load into PIL
                image = Image.open(io.BytesIO(image_bytes))
                
                # Filter small icons/noise (Adjust 100x100 if needed)
                if image.width < 100 or image.height < 100:
                    continue
                
                # Run OCR
                text = pytesseract.image_to_string(image, lang='eng')
                if text.strip():
                    ocr_text_list.append(f"[Image Text]: {text.strip()}")
            except Exception:
                continue # Skip image if OCR fails

        # 3. MERGE
        page_content = f"--- Page {page_num + 1} ---\n{text_layer}\n"
        if ocr_text_list:
            page_content += "\n--- Image Content (OCR) ---\n" + "\n".join(ocr_text_list)
            
        full_text.append(page_content)

    return "\n".join(full_text)

# ==================================
# 3) PHASE 1: PARSING TO JSON
# ==================================
PHASE1_PROMPT = """
You are extracting data from a paramedic report. 
The text provided below was extracted from a PDF (including OCR from images).

Your job is Phase 1 ONLY:
- Read the text as-is.
- Produce a faithful JSON representation of the visible fields and tables.
- Do NOT infer diagnoses or add medical interpretations.

Rules:
1) Return ONLY valid JSON. No markdown, no commentary.
2) Every extracted field MUST include evidence: a list of exact text snippets copied from the document.
3) If something is unclear/ambiguous, include raw text, confidence, and add a quality_flag.

Schema (return exactly these top-level keys, even if empty lists/nulls):
{
  "document_meta": {
    "source_file": "string",
    "page_count": "int|null",
    "printed_datetime": { "value": "string|null", "evidence": ["..."], "confidence": "high|medium|low" }
  },
  "patient_info": {
    "patient_id": { "value": "string|null", "evidence": ["..."], "confidence": "high|medium|low" },
    "first_name": { "value": "string|null", "evidence": ["..."], "confidence": "high|medium|low" },
    "surname": { "value": "string|null", "evidence": ["..."], "confidence": "high|medium|low" },
    "date_of_birth": { "value": "string|null", "evidence": ["..."], "confidence": "high|medium|low" },
    "age_years": { "value": "int|null", "raw": "string|null", "evidence": ["..."], "confidence": "high|medium|low" },
    "gender": { "value": "string|null", "evidence": ["..."], "confidence": "high|medium|low" }
  },
  "chief_complaint": {
    "text": { "value": "string|null", "evidence": ["..."], "confidence": "high|medium|low" }
  },
  "vitals": {
    "bp_systolic": { "value": "int|null", "evidence": ["..."], "confidence": "high|medium|low" },
    "bp_diastolic": { "value": "int|null", "evidence": ["..."], "confidence": "high|medium|low" },
    "pulse_rate": { "value": "int|null", "evidence": ["..."], "confidence": "high|medium|low" },
    "spo2": { "value": "int|null", "evidence": ["..."], "confidence": "high|medium|low" },
    "gcs_total": { "value": "int|null", "evidence": ["..."], "confidence": "high|medium|low" }
  },
  "clinical_impression": [
    { "system": "string", "impression": "string", "evidence": ["..."], "confidence": "high|medium|low" }
  ],
  "medications_administered": [
    { "medication": { "value": "string", "evidence": ["..."], "confidence": "high|medium|low" } }
  ],
  "notes": {
    "raw_text": { "value": "string|null", "evidence": ["..."], "confidence": "high|medium|low" }
  },
  "symptoms_reported": [
    { "text": "string", "evidence": ["..."], "confidence": "high|medium|low" }
  ]
}
Important: Use null when absent; do not invent.
"""

def generate_phase1_json_text(client: genai.Client, models: List[str], file_input) -> Optional[str]:
    # 1. Run the Hybrid Extractor
    extracted_text = extract_hybrid_content(file_input)
    
    if not extracted_text or len(extracted_text) < 50:
        print("❌ Extraction failed or document empty.")
        return None

    # 2. Construct Prompt with Extracted Text
    contents = [PHASE1_PROMPT + "\n\n---\nEXTRACTED DOCUMENT CONTENT:\n" + extracted_text]

    # 3. Call AI with Retry Logic
    last_err = None
    for m in models:
        try:
            print(f"🚀 Phase 1: Extracting JSON using model: {m}")
            LLM_STATS["phase1_calls"] += 1
            LLM_STATS["phase1_mode_text"] += 1 

            resp = client.models.generate_content(
                model=m,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            return resp.text

        except Exception as e:
            last_err = e
            # Handle Rate Limits (429) explicitly
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"⏳ Rate limit hit on {m}. Waiting 30 seconds before retrying...")
                time.sleep(30)
                # Try the same model again or just let the loop continue to the next one
                # For simplicity, we continue to the next model (which might be the same type)
            else:
                print(f"⚠️ Failed on {m}: {e}")

    print(f"❌ All models failed. Last error: {last_err}")
    return None

def beautify_phase1(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj = dict(obj)
    vitals = obj.get("vitals", {})
    if "heart_rate" in vitals and "pulse_rate" not in vitals:
        vitals["pulse_rate"] = vitals.pop("heart_rate")
    obj["vitals"] = vitals
    return obj

def apply_phase1_5_normalization(p1: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(p1)
    normalization = { "tier_a_expansions": [], "normalized_fields": {} }

    # Helper to normalize a single field path
    def normalize_field(val, field_name):
        if isinstance(val, str) and val.strip():
            expanded, exps = expand_tier_a_shorthand(val)
            if exps:
                normalization["tier_a_expansions"].extend(
                    [{"field": field_name, **e} for e in exps]
                )
                normalization["normalized_fields"][field_name] = expanded
                return expanded
        return val

    # 1. Chief Complaint
    cc_val = (out.get("chief_complaint") or {}).get("text", {}).get("value")
    normalize_field(cc_val, "chief_complaint.text")

    # 2. Notes
    notes_val = (out.get("notes") or {}).get("raw_text", {}).get("value")
    normalize_field(notes_val, "notes.raw_text")

    # 3. Clinical Impressions
    cis = out.get("clinical_impression") or []
    if isinstance(cis, list):
        normalized_cis = []
        for idx, item in enumerate(cis):
            item = dict(item)
            imp = item.get("impression")
            norm_imp = normalize_field(imp, f"clinical_impression[{idx}].impression")
            if norm_imp != imp:
                item["impression_normalized"] = norm_imp
            normalized_cis.append(item)
        out["clinical_impression"] = normalized_cis

    out["normalization"] = normalization
    return out

def extract_phase1(file_path_or_obj) -> Optional[Dict[str, Any]]:
    client = get_client()
    
    # Model selection
    try:
        available = [m.name.split("/")[-1] for m in client.models.list()]
        models = [c for c in DEFAULT_MODEL_CANDIDATES if c in available]
        if not models: models = DEFAULT_MODEL_CANDIDATES
    except:
        models = DEFAULT_MODEL_CANDIDATES

    # Generate JSON
    json_text = generate_phase1_json_text(client, models, file_path_or_obj)
    
    if not json_text:
        return None

    try:
        raw_obj = json.loads(json_text)
        p1 = beautify_phase1(raw_obj)
        p1 = apply_phase1_5_normalization(p1)
        return p1
    except json.JSONDecodeError:
        print("❌ Invalid JSON returned by model.")
        return None

# ==================================
# 4) PHASE 2: CODING AGENT
# ==================================
PHASE2_PROMPT_TEMPLATE = """
You are a professional Medical Coder.
Your task is to assign ICD-10-CM codes based STRICTLY on the extracted text provided.

---
SAFE ABBREVIATION REFERENCE GUIDE (UNAMBIGUOUS ONLY):
- "PVC" / "PVCs" -> Premature Ventricular Contractions
- "SOB" -> Shortness of Breath
- "SpO2" / "SPO2" -> Oxygen saturation
- "GCS" -> Glasgow Coma Scale
- "ECG" / "EKG" -> Electrocardiogram
- "IV" -> Intravenous
- "NS" -> Normal saline
- "ETCO2" -> End-tidal carbon dioxide
- "RR" -> Respiratory rate
- "HR" -> Heart rate
- "BP" -> Blood pressure

IMPORTANT SAFETY RULES:
1. NEVER expand ambiguous abbreviations (e.g., CP, MI) without context.
2. Do NOT infer qualifiers (e.g., "acute", "chronic") unless explicitly written.
3. Code ONLY what is documented.

PATIENT DATA (JSON):
{patient_data_json}

REQUIRED OUTPUT FORMAT (JSON ONLY):
{{
  "coding_results": [
    {{
      "icd10_code": "string",
      "description": "string",
      "source_text": "string (exact snippet)",
      "reasoning": "string"
    }}
  ]
}}
"""

def run_phase2_coding(phase1_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    client = get_client()
    models = DEFAULT_MODEL_CANDIDATES # Use same list

    data_str = json.dumps(phase1_data, indent=2)
    prompt = PHASE2_PROMPT_TEMPLATE.format(patient_data_json=data_str)

    print("\n🧠 Phase 2: Analyzing symptoms & Assigning Codes...")

    for m in models:
        try:
            LLM_STATS["phase2_calls"] += 1
            resp = client.models.generate_content(
                model=m,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            return json.loads(resp.text)
        except Exception as e:
            # Simple retry for phase 2 as well
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"⏳ Rate limit hit on Phase 2. Waiting 30 seconds...")
                time.sleep(30)
                # In a real loop we'd retry, but here we just catch it so it doesn't crash 
                # and loop to the next model candidate
            print(f"⚠️ Phase 2 failed on {m}: {e}")
            
    return None

# ==================================
# 5) MAIN EXECUTION PIPELINE
# ==================================
if __name__ == "__main__":
    # Update with your actual file path
    TARGET_FILE = r"C:\Users\rieta\OneDrive\Desktop\medical_ocr\WWD2YWN-A-U49.pdf"

    print(f"🚀 STARTING PIPELINE for {TARGET_FILE}")

    # --- STEP 1: EXTRACT ---
    p1_result = extract_phase1(TARGET_FILE)

    if p1_result:
        print("\n✅ PHASE 1 COMPLETE")
        with open("phase1_extracted.json", "w", encoding="utf-8") as f:
            json.dump(p1_result, f, indent=2, ensure_ascii=False)
        
        # --- STEP 2: CODE ---
        p2_result = run_phase2_coding(p1_result)
        
        if p2_result:
            print("\n✅ PHASE 2 COMPLETE")
            print(json.dumps(p2_result, indent=2))
            
            with open("phase2_coded.json", "w", encoding="utf-8") as f:
                json.dump(p2_result, f, indent=2, ensure_ascii=False)

            # --- STEP 3: FINAL MERGE ---
            final_report = {
                "patient_summary": p1_result,
                "icd10_coding": p2_result.get("coding_results", [])
            }
            with open("final_medical_report.json", "w", encoding="utf-8") as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
                print("\n💾 FINAL REPORT SAVED: final_medical_report.json")
        else:
            print("❌ Phase 2 Failed.")
    else:
        print("❌ Phase 1 Failed.")
    
