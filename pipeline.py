from google import genai
from google.genai import types
import os
import json
from typing import Any, Dict, Optional, List, Tuple
from pypdf import PdfReader
from typing import Tuple
import re
from typing import Dict, List, Any, Tuple

# Tier A: unambiguous EMS/clinical abbreviations (safe auto-expand)
TIER_A_ABBREVIATIONS: Dict[str, str] = {
    "SOB": "shortness of breath",
    "SPO2": "oxygen saturation",
    "SpO2": "oxygen saturation",  # handle common casing
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

_ABBR_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9]{1,6}s?\b")  # catches SOB, PVCs, SpO2, etc.

def expand_tier_a_shorthand(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Safe expansion:
    - Only expands Tier-A abbreviations.
    - Returns (expanded_text, expansions_metadata)
    - Does NOT guess. Unknown tokens are ignored.
    """
    if not text:
        return text, []

    expansions: List[Dict[str, str]] = []

    def repl(match: re.Match) -> str:
        token = match.group(0)
        # check exact token then uppercase token
        if token in TIER_A_ABBREVIATIONS:
            full = TIER_A_ABBREVIATIONS[token]
        elif token.upper() in TIER_A_ABBREVIATIONS:
            full = TIER_A_ABBREVIATIONS[token.upper()]
        else:
            return token

        expansions.append({"abbr": token, "expanded": full})
        # Add expansion without deleting original token (safest)
        return f"{token} ({full})"

    expanded = _ABBR_PATTERN.sub(repl, text)
    return expanded, expansions

def extract_pdf_text_layer(file_path: str) -> Tuple[str, float]:
    """
    Extract embedded text from a PDF and return (text, quality_score 0..1).
    quality_score is a heuristic: higher = likely a clean text-layer PDF.
    """
    reader = PdfReader(file_path)
    pages_text = []
    total_chars = 0
    printable_chars = 0

    for page in reader.pages:
        t = page.extract_text() or ""
        pages_text.append(t)

        total_chars += len(t)
        printable_chars += sum(1 for ch in t if ch.isprintable() and not ch.isspace())

    text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text).strip()

    if total_chars == 0:
        return "", 0.0

    # Heuristics:
    density = printable_chars / max(total_chars, 1)          # 0..1
    length_ok = 1.0 if total_chars > 1500 else total_chars / 1500.0
    quality = 0.6 * density + 0.4 * length_ok               # 0..1

    return text, max(0.0, min(1.0, quality))

# ==============================
# 0) SECURITY: BRUTE FORCE (TEMPORARY)
# ==============================
# PASTE YOUR REAL KEY INSIDE THE QUOTES BELOW
API_KEY = "AIzaSyDVuyyOuPkfCN_RzXWJ_AXLonk0OsTlb6w" 

from google import genai
# ... rest of imports ...

def get_client():
    return genai.Client(api_key=API_KEY)
# 3. If STILL not found, don't crash yet! 
# We will handle the error when we try to use the client, 
# allowing the UI to load so the user can enter a key manually.

# ---- DEBUG / COST VISIBILITY ----
LLM_STATS = {
"phase1_calls": 0,
"phase1_mode_vision": 0,
"phase1_mode_text": 0,
"phase2_calls": 0,
}

DEFAULT_MODEL_CANDIDATES = [
    
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

# ==================================
# 1) CLIENT + MODEL SELECTION
# ==================================
def get_client() -> genai.Client:
    return genai.Client(api_key=API_KEY)

def choose_model(client: genai.Client, candidates: List[str]) -> List[str]:
    """Returns an ordered list of models to try (best effort)."""
    try:
        available = []
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            available.append(name.split("/")[-1])

        ordered = [c for c in candidates if c in available]
        return ordered if ordered else candidates
    except Exception:
        return candidates

# ==================================
# 2) PHASE 1 PROMPT (FULL DOCUMENT)
# ==================================
PHASE1_PROMPT = """
You are extracting data from a paramedic report (image/PDF). Your job is Phase 1 ONLY:
- Read the document as-is.
- Produce a faithful JSON representation of the visible fields and tables.
- Do NOT infer diagnoses or add medical interpretations.
- Do NOT convert symptoms into ICD-10.

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

# ==================================
# 3) PHASE 1 GENERATION
# ==================================
# ==================================
# 3) PHASE 1 GENERATION
# ==================================
def generate_phase1_json_text(
    client: genai.Client,
    models: List[str],
    file_path: str
) -> Optional[str]:
    last_err = None

    # Decide how we will feed content to the model
    if file_path.lower().endswith(".pdf"):
        pdf_text, quality = extract_pdf_text_layer(file_path)
        print(f"[Router] PDF text-layer quality={quality:.2f} chars={len(pdf_text)}")

        # NEW: if text layer is good, do cheaper text-only Phase 1
        if quality >= 0.70 and len(pdf_text) >= 800:
            contents = [PHASE1_PROMPT + "\n\n---\nPDF TEXT LAYER (verbatim):\n" + pdf_text]
            phase1_mode = "text"
        else:
            uploaded = client.files.upload(file=file_path)
            contents = [uploaded, PHASE1_PROMPT]
            phase1_mode = "vision"

    else:
        from PIL import Image
        img = Image.open(file_path)
        contents = [img, PHASE1_PROMPT]
        phase1_mode = "vision"

    # Try models in order
    for m in models:
        try:
            print(f"🚀 Phase 1: Extraction using model: {m}")

            # Cost visibility
            LLM_STATS["phase1_calls"] += 1
            if phase1_mode == "text":
                LLM_STATS["phase1_mode_text"] += 1
            else:
                LLM_STATS["phase1_mode_vision"] += 1

            print(f"[LLM] Phase 1 call #{LLM_STATS['phase1_calls']} ({phase1_mode}) model={m}")

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
            print(f"⚠️ Failed on {m}: {e}")

    print(f"❌ All models failed. Last error: {last_err}")
    return None

def beautify_phase1(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Basic cleanup to ensure consistent structure."""
    obj = dict(obj)
    vitals = obj.get("vitals", {})
    if "heart_rate" in vitals and "pulse_rate" not in vitals:
        vitals["pulse_rate"] = vitals.pop("heart_rate")
    obj["vitals"] = vitals
    return obj
def apply_phase1_5_normalization(p1: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a COPY of Phase-1 JSON with additive normalization fields.
    Raw evidence + original values remain unchanged.
    """
    out = dict(p1)  # shallow copy

    normalization = {
        "tier_a_expansions": [],
        "normalized_fields": {}
    }

    # Normalize specific text-bearing fields safely
    # 1) chief complaint
    try:
        cc = (out.get("chief_complaint") or {}).get("text") or {}
        cc_val = cc.get("value")
        if isinstance(cc_val, str) and cc_val.strip():
            expanded, exps = expand_tier_a_shorthand(cc_val)
            if exps:
                normalization["tier_a_expansions"].extend(
                    [{"field": "chief_complaint.text", **e} for e in exps]
                )
                normalization["normalized_fields"]["chief_complaint.text"] = expanded
    except Exception:
        pass

    # 2) notes raw text
    try:
        notes = (out.get("notes") or {}).get("raw_text") or {}
        notes_val = notes.get("value")
        if isinstance(notes_val, str) and notes_val.strip():
            expanded, exps = expand_tier_a_shorthand(notes_val)
            if exps:
                normalization["tier_a_expansions"].extend(
                    [{"field": "notes.raw_text", **e} for e in exps]
                )
                normalization["normalized_fields"]["notes.raw_text"] = expanded
    except Exception:
        pass

    # 3) clinical impressions
    try:
        cis = out.get("clinical_impression") or []
        if isinstance(cis, list):
            normalized_cis = []
            for idx, item in enumerate(cis):
                item = dict(item)
                imp = item.get("impression")
                if isinstance(imp, str) and imp.strip():
                    expanded, exps = expand_tier_a_shorthand(imp)
                    if exps:
                        normalization["tier_a_expansions"].extend(
                            [{"field": f"clinical_impression[{idx}].impression", **e} for e in exps]
                        )
                        item["impression_normalized"] = expanded  # additive field
                normalized_cis.append(item)
            out["clinical_impression"] = normalized_cis
    except Exception:
        pass

    # Attach normalization metadata
    out["normalization"] = normalization
    return out
def extract_phase1(file_path: str) -> Optional[Dict[str, Any]]:
    client = get_client()
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    models = choose_model(client, DEFAULT_MODEL_CANDIDATES)
    json_text = generate_phase1_json_text(client, models, file_path)
    if not json_text:
        return None

    try:
        raw_obj = json.loads(json_text)

        # Phase 1 cleanup
        p1 = beautify_phase1(raw_obj)

        # ⬇️ THIS IS THE EXACT LINE YOU ADD
        p1 = apply_phase1_5_normalization(p1)

        return p1

    except json.JSONDecodeError:
        print("❌ Invalid JSON returned by model.")
        return None

# ==================================
# 4) PHASE 2: CODING AGENT (NEW)
# ==================================
PHASE2_PROMPT_TEMPLATE = """
You are a professional Medical Coder.
Your task is to assign ICD-10-CM codes based STRICTLY on the extracted text provided.
Do not infer, assume, or reinterpret clinical meaning beyond what is explicitly documented.

---
SAFE ABBREVIATION REFERENCE GUIDE (UNAMBIGUOUS ONLY):
The following abbreviations are guaranteed to be unambiguous in this context and may be safely interpreted:

- "PVC" / "PVCs" → Premature Ventricular Contractions
- "SOB" → Shortness of Breath
- "SpO2" / "SPO2" → Oxygen saturation
- "GCS" → Glasgow Coma Scale
- "ECG" / "EKG" → Electrocardiogram
- "IV" → Intravenous
- "NS" → Normal saline
- "ETCO2" → End-tidal carbon dioxide
- "RR" → Respiratory rate
- "HR" → Heart rate
- "BP" → Blood pressure

No other abbreviations may be expanded unless the full meaning is explicitly written in the text.

---
IMPORTANT SAFETY RULES (MANDATORY):

1. NEVER expand ambiguous abbreviations (e.g., DOB, CP, BS, PT, RA, MI, OD).
   - Do NOT guess their meaning from context.
   - Do NOT code based on them unless the expanded meaning is explicitly written out.

2. Do NOT infer qualifiers, severity, timing, or subtypes.
   - Examples: "on exertion", "acute", "chronic", "with complication".
   - These may ONLY be used if the exact wording appears in the text.

3. Prefer conservative, unspecified symptom codes when documentation lacks specificity.
   - Example: "chest discomfort" without further qualifiers → code as R07.4 (Chest pain, unspecified).

4. Code ONLY what is clearly and explicitly documented.
   - If documentation is vague or ambiguous, do NOT code it.

5. If a condition is suspected, ruled out, or not confirmed,
   code the presenting symptom — NOT the suspected diagnosis.

---
PATIENT DATA (JSON):
{patient_data_json}

---
INSTRUCTIONS:

1. Identify clinical findings, symptoms, and impressions exactly as written.
2. Apply ONLY the SAFE abbreviation guide above.
3. Assign the most accurate ICD-10-CM code supported by explicit text.
4. Do NOT add, infer, or refine beyond documentation.

---
REQUIRED OUTPUT FORMAT (JSON ONLY):

{{
  "coding_results": [
    {{
      "icd10_code": "string (e.g., R07.4)",
      "description": "string (official ICD-10 description)",
      "source_text": "string (exact text snippet being coded)",
      "reasoning": "string (why this code is supported by the text)"
    }}
  ]
}}
"""

def run_phase2_coding(phase1_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    client = get_client()
    models = choose_model(client, DEFAULT_MODEL_CANDIDATES)

    # Prepare data for prompt
    data_str = json.dumps(phase1_data, indent=2)
    prompt = PHASE2_PROMPT_TEMPLATE.format(patient_data_json=data_str)

    print("\n🧠 Phase 2: Analyzing symptoms & Assigning Codes...")

    last_err = None
    for m in models:
        try:
            LLM_STATS["phase2_calls"] += 1
            print(f"[LLM] Phase 2 call #{LLM_STATS['phase2_calls']} model={m}")
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
            last_err = e
            print(f"⚠️ Phase 2 failed on {m}: {e}")
            
    print(f"❌ Phase 2 failed. Last error: {last_err}")
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
        # Save Phase 1
        with open("phase1_extracted.json", "w", encoding="utf-8") as f:
            json.dump(p1_result, f, indent=2, ensure_ascii=False)
        
        # --- STEP 2: CODE ---
        p2_result = run_phase2_coding(p1_result)
        
        if p2_result:
            print("\n✅ PHASE 2 COMPLETE")
            print(json.dumps(p2_result, indent=2))
            
            # Save Phase 2
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
