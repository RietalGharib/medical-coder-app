import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import json
import re
import time
from typing import Any, Dict, Optional, List, Tuple
from google import genai
from google.genai import types

# --- CONFIGURATION ---
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

DEFAULT_MODEL_CANDIDATES = [
    "gemini-1.5-flash", # Often more stable for free tier
    "gemini-2.0-flash",
]

# ==============================
# 1) HELPER: GET CLIENT DYNAMICALLY
# ==============================
def get_client(api_key: str):
    if not api_key:
        raise ValueError("API Key is missing.")
    return genai.Client(api_key=api_key)

# ==============================
# 2) ABBREVIATION UTILS
# ==============================
TIER_A_ABBREVIATIONS: Dict[str, str] = {
    "SOB": "shortness of breath", "SPO2": "oxygen saturation", "SpO2": "oxygen saturation",
    "GCS": "Glasgow Coma Scale", "ECG": "electrocardiogram", "EKG": "electrocardiogram",
    "IV": "intravenous", "NS": "normal saline", "PVC": "premature ventricular contractions",
    "PVCs": "premature ventricular contractions", "ETCO2": "end-tidal carbon dioxide",
    "RR": "respiratory rate", "HR": "heart rate", "BP": "blood pressure",
}
_ABBR_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9]{1,6}s?\b")

def expand_tier_a_shorthand(text: str) -> Tuple[str, List[Dict[str, str]]]:
    if not text: return text, []
    expansions = []
    def repl(match):
        token = match.group(0)
        full = TIER_A_ABBREVIATIONS.get(token) or TIER_A_ABBREVIATIONS.get(token.upper())
        if full:
            expansions.append({"abbr": token, "expanded": full})
            return f"{token} ({full})"
        return token
    return _ABBR_PATTERN.sub(repl, text), expansions

# ==============================
# 3) HYBRID EXTRACTION
# ==============================
def extract_hybrid_content(source):
    try:
        if isinstance(source, str):
            if not os.path.exists(source): return f"Error: File not found at {source}"
            doc = fitz.open(source)
        else:
            source.seek(0)
            doc = fitz.open(stream=source.read(), filetype="pdf")
    except Exception as e:
        return f"Error reading PDF: {e}"
    
    full_text = []
    for page in doc:
        text_layer = page.get_text()
        ocr_text_list = []
        for img in page.get_images(full=True):
            try:
                base = doc.extract_image(img[0])
                image = Image.open(io.BytesIO(base["image"]))
                if image.width < 100 or image.height < 100: continue
                text = pytesseract.image_to_string(image, lang='eng')
                if text.strip(): ocr_text_list.append(f"[Image Text]: {text.strip()}")
            except: continue
        
        content = f"--- Page ---\n{text_layer}\n"
        if ocr_text_list: content += "\n--- OCR ---\n" + "\n".join(ocr_text_list)
        full_text.append(content)
    return "\n".join(full_text)

# ==============================
# 4) PHASE 1: EXTRACTION (With Retry Loop)
# ==============================
PHASE1_PROMPT = """
You are extracting data from a paramedic report. Return ONLY valid JSON.
Schema:
{
  "patient_info": { "patient_id": {"value": "string"}, "name": {"value": "string"}, "age": {"value": "int"} },
  "vitals": { "bp": {"value": "string"}, "pulse": {"value": "int"}, "spo2": {"value": "int"} },
  "chief_complaint": { "text": {"value": "string"} },
  "notes": { "raw_text": {"value": "string"} }
}
"""

def extract_phase1(file_input, api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key: 
        print("❌ Error: Missing API Key")
        return None
        
    client = get_client(api_key)
    text = extract_hybrid_content(file_input)
    if not text: return None

    # Loop through models
    for m in DEFAULT_MODEL_CANDIDATES:
        retries = 0
        # Retry up to 3 times per model
        while retries < 3:
            try:
                resp = client.models.generate_content(
                    model=m,
                    contents=[PHASE1_PROMPT + "\n\nDATA:\n" + text],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                data = json.loads(resp.text)
                
                # Normalization
                if "notes" in data:
                    norm, _ = expand_tier_a_shorthand(data["notes"].get("raw_text", {}).get("value", ""))
                    data["notes"]["raw_text"]["value_normalized"] = norm
                return data

            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"⏳ Rate limit hit on {m}. Waiting 60 seconds... (Attempt {retries+1}/3)")
                    time.sleep(60)
                    retries += 1
                else:
                    print(f"⚠️ Error on {m}: {e}")
                    break 
            
    return None

# ==============================
# 5) PHASE 2: CODING (With Retry Loop)
# ==============================
def run_phase2_coding(phase1_data: Dict, api_key: str) -> Optional[Dict]:
    if not api_key: return None
    client = get_client(api_key)
    prompt = f"Assign ICD-10 codes based on this JSON. Return JSON {{'coding_results': [...]}}.\n{json.dumps(phase1_data)}"
    
    for m in DEFAULT_MODEL_CANDIDATES:
        retries = 0
        while retries < 3:
            try:
                resp = client.models.generate_content(
                    model=m,
                    contents=[prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                return json.loads(resp.text)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"⏳ Rate limit hit on {m} (Phase 2). Waiting 60 seconds...")
                    time.sleep(60)
                    retries += 1
                else:
                    break
    return None
