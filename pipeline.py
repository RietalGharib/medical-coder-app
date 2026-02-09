import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import json
import re
import time
import random
from typing import Any, Dict, Optional, List, Tuple, Union

import requests

# --- CONFIGURATION ---
HF_MODEL_PHASE1 = "mistralai/Mistral-7B-Instruct-v0.3"
HF_MODEL_PHASE2 = "mistralai/Mistral-7B-Instruct-v0.3"

HF_API_URL = "https://api-inference.huggingface.co/models/{}"

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

# ==============================
# 2) ABBREVIATION UTILS
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

    def repl(match):
        token = match.group(0)
        full = TIER_A_ABBREVIATIONS.get(token) or TIER_A_ABBREVIATIONS.get(token.upper())
        if full:
            expansions.append({"abbr": token, "expanded": full})
            return f"{token} ({full})"
        return token

    return _ABBR_PATTERN.sub(repl, text), expansions

# ==============================
# 3) PDF HYBRID EXTRACTION
# ==============================
def extract_hybrid_content(source: Union[str, io.BytesIO]) -> str:
    try:
        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"File not found at {source}")
            doc = fitz.open(source)
        else:
            source.seek(0)
            doc = fitz.open(stream=source.read(), filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}") from e

    try:
        full_text: List[str] = []
        for page_index, page in enumerate(doc, start=1):
            text_layer = page.get_text() or ""
            ocr_text_list: List[str] = []

            for img in page.get_images(full=True):
                try:
                    base = doc.extract_image(img[0])
                    image = Image.open(io.BytesIO(base["image"]))

                    if image.width < 100 or image.height < 100:
                        continue

                    text = pytesseract.image_to_string(image, lang="eng")
                    if text and text.strip():
                        ocr_text_list.append(text.strip())
                except Exception:
                    continue

            content = f"--- Page {page_index} ---\n{text_layer.strip()}\n"
            if ocr_text_list:
                content += "\n--- OCR (Images) ---\n" + "\n\n".join(ocr_text_list)

            full_text.append(content)

        return "\n\n".join(full_text).strip()
    finally:
        doc.close()

# ==============================
# 4) JSON HELPERS
# ==============================
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def _safe_json_loads(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty model response text")

    text = text.strip()

    m = _JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    return json.loads(text)

def _backoff_sleep(attempt: int, base: float = 2.0, cap: float = 30.0) -> None:
    delay = min(cap, base * (2 ** attempt)) + random.uniform(0, 0.8)
    time.sleep(delay)

def _truncate_for_model(text: str, max_chars: int = 60_000) -> str:
    # HF endpoints can be tighter; keep smaller than OpenAI version
    if not text:
        return text
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[:40_000]
    tail = text[-20_000:]
    return head + "\n\n--- TRUNCATED ---\n\n" + tail

# ==============================
# 5) HUGGING FACE CALL
# ==============================
def _hf_generate(token: str, model: str, prompt: str, max_new_tokens: int = 800) -> str:
    token = (token or "").strip()
    if not token:
        raise ValueError("Hugging Face token is missing (expected hf_...).")

    url = HF_API_URL.format(model)
    headers = {"Authorization": f"Bearer {token}"}

    # Most text-generation models accept this inference payload format
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "temperature": 0.2,
        },
        "options": {
            "wait_for_model": True
        }
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)

    # Useful HF errors are in JSON
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"raw": r.text}
        raise RuntimeError(f"HF API error {r.status_code}: {err}")

    data = r.json()

    # Common response formats:
    # 1) [{"generated_text": "..."}]
    # 2) {"generated_text": "..."}  (less common)
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return data[0]["generated_text"] or ""
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"] or ""

    # If format differs, fall back:
    return json.dumps(data)

# ==============================
# 6) PHASE 1
# ==============================
def extract_phase1(file_input: Union[str, io.BytesIO], api_key: str) -> Dict[str, Any]:
    try:
        text = extract_hybrid_content(file_input)
        text = _truncate_for_model(text)
    except Exception as e:
        return {"ok": False, "data": None, "error": f"PDF extraction failed: {e}", "model_used": None}

    if not text.strip():
        return {"ok": False, "data": None, "error": "No text could be extracted from the PDF.", "model_used": None}

    # Give the model very explicit formatting instructions
    prompt = (
        "You are a strict information extraction engine.\n"
        "Return ONLY valid JSON. No markdown. No commentary. No code fences.\n\n"
        f"{PHASE1_PROMPT}\n\n"
        "DATA:\n"
        f"{text}\n\n"
        "Return JSON now:"
    )

    last_error: Optional[str] = None

    for attempt in range(3):
        try:
            out = _hf_generate(api_key, HF_MODEL_PHASE1, prompt, max_new_tokens=900)
            data = _safe_json_loads(out)

            # Normalization
            if isinstance(data, dict) and "notes" in data and isinstance(data["notes"], dict):
                raw = data["notes"].get("raw_text", {}).get("value", "")
                norm, _ = expand_tier_a_shorthand(raw)
                if "raw_text" in data["notes"] and isinstance(data["notes"]["raw_text"], dict):
                    data["notes"]["raw_text"]["value_normalized"] = norm

            return {"ok": True, "data": data, "error": None, "model_used": HF_MODEL_PHASE1}

        except Exception as e:
            last_error = str(e)
            _backoff_sleep(attempt)

    return {"ok": False, "data": None, "error": f"All attempts failed in Phase 1. Last error: {last_error}", "model_used": HF_MODEL_PHASE1}

# ==============================
# 7) PHASE 2
# ==============================
def run_phase2_coding(phase1_data: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    prompt = (
        "You are a medical coding assistant.\n"
        "Assign ICD-10 codes based on this JSON.\n"
        "Return ONLY valid JSON. No markdown.\n"
        "Schema: {\"coding_results\": [{\"icd10_code\":\"...\",\"description\":\"...\"}]}\n\n"
        + json.dumps(phase1_data, ensure_ascii=False)
        + "\n\nReturn JSON now:"
    )

    last_error: Optional[str] = None

    for attempt in range(3):
        try:
            out = _hf_generate(api_key, HF_MODEL_PHASE2, prompt, max_new_tokens=500)
            data = _safe_json_loads(out)
            return {"ok": True, "data": data, "error": None, "model_used": HF_MODEL_PHASE2}
        except Exception as e:
            last_error = str(e)
            _backoff_sleep(attempt)

    return {"ok": False, "data": None, "error": f"All attempts failed in Phase 2. Last error: {last_error}", "model_used": HF_MODEL_PHASE2}

