# =========================
# pipeline.py  (REPLACE FULL FILE)
# =========================
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

from openai import OpenAI

# --- CONFIGURATION ---
OPENAI_MODEL_PHASE1 = "gpt-4o-mini"
OPENAI_MODEL_PHASE2 = "gpt-4o-mini"

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
# 1) CLIENT
# ==============================
def get_client(api_key: str) -> OpenAI:
    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError("API Key is missing.")
    return OpenAI(api_key=api_key)

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
    """
    Extracts:
    - text layer from PDF pages
    - OCR from embedded images
    Returns a single combined string.
    """
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

                    # ignore tiny icons
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

def _truncate_for_model(text: str, max_chars: int = 120_000) -> str:
    if not text:
        return text
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[:80_000]
    tail = text[-40_000:]
    return head + "\n\n--- TRUNCATED ---\n\n" + tail

# ==============================
# 5) PHASE 1: EXTRACTION (OpenAI)
# ==============================
def extract_phase1(file_input: Union[str, io.BytesIO], api_key: str) -> Dict[str, Any]:
    try:
        client = get_client(api_key)
    except Exception as e:
        return {"ok": False, "data": None, "error": f"Client init failed: {e}", "model_used": None}

    try:
        text = extract_hybrid_content(file_input)
        text = _truncate_for_model(text)
    except Exception as e:
        return {"ok": False, "data": None, "error": f"PDF extraction failed: {e}", "model_used": None}

    if not text.strip():
        return {"ok": False, "data": None, "error": "No text could be extracted from the PDF.", "model_used": None}

    last_error: Optional[str] = None

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL_PHASE1,
                messages=[
                    {"role": "system", "content": "You are a strict information extraction engine. Return ONLY valid JSON. No markdown."},
                    {"role": "user", "content": PHASE1_PROMPT + "\n\nDATA:\n" + text},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            content = resp.choices[0].message.content or ""
            data = _safe_json_loads(content)

            # Normalization: expand abbreviations in notes.raw_text
            if isinstance(data, dict) and "notes" in data and isinstance(data["notes"], dict):
                raw = data["notes"].get("raw_text", {}).get("value", "")
                norm, _ = expand_tier_a_shorthand(raw)
                if "raw_text" in data["notes"] and isinstance(data["notes"]["raw_text"], dict):
                    data["notes"]["raw_text"]["value_normalized"] = norm

            return {"ok": True, "data": data, "error": None, "model_used": OPENAI_MODEL_PHASE1}

        except Exception as e:
            last_error = str(e)
            _backoff_sleep(attempt)

    return {"ok": False, "data": None, "error": f"All attempts failed in Phase 1. Last error: {last_error}", "model_used": OPENAI_MODEL_PHASE1}

# ==============================
# 6) PHASE 2: CODING (OpenAI)
# ==============================
def run_phase2_coding(phase1_data: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    try:
        client = get_client(api_key)
    except Exception as e:
        return {"ok": False, "data": None, "error": f"Client init failed: {e}", "model_used": None}

    prompt = (
        "Assign ICD-10 codes based on this JSON. "
        "Return ONLY JSON in this schema: "
        "{\"coding_results\": [{\"icd10_code\":\"...\",\"description\":\"...\"}]}\n\n"
        + json.dumps(phase1_data, ensure_ascii=False)
    )

    last_error: Optional[str] = None

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL_PHASE2,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            content = resp.choices[0].message.content or ""
            data = _safe_json_loads(content)

            return {"ok": True, "data": data, "error": None, "model_used": OPENAI_MODEL_PHASE2}

        except Exception as e:
            last_error = str(e)
            _backoff_sleep(attempt)

    return {"ok": False, "data": None, "error": f"All attempts failed in Phase 2. Last error: {last_error}", "model_used": OPENAI_MODEL_PHASE2}
