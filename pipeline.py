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

# --- Hugging Face OpenAI-compatible router configuration ---
HF_BASE_URL = "https://router.huggingface.co/v1"

# Small chat/instruct model that works on HF Inference via router.
HF_CHAT_MODEL = "HuggingFaceTB/SmolLM3-3B:hf-inference"

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
def get_client(hf_token: str) -> OpenAI:
    hf_token = (hf_token or "").strip()
    if not hf_token:
        raise ValueError("Hugging Face token is missing (expected hf_...).")
    return OpenAI(base_url=HF_BASE_URL, api_key=hf_token)

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
    if not text:
        return text
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[:40_000]
    tail = text[-20_000:]
    return head + "\n\n--- TRUNCATED ---\n\n" + tail

# ==============================
# 5) PHASE 1 (HF Router)
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
            resp = client.chat.completions.create(
                model=HF_CHAT_MODEL,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                timeout=60,  # prevent long hangs
            )

            content = resp.choices[0].message.content or ""
            data = _safe_json_loads(content)

            if isinstance(data, dict) and "notes" in data and isinstance(data["notes"], dict):
                raw = data["notes"].get("raw_text", {}).get("value", "")
                norm, _ = expand_tier_a_shorthand(raw)
                if "raw_text" in data["notes"] and isinstance(data["notes"]["raw_text"], dict):
                    data["notes"]["raw_text"]["value_normalized"] = norm

            return {"ok": True, "data": data, "error": None, "model_used": HF_CHAT_MODEL}

        except Exception as e:
            last_error = str(e)
            _backoff_sleep(attempt)

    return {"ok": False, "data": None, "error": f"All attempts failed in Phase 1. Last error: {last_error}", "model_used": HF_CHAT_MODEL}

# ==================================
# 6) PHASE 2: CODING AGENT (YOUR NEW VERSION, INTEGRATED)
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

PHASE2_MODEL_CANDIDATES = [
    HF_CHAT_MODEL,
    "mistralai/Mistral-7B-Instruct-v0.3:hf-inference",
]

LLM_STATS: Dict[str, int] = {"phase2_calls": 0}

def run_phase2_coding(phase1_data: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """
    Integrated Phase 2:
    - Uses your prompt template exactly
    - Tries models in order
    - Returns Streamlit envelope like Phase 1
    """
    try:
        client = get_client(api_key)
    except Exception as e:
        return {"ok": False, "data": None, "error": f"Client init failed: {e}", "model_used": None}

    data_str = json.dumps(phase1_data, indent=2, ensure_ascii=False)
    prompt = PHASE2_PROMPT_TEMPLATE.format(patient_data_json=data_str)

    last_err: Optional[str] = None

    for m in PHASE2_MODEL_CANDIDATES:
        for attempt in range(3):
            try:
                LLM_STATS["phase2_calls"] += 1

                resp = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    timeout=60,  # prevent long hangs
                )

                content = resp.choices[0].message.content or ""
                data = _safe_json_loads(content)

                return {"ok": True, "data": data, "error": None, "model_used": m}

            except Exception as e:
                last_err = f"model={m}, attempt={attempt+1}/3, error={e}"
                _backoff_sleep(attempt)

    return {"ok": False, "data": None, "error": f"Phase 2 failed. Last error: {last_err}", "model_used": None}
