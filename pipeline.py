import os
import json
import re
import asyncio
import time
import logging
import hashlib
from typing import Any, Dict, Optional, List, Tuple, Generic, TypeVar, Union

from pypdf import PdfReader
from PIL import Image

from google import genai
from google.genai import types

from pydantic import BaseModel, Field, ValidationError, field_validator

# ==============================
# 0) LOGGING CONFIGURATION
# ==============================
LOG_LEVEL = logging.DEBUG

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==============================
# 1) SECURITY
# ==============================
# ✅ DO NOT hardcode keys in source code (especially if repo is public)
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# ==============================
# 2) CONFIGURATION
# ==============================
DEFAULT_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-exp",
]

MAX_CONCURRENT_REQUESTS = 5

# Tier A: safe expansions only
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


def safe_job_id(file_path: str) -> str:
    """PHI-safe job id: derived hash, no filename leakage."""
    h = hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:10]
    return f"job_{h}"


# ==============================
# 3) TEXT NORMALIZATION (SAFE)
# ==============================
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


# ==============================
# 4) PDF TEXT EXTRACTION
# ==============================
def extract_pdf_text_layer(file_path: str) -> Tuple[str, float]:
    """
    Returns: (text, quality_score 0..1)
    """
    try:
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

        density = printable_chars / max(total_chars, 1)
        length_ok = 1.0 if total_chars > 1500 else total_chars / 1500.0
        quality = 0.6 * density + 0.4 * length_ok
        return text, max(0.0, min(1.0, quality))

    except Exception as e:
        logger.error(f"PDF read failed: {type(e).__name__}")
        return "", 0.0


# ==============================
# 5) GEMINI CLIENT
# ==============================
def get_client() -> genai.Client:
    return genai.Client(api_key=API_KEY)


def choose_model(client: genai.Client, candidates: List[str]) -> List[str]:
    try:
        available = []
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            available.append(name.split("/")[-1])
        ordered = [c for c in candidates if c in available]
        return ordered if ordered else candidates
    except Exception:
        return candidates


# ==============================
# 6) PYDANTIC MODELS
# ==============================
T = TypeVar("T")


class EvidenceField(BaseModel, Generic[T]):
    value: Optional[T] = None
    evidence: List[str] = Field(default_factory=list)


class DocumentMeta(BaseModel):
    source_file: Optional[str] = None
    page_count: Optional[int] = None


class PatientInfo(BaseModel):
    patient_id: Optional[EvidenceField[str]] = None
    first_name: Optional[EvidenceField[str]] = None
    surname: Optional[EvidenceField[str]] = None


# ------------------------------
# Helper coercion (CRITICAL FIX)
# ------------------------------
def _digits_to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else None
    return None


def coerce_evidence_int(v: Any) -> Optional[Dict[str, Any]]:
    """
    Accepts:
      - "153"
      - 153
      - {"value": "153", "evidence": [...]}
    Returns EvidenceField[int] dict or None.
    """
    if v is None:
        return None

    if isinstance(v, dict):
        # Already evidence-ish
        val = _digits_to_int(v.get("value"))
        ev = v.get("evidence") or []
        if not isinstance(ev, list):
            ev = [str(ev)]
        return {"value": val, "evidence": [str(e) for e in ev if e is not None]}

    # string/int/float -> wrap
    val = _digits_to_int(v)
    return {"value": val, "evidence": []}


class Vitals(BaseModel):
    bp_systolic: Optional[EvidenceField[int]] = None
    bp_diastolic: Optional[EvidenceField[int]] = None
    pulse_rate: Optional[EvidenceField[int]] = None
    ecg_rate: Optional[EvidenceField[int]] = None
    spo2: Optional[EvidenceField[int]] = None

    # ✅ accept str/int/dict and coerce BEFORE EvidenceField parsing
    @field_validator("bp_systolic", "bp_diastolic", "pulse_rate", "ecg_rate", "spo2", mode="before")
    @classmethod
    def clean_numeric(cls, v):
        return coerce_evidence_int(v)

    @field_validator("spo2")
    @classmethod
    def validate_spo2(cls, v):
        if isinstance(v, EvidenceField):
            val = v.value
            if val is not None and not (0 <= val <= 100):
                v.value = None
                v.evidence = []
        return v

    @field_validator("pulse_rate", "ecg_rate")
    @classmethod
    def validate_rates(cls, v):
        if isinstance(v, EvidenceField):
            val = v.value
            if val is not None and not (20 <= val <= 250):
                v.value = None
                v.evidence = []
        return v

    @field_validator("bp_systolic")
    @classmethod
    def validate_sys(cls, v):
        if isinstance(v, EvidenceField):
            val = v.value
            if val is not None and not (60 <= val <= 260):
                v.value = None
                v.evidence = []
        return v

    @field_validator("bp_diastolic")
    @classmethod
    def validate_dia(cls, v):
        if isinstance(v, EvidenceField):
            val = v.value
            if val is not None and not (30 <= val <= 180):
                v.value = None
                v.evidence = []
        return v


class ChiefComplaint(BaseModel):
    text: Optional[EvidenceField[str]] = None


class Notes(BaseModel):
    raw_text: Optional[EvidenceField[str]] = None


class MedicationItem(BaseModel):
    name: EvidenceField[str] = Field(default_factory=EvidenceField)
    dose: EvidenceField[str] = Field(default_factory=EvidenceField)
    route: EvidenceField[str] = Field(default_factory=EvidenceField)


class ClinicalImpressionItem(BaseModel):
    impression: EvidenceField[str] = Field(default_factory=EvidenceField)


class MedicalReport(BaseModel):
    document_meta: Optional[DocumentMeta] = None
    patient_info: Optional[PatientInfo] = None
    vitals: Optional[Vitals] = None
    chief_complaint: Optional[ChiefComplaint] = None
    clinical_impression: List[ClinicalImpressionItem] = Field(default_factory=list)
    medications_administered: List[MedicationItem] = Field(default_factory=list)
    notes: Optional[Notes] = None
    normalization: Optional[Dict[str, Any]] = None


# ==============================
# 7) PHASE 1 PROMPT (SAFER)
# ==============================
PHASE1_PROMPT = """
You are a Medical Data Extraction Specialist. 
Your goal is to extract clinical data VERBATIM, regardless of the document layout.

STRICT EXTRACTION RULES:

1) SEMANTIC MAPPING (Vitals):
   - 'bp_systolic': The first/top number in Blood Pressure (e.g., 153).
   - 'bp_diastolic': The second/bottom number in Blood Pressure (e.g., 95).
   - 'pulse_rate': Physical pulse/HR labels.
   - 'ecg_rate': Electrical/Monitor ECG labels.
   - 'spo2': Oxygen saturation (%SaO2).
   - 'spo2': Search for values associated with oxygen saturation labels. 
     This includes all variations such as: "SpO2", "SPo2", "SaO2", "%SaO2", "O2 Sat" or "Pulse Ox" when located near respiratory data.

2) TABLE EXTRACTION (Medications & Impressions):
   - Identify any section or table containing "Medication", "Drugs", or "Administered". 
   - For each entry, extract the Name, Dosage, and Route.
   - Identify any section containing "Impression", "Assessment", or "Diagnosis". Extract these as a list.
   - Return JSON file only.

3) CLINICAL NARRATIVE (Notes):
   - Locate the main narrative section (often labeled "Notes", "History of Present Illness", or "Narrative").
   - Extract the clinical story. 
   - FILTER: Exclude administrative or logistical phrases (e.g., "Consent obtained", "Arrived at destination", "Mobility assessment").

4) DATA INTEGRITY:
   - If a field is missing, set value to null.
   - EVIDENCE: For every field, capture the exact snippet of text + the surrounding labels to prove the context.

Schema:
{
  "document_meta": { "source_file": "string", "page_count": "int" },
  "patient_info": {
      "patient_id": { "value": "string", "evidence": [] },
      "first_name": { "value": "string", "evidence": [] },
      "surname": { "value": "string", "evidence": [] }
  },
  "vitals": {
    "pulse_rate": { "value": "int", "evidence": [] },
    "ecg_rate": { "value": "int", "evidence": [] },
    "spo2": { "value": "int", "evidence": [] },
    "bp_systolic": { "value": "int", "evidence": [] },
    "bp_diastolic": { "value": "int", "evidence": [] }
  },
  "clinical_impression": [{"impression": {"value": "string", "evidence": []}}],
  "medications_administered": [
    {"name": {"value": "string", "evidence": []}, "dose": {"value": "string", "evidence": []}, "route": {"value": "string", "evidence": []}}
  ],
  "notes": { "raw_text": { "value": "string", "evidence": [] } }
}
"""


# ==============================
# 8) PHASE 1 GENERATION (ASYNC)
# ==============================
async def generate_phase1_json_text_async(
        client: genai.Client,
        models: List[str],
        file_path: str
) -> Optional[str]:

    pdf_text = None
    mode = "vision"

    if file_path.lower().endswith(".pdf"):
        pdf_text_tmp, quality = await asyncio.to_thread(extract_pdf_text_layer, file_path)

        logger.debug(f"[Router] quality={quality:.2f} chars={len(pdf_text_tmp)}")

        if quality >= 0.70 and len(pdf_text_tmp) >= 800:
            pdf_text = pdf_text_tmp
            contents = [PHASE1_PROMPT + "\n\n---\nPDF TEXT LAYER (verbatim):\n" + pdf_text]
            mode = "text"
        else:
            uploaded = await asyncio.to_thread(client.files.upload, file=file_path)
            contents = [uploaded, PHASE1_PROMPT]
            mode = "vision"
    else:
        img = await asyncio.to_thread(Image.open, file_path)
        contents = [img, PHASE1_PROMPT]
        mode = "vision"

    for m in models:
        try:
            logger.info(f"🚀 Phase 1 extraction ({mode}) using {m}")

            resp = await client.aio.models.generate_content(
                model=m,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            return resp.text

        except Exception as e:
            logger.warning(f"⚠️ Model failed: {m} ({type(e).__name__})")

    return None


# ==============================
# 9) POST-CLEANUP / NORMALIZATION
# ==============================
def beautify_phase1(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj = dict(obj)
    vitals = obj.get("vitals") or {}

    # Ensure all vital keys exist
    required_vitals = ["bp_systolic", "bp_diastolic", "pulse_rate", "ecg_rate", "spo2"]
    for v in required_vitals:
        vitals.setdefault(v, {"value": None, "evidence": []})

    obj["vitals"] = vitals
    return obj


def normalize_schema(p1: Dict[str, Any]) -> Dict[str, Any]:
    """
    ✅ Fixes:
      - clinical_impression must ALWAYS be a list of {"impression": {"value":..., "evidence":[...]}}
      - vitals fields might be strings -> wrap into evidence dict so Pydantic accepts
    """
    out = dict(p1)

    # ---- vitals shape fix (string/int -> dict) ----
    vitals = out.get("vitals") or {}
    for k in ["bp_systolic", "bp_diastolic", "pulse_rate", "ecg_rate", "spo2"]:
        vitals[k] = coerce_evidence_int(vitals.get(k))
        if vitals[k] is None:
            vitals[k] = {"value": None, "evidence": []}
    out["vitals"] = vitals

    # ---- clinical_impression shape fix ----
    cis = out.get("clinical_impression")

    normalized_cis: List[Dict[str, Any]] = []

    # If Gemini returned a dict (your error case) -> convert to list
    if isinstance(cis, dict):
        # try common patterns: {"impression": {...}} or {"items":[...]} or {"0":...}
        if "impression" in cis:
            normalized_cis.append(cis)
        elif "items" in cis and isinstance(cis["items"], list):
            cis = cis["items"]
        else:
            cis = []

    if isinstance(cis, list):
        for item in cis:
            if isinstance(item, dict) and "impression" in item and isinstance(item["impression"], dict):
                normalized_cis.append(item)
            elif isinstance(item, dict) and "value" in item:
                # sometimes returns {"value":"...", "evidence":[...]} directly
                normalized_cis.append({"impression": item})
            elif isinstance(item, str):
                normalized_cis.append({"impression": {"value": item, "evidence": []}})

    out["clinical_impression"] = normalized_cis
    return out


def drop_impression_headers(p1: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(p1)
    headers = {"respiratory", "cardiac", "neuro", "neurological", "gi", "msk", "general"}
    cleaned = []
    for item in out.get("clinical_impression", []) or []:
        val = (((item or {}).get("impression") or {}).get("value") or "").strip().lower()
        if val and val not in headers:
            cleaned.append(item)
    out["clinical_impression"] = cleaned
    return out


def apply_phase1_5_normalization(p1: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(p1)
    normalization = {"tier_a_expansions": [], "normalized_fields": {}}

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

    out["normalization"] = normalization
    return out


# ==============================
# 9.5) PHASE 2 (ICD-10 CODING)
# ==============================
PHASE2_PROMPT_TEMPLATE = """
You are a professional Medical Coder.
Your task is to assign ICD-10-CM codes based STRICTLY on the extracted text provided.
Do not infer, assume, or reinterpret clinical meaning beyond what is explicitly documented.

---
SAFE ABBREVIATION REFERENCE GUIDE (UNAMBIGUOUS ONLY):
The following abbreviations are guaranteed to be unambiguous in this context and may be safely interpreted:

- "PVC" / "PVCs" → Premature Ventricular Contractions
- "SOB" → Short Of Breath
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

3. Prefer conservative, unspecified symptom codes when documentation lacks specificity.

4. Code ONLY what is clearly and explicitly documented.

5. If a condition is suspected, ruled out, or not confirmed,
   code the presenting symptom — NOT the suspected diagnosis.

---
PATIENT DATA (JSON):
{patient_data_json}

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


async def run_phase2_coding_async(
    client: genai.Client,
    models: List[str],
    phase1_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    data_str = json.dumps(phase1_data, indent=2, ensure_ascii=False)
    prompt = PHASE2_PROMPT_TEMPLATE.format(patient_data_json=data_str)

    last_err: Optional[Exception] = None

    for m in models:
        try:
            logger.info(f"🧠 Phase 2 coding using {m}")
            resp = await client.aio.models.generate_content(
                model=m,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            return json.loads(resp.text)

        except Exception as e:
            last_err = e
            logger.warning(f"⚠️ Phase 2 failed on {m}: {type(e).__name__}")

    logger.error(f"❌ Phase 2 failed. Last error: {type(last_err).__name__ if last_err else 'Unknown'}")
    return None


# ==============================
# 10) SINGLE FILE PROCESSOR
# ==============================
async def process_single_file(
        file_path: str,
        sem: asyncio.Semaphore,
        client: genai.Client,
        models: List[str]
) -> Optional[Dict[str, Any]]:

    job_id = safe_job_id(file_path)

    async with sem:
        if not os.path.exists(file_path):
            logger.error(f"{job_id}: file missing")
            return None

        json_text = await generate_phase1_json_text_async(client, models, file_path)
        if not json_text:
            logger.error(f"{job_id}: phase1 failed")
            return None

        try:
            raw_obj = json.loads(json_text)

            p1 = beautify_phase1(raw_obj)
            p1 = normalize_schema(p1)
            p1 = drop_impression_headers(p1)
            p1 = apply_phase1_5_normalization(p1)

            validated = MedicalReport(**p1)

            logger.debug(f"{job_id}: validated successfully")

            phase1_out = validated.model_dump(exclude_none=True)

            # --- Phase 2 ---
            phase2_out = await run_phase2_coding_async(client, models, phase1_out)

            if phase2_out:
                phase1_out["icd10_coding"] = phase2_out.get("coding_results", [])
                phase1_out["icd10_coding_raw"] = phase2_out

            return phase1_out

        except ValidationError as e:
            safe_errors = [(err.get("loc"), err.get("msg")) for err in e.errors()]
            logger.error(f"{job_id}: validation error {safe_errors}")
            return None

        except json.JSONDecodeError:
            logger.error(f"{job_id}: invalid JSON")
            return None


# ==============================
# 11) BATCH RUNNER
# ==============================
async def run_batch_job(file_paths: List[str]) -> List[Dict[str, Any]]:
    if not API_KEY:
        logger.critical("GEMINI_API_KEY is missing. Set it as an environment variable.")
        return []

    client = get_client()
    models = choose_model(client, DEFAULT_MODEL_CANDIDATES)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    start_time = time.perf_counter()
    logger.info(f"Starting batch of {len(file_paths)} files...")

    tasks = [process_single_file(fp, sem, client, models) for fp in file_paths]
    results = await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    valid = [r for r in results if r is not None]

    logger.info("=" * 55)
    logger.info(f"✅ BATCH COMPLETE in {end_time - start_time:.2f}s")
    logger.info(f"📄 Success: {len(valid)}/{len(file_paths)}")
    logger.info("=" * 55)

    return valid


# ==============================
# 12) TESTING BLOCK
# ==============================
if __name__ == "__main__":
    test_file_path = r"C:\Users\rieta\OneDrive\Desktop\medical_ocr\WWD2YWN-A-U49.pdf"

    if not API_KEY:
        logger.critical("API Key is missing! Set GEMINI_API_KEY env var.")
    elif not os.path.exists(test_file_path):
        logger.critical("Test file not found.")
    else:
        batch_files = [test_file_path]

        final_data = asyncio.run(run_batch_job(batch_files))

        if final_data:
            output_name = "batch_results.json"
            with open(output_name, "w", encoding="utf-8") as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to disk: {output_name}")
