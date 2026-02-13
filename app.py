import streamlit as st
import os
import json
import asyncio
import tempfile
import logging
from pypdf import PdfReader
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Optional, Dict, Any, Generic, TypeVar

# ==============================
# 1) CONFIG & LOGGING
# ==============================
st.set_page_config(page_title="MediCode AI", layout="wide", page_icon="🏥")

# Setup logging to show in console (Streamlit logs)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================
# 2) DATA MODELS (Your Existing Schema)
# ==============================
T = TypeVar("T")

class EvidenceField(BaseModel, Generic[T]):
    value: Optional[T] = None
    evidence: List[str] = Field(default_factory=list)

class Vitals(BaseModel):
    bp_systolic: Optional[EvidenceField[int]] = None
    bp_diastolic: Optional[EvidenceField[int]] = None
    pulse_rate: Optional[EvidenceField[int]] = None
    ecg_rate: Optional[EvidenceField[int]] = None
    spo2: Optional[EvidenceField[int]] = None

    @field_validator("bp_systolic", "bp_diastolic", "pulse_rate", "ecg_rate", "spo2", mode="before")
    @classmethod
    def clean_numeric(cls, v):
        if isinstance(v, dict) and "value" in v:
            val = str(v.get("value") or "")
            digits = "".join(ch for ch in val if ch.isdigit())
            v["value"] = int(digits) if digits else None
        return v

class MedicationItem(BaseModel):
    name: EvidenceField[str] = Field(default_factory=EvidenceField)
    dose: EvidenceField[str] = Field(default_factory=EvidenceField)
    route: EvidenceField[str] = Field(default_factory=EvidenceField)

class ClinicalImpressionItem(BaseModel):
    impression: EvidenceField[str] = Field(default_factory=EvidenceField)

class ICD10Code(BaseModel):
    icd10_code: str
    description: str
    source_text: str
    reasoning: str

class ICD10Result(BaseModel):
    coding_results: List[ICD10Code] = Field(default_factory=list)

class MedicalReport(BaseModel):
    document_meta: Optional[Dict[str, Any]] = None
    patient_info: Optional[Dict[str, Any]] = None
    vitals: Optional[Vitals] = None
    chief_complaint: Optional[EvidenceField[str]] = None
    clinical_impression: List[ClinicalImpressionItem] = Field(default_factory=list)
    medications_administered: List[MedicationItem] = Field(default_factory=list)
    notes: Optional[EvidenceField[str]] = None
    normalization: Optional[Dict[str, Any]] = None
    coding: Optional[ICD10Result] = None

# ==============================
# 3) PROMPTS
# ==============================
PHASE1_PROMPT = """
You are a Medical Data Extraction Specialist. Extract clinical data VERBATIM.

1) SEMANTIC MAPPING (Vitals):
   - 'bp_systolic' / 'bp_diastolic': Extract BP (e.g., 153/95).
   - 'pulse_rate': Map labels like "Pulse", "HR", or "Mechanical Pulse". 
   - 'ecg_rate': Map labels specifically like "ECG", "Monitor Rate", or "Electrical Rate".
   - 'spo2': Map ALL variations: "SpO2", "SPo2", "SaO2", "%SaO2", "O2 Sat", "Pulse Ox".
   - MULTI-VALUE: In a cell with two numbers (e.g., 83 77), use the label to pair them correctly.

2) TABLE EXTRACTION:
   - Identify "Medication Administered" and "Clinical Impression" tables.
   - For Impressions, return: {"impression": {"value": "...", "evidence": [...]}}

3) CLINICAL NARRATIVE:
   - Extract narrative from "Notes". Filter out logistics (e.g., "mobility assessment").
"""

PHASE2_PROMPT_TEMPLATE = """
You are a professional Medical Coder.
Your task is to assign ICD-10-CM codes based STRICTLY on the extracted text provided.

SAFE ABBREVIATIONS: PVC, SOB, SpO2, GCS, ECG, IV, NS, ETCO2, RR, HR, BP.
RULES: Do not infer subtypes. Prefer conservative codes (e.g., R07.4 for chest discomfort).

PATIENT DATA (JSON):
{patient_data_json}

REQUIRED OUTPUT FORMAT (JSON ONLY):
{{
  "coding_results": [
    {{
      "icd10_code": "string",
      "description": "string",
      "source_text": "string",
      "reasoning": "string"
    }}
  ]
}}
"""

# ==============================
# 4) ASYNC PIPELINE LOGIC
# ==============================
async def run_pipeline(api_key, file_path, status_container):
    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.0-flash-exp" # Or gemini-2.5-flash

    try:
        # --- PHASE 1: EXTRACTION ---
        status_container.info("🚀 Phase 1: Extracting Clinical Data...")
        
        # Read text layer for cheap processing if possible
        reader = PdfReader(file_path)
        text_layer = "\n".join([p.extract_text() or "" for p in reader.pages])
        
        # Call Gemini (Async)
        resp = await client.aio.models.generate_content(
            model=model_name,
            contents=[PHASE1_PROMPT, text_layer],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
        )
        p1_raw = json.loads(resp.text)
        
        # --- PHASE 2: CODING ---
        status_container.info("🧠 Phase 2: Assigning ICD-10 Codes...")
        
        prompt_p2 = PHASE2_PROMPT_TEMPLATE.format(patient_data_json=json.dumps(p1_raw))
        resp_p2 = await client.aio.models.generate_content(
            model=model_name,
            contents=[prompt_p2],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
        )
        p2_raw = json.loads(resp_p2.text)
        
        # Merge
        p1_raw["coding"] = p2_raw
        
        # Validate
        report = MedicalReport(**p1_raw)
        status_container.success("✅ Processing Complete!")
        
        return report.model_dump(exclude_none=True)

    except Exception as e:
        status_container.error(f"Error: {str(e)}")
        return None

# ==============================
# 5) STREAMLIT UI
# ==============================
def main():
    st.title("🏥 MediCode AI: EMS to ICD-10")
    st.markdown("Upload an EMS PDF report to extract clinical data and generate billing codes.")

    # API Key Handling (Secrets or Input)
    api_key = None
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.text_input("Enter Google API Key:", type="password")

    uploaded_file = st.file_uploader("Upload Report (PDF)", type=["pdf"])

    if uploaded_file and api_key:
        if st.button("Analyze Report"):
            # Save to temp file because Google SDK needs a path or file-like object
            # and it's safer to handle PDF libraries with a physical file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Run Async Loop
                status_box = st.empty()
                result = asyncio.run(run_pipeline(api_key, tmp_path, status_box))

                if result:
                    # --- DISPLAY RESULTS ---
                    tab1, tab2, tab3 = st.tabs(["📄 ICD-10 Codes", "🩺 Clinical Data", "🔍 Raw JSON"])

                    with tab1:
                        st.subheader("Billing Codes")
                        codes = result.get("coding", {}).get("coding_results", [])
                        if codes:
                            for c in codes:
                                with st.expander(f"**{c.get('icd10_code')}** - {c.get('description')}"):
                                    st.write(f"**Reasoning:** {c.get('reasoning')}")
                                    st.caption(f"Source: \"{c.get('source_text')}\"")
                        else:
                            st.warning("No ICD-10 codes found.")

                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Vitals")
                            vitals = result.get("vitals", {})
                            st.metric("BP", f"{vitals.get('bp_systolic', {}).get('value')}/{vitals.get('bp_diastolic', {}).get('value')}")
                            st.metric("Pulse", vitals.get('pulse_rate', {}).get('value'))
                            st.metric("SpO2", f"{vitals.get('spo2', {}).get('value')}%")
                        
                        with col2:
                            st.subheader("Impressions")
                            imps = result.get("clinical_impression", [])
                            for i in imps:
                                st.info(i.get("impression", {}).get("value"))

                    with tab3:
                        st.json(result)
                        
                    # Download Button
                    st.download_button(
                        label="Download JSON Report",
                        data=json.dumps(result, indent=2),
                        file_name="medical_report.json",
                        mime="application/json"
                    )

            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

if __name__ == "__main__":
    main()
