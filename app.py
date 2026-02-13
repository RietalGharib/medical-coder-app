import os
import json
import tempfile
import streamlit as st
import asyncio

# ✅ import your pipeline (the fixed one)
import pipeline

st.set_page_config(page_title="MediCode AI", layout="wide", page_icon="🏥")

def run_async(coro):
    """
    Streamlit-safe async runner.
    - If no running loop: asyncio.run
    - If loop exists (some environments): create task and wait
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Fallback: run in a new loop via asyncio.run in a thread-like approach
        # simplest reliable approach for Streamlit:
        return asyncio.run(coro)
    else:
        return asyncio.run(coro)

def main():
    st.title("🏥 MediCode AI: EMS to ICD-10")
    st.markdown("Upload an EMS PDF report to extract clinical data and generate billing codes.")

    # ✅ Use one key name everywhere
    api_key = None
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        api_key = st.text_input("Enter Gemini API Key:", type="password")

    # Put key where pipeline expects it
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    uploaded_file = st.file_uploader("Upload Report (PDF)", type=["pdf"])

    if uploaded_file and api_key:
        if st.button("Analyze Report"):
            status_box = st.empty()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                status_box.info("🚀 Running extraction + coding pipeline...")

                # ✅ Call pipeline (which includes normalization + validation + phase2)
                results = run_async(pipeline.run_batch_job([tmp_path]))

                if not results:
                    status_box.error("No valid output returned (validation failed or model error).")
                    return

                result = results[0]
                status_box.success("✅ Processing Complete!")

                # --- DISPLAY RESULTS ---
                tab1, tab2, tab3 = st.tabs(["📄 ICD-10 Codes", "🩺 Clinical Data", "🔍 Raw JSON"])

                with tab1:
                    st.subheader("Billing Codes")
                    codes = result.get("icd10_coding", []) or []
                    if codes:
                        for c in codes:
                            with st.expander(f"**{c.get('icd10_code')}** - {c.get('description')}"):
                                st.write(f"**Reasoning:** {c.get('reasoning')}")
                                st.caption(f'Source: "{c.get("source_text")}"')
                    else:
                        st.warning("No ICD-10 codes found.")

                with tab2:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Vitals")
                        vitals = result.get("vitals", {}) or {}

                        sys_v = (vitals.get("bp_systolic") or {}).get("value")
                        dia_v = (vitals.get("bp_diastolic") or {}).get("value")
                        pulse_v = (vitals.get("pulse_rate") or {}).get("value")
                        spo2_v = (vitals.get("spo2") or {}).get("value")

                        st.metric("BP", f"{sys_v}/{dia_v}" if sys_v is not None and dia_v is not None else "—")
                        st.metric("Pulse", pulse_v if pulse_v is not None else "—")
                        st.metric("SpO2", f"{spo2_v}%" if spo2_v is not None else "—")

                    with col2:
                        st.subheader("Impressions")
                        imps = result.get("clinical_impression", []) or []
                        if imps:
                            for i in imps:
                                st.info(((i.get("impression") or {}).get("value")) or "—")
                        else:
                            st.write("—")

                with tab3:
                    st.json(result)

                st.download_button(
                    label="Download JSON Report",
                    data=json.dumps(result, indent=2, ensure_ascii=False),
                    file_name="medical_report.json",
                    mime="application/json"
                )

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

if __name__ == "__main__":
    main()
