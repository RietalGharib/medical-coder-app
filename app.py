import streamlit as st
import pipeline
import os
import json

st.set_page_config(page_title="Paramedic AI Coder", page_icon="🚑", layout="wide")

st.title("🚑 Agentic Medical Coder")
st.markdown("This tool processes medical PDFs and assigns ICD-10 codes using AI.")

with st.sidebar:
    st.header("🔑 Configuration")
    user_api_key = st.text_input("Enter Hugging Face API Token (hf_...)", type="password")

    st.caption("Tip: Create a Read token at https://huggingface.co/settings/tokens")

    if not user_api_key:
        st.warning("⚠️ You must enter a Hugging Face token to run the analysis.")
    else:
        st.success("Token ready!")

uploaded_file = st.file_uploader("Upload Medical Report (PDF)", type=["pdf"])

if uploaded_file:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.toast("File Uploaded Successfully", icon="✅")

    if st.button("Run Analysis Pipeline", type="primary"):
        if not user_api_key:
            st.error("❌ Please enter your Hugging Face token in the sidebar first.")
            st.stop()

        # --- PHASE 1 ---
        with st.spinner("Phase 1: Extracting Clinical Data..."):
            p1 = pipeline.extract_phase1(file_path, api_key=user_api_key)

        if not p1["ok"]:
            st.error(f"Phase 1 failed: {p1['error']}")
            if p1.get("model_used"):
                st.caption(f"Model attempted: {p1['model_used']}")
            st.stop()

        p1_data = p1["data"]
        st.success("Phase 1 complete!")
        st.caption(f"Model used: {p1.get('model_used')}")

        with st.expander("View Extracted Data", expanded=False):
            st.json(p1_data)

        # --- PHASE 2 ---
        with st.spinner("Phase 2: Assigning ICD-10 Codes..."):
            p2 = pipeline.run_phase2_coding(p1_data, api_key=user_api_key)

        if not p2["ok"]:
            st.error(f"Phase 2 failed: {p2['error']}")
            if p2.get("model_used"):
                st.caption(f"Model attempted: {p2['model_used']}")
            st.stop()

        p2_data = p2["data"]
        st.balloons()
        st.success("Coding Complete!")
        st.caption(f"Model used: {p2.get('model_used')}")

        for item in p2_data.get("coding_results", []):
            st.markdown(
                f"""
                <div style="background:#f0f2f6;padding:10px;border-radius:5px;border-left:4px solid #ff4b4b;margin-bottom:10px">
                    <b>{item.get('icd10_code')}</b>: {item.get('description')}<br/>
                    <small><b>Source:</b> {item.get('source_text','')}</small><br/>
                    <small><b>Reasoning:</b> {item.get('reasoning','')}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        full_report = json.dumps({"patient": p1_data, "coding": p2_data}, indent=2)
        st.download_button("Download JSON", full_report, "report.json", "application/json")
