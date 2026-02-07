import streamlit as st
import pipeline  # Imports your backend logic
import os
import json
import time

# --- 1. PAGE CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="MediCode AI | Agentic Pipeline",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS FOR "CHIC" LOOK ---
st.markdown("""
    <style>
    /* Main container padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Card Styling for ICD Codes */
    .icd-card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #2e86de; /* Nice Blue Accent */
    }
    .icd-code { color: #2e86de; font-weight: bold; font-size: 1.2em; }
    .icd-desc { color: #333; font-weight: 600; font-size: 1.1em; }
    .evidence { font-style: italic; color: #555; background: #f8f9fa; padding: 4px 8px; border-radius: 4px; }
    
    /* Metrics styling */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #eee;
        padding: 15px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    st.title("MediCode AI")
    st.caption("v1.0.0 | Powered by Gemini 2.0")
    
    st.divider()
    
    st.subheader("📂 Upload Patient Record")
    uploaded_file = st.file_uploader("Drop PDF or Image", type=["pdf", "jpg", "png", "jpeg"])
    
    st.divider()
    
    st.subheader("⚙️ System Status")
    if os.environ.get("GEMINI_API_KEY"):
        st.success("🟢 API Key Active")
    else:
        st.warning("🔴 API Key Missing")
        api_key = st.text_input("Enter Key", type="password")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            st.rerun()

    st.markdown("---")
    st.markdown("**Architecture:**\n\n1️⃣ **OCR Vision:** Extract Text\n2️⃣ **Context Agent:** Expand Abbrv.\n3️⃣ **Coder Agent:** Assign ICD-10")

# --- 4. MAIN PAGE LOGIC ---

# Hero Section
if not uploaded_file:
    st.markdown("## 👋 Welcome to MediCode AI")
    st.markdown("""
    This agentic pipeline automates the translation of paramedic reports into **ICD-10 Billing Codes**.
    
    **Get Started:**
    1. Upload a **Paramedic Report (PDF/Image)** in the sidebar.
    2. Watch the AI extract vitals, demographics, and clinical narratives.
    3. Review the generated coding report.
    """)
    
    # Optional: Show a "Demo" button if you want to preload a file (advanced)
    # if st.button("Load Demo Data"): ...

else:
    # --- FILE PROCESSING ---
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- LAYOUT: 2 Columns (Doc Preview | Results) ---
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("📄 Document Source")
        if uploaded_file.type == "application/pdf":
            st.info("PDF Document Loaded Successfully")
            # If you want to show PDF preview (requires extra library), keep it simple for now:
            st.markdown(f"**Filename:** `{uploaded_file.name}`")
        else:
            st.image(uploaded_file, use_container_width=True, caption="Scanned Report")

    with col2:
        st.subheader("🧠 Analysis Pipeline")
        
        # The Big Blue Button
        if st.button("🚀 Run Agentic Analysis", type="primary", use_container_width=True):
            
            # --- PHASE 1: EXTRACTION ---
            with st.status("🔍 Phase 1: Analyzing Document...", expanded=True) as status:
                st.write("Extracting text via Optical Character Recognition (OCR)...")
                p1_data = pipeline.extract_phase1(file_path)
                
                if p1_data:
                    st.write("Identified Demographics & Vitals...")
                    st.write("Normalizing Medical Abbreviations...")
                    status.update(label="✅ Phase 1 Complete!", state="complete", expanded=False)
                else:
                    status.update(label="❌ Phase 1 Failed", state="error")
                    st.stop()

            # --- PHASE 2: CODING ---
            with st.spinner("💊 Phase 2: Mapping Symptoms to ICD-10 Standards..."):
                p2_data = pipeline.run_phase2_coding(p1_data)
            
            if p2_data:
                # --- SUCCESS DASHBOARD ---
                st.balloons()
                
                # 1. VITALS ROW (The "Pro" Dashboard Look)
                vitals = p1_data.get("patient_info", {})
                v_nums = p1_data.get("vitals", {})
                
                # Safe Extraction helpers
                age = vitals.get("age_years", {}).get("value", "N/A")
                gender = vitals.get("gender", {}).get("value", "N/A")
                bp = f"{v_nums.get('bp_systolic', {}).get('value', '?')}/{v_nums.get('bp_diastolic', {}).get('value', '?')}"
                hr = v_nums.get("pulse_rate", {}).get('value', 'N/A')
                spo2 = v_nums.get("spo2", {}).get('value', 'N/A')

                # Display Vitals
                st.markdown("### 📊 Patient Vitals Snapshot")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Patient Age", f"{age} Y/O", gender)
                m2.metric("Blood Pressure", bp, "mmHg")
                m3.metric("Heart Rate", f"{hr} BPM")
                m4.metric("SpO2", f"{spo2}%")
                
                st.divider()

                # 2. TABBED RESULTS
                tab_coding, tab_clinical, tab_json = st.tabs(["🏥 ICD-10 Coding", "📋 Clinical Summary", "💾 Raw Data"])
                
                with tab_coding:
                    st.markdown("#### Generated Billing Codes")
                    codes = p2_data.get("coding_results", [])
                    
                    if not codes:
                        st.warning("No specific ICD-10 codes could be determined.")
                    
                    for item in codes:
                        code = item.get('icd10_code', 'N/A')
                        desc = item.get('description', 'Unknown')
                        source = item.get('source_text', 'N/A')
                        reason = item.get('reasoning', 'No reasoning provided')
                        
                        # The "Chic" HTML Card
                        st.markdown(f"""
                        <div class="icd-card">
                            <div class="icd-code">{code}</div>
                            <div class="icd-desc">{desc}</div>
                            <div style="margin-top: 8px;">
                                <span style="font-size:0.9em; color:#666;">Source Text:</span>
                                <span class="evidence">{source}</span>
                            </div>
                            <div style="margin-top: 8px; font-size: 0.9em; color: #444;">
                                <b>🤖 AI Reasoning:</b> {reason}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                with tab_clinical:
                    st.markdown("#### Extracted Clinical Narrative")
                    
                    # Notes
                    notes = p1_data.get("notes", {}).get("raw_text", {}).get("value", "No notes found.")
                    st.info(f"**📝 Clinical Notes:**\n\n{notes}")
                    
                    # Chief Complaint
                    cc = p1_data.get("chief_complaint", {}).get("text", {}).get("value", "N/A")
                    st.write(f"**Chief Complaint:** {cc}")
                    
                    # Medications
                    meds = p1_data.get("medications_administered", [])
                    if meds:
                        st.write("**💊 Medications Administered:**")
                        for m in meds:
                            val = m.get("medication", {}).get("value", "Unknown")
                            st.markdown(f"- {val}")

                with tab_json:
                    # Download Button
                    final_report = {"phase1_extraction": p1_data, "phase2_coding": p2_data}
                    json_str = json.dumps(final_report, indent=2)
                    
                    st.download_button(
                        label="📥 Download Full JSON Report",
                        data=json_str,
                        file_name="medical_report.json",
                        mime="application/json"
                    )
                    
                    st.markdown("#### Phase 1: Vision Extraction")
                    st.json(p1_data, expanded=False)
                    st.markdown("#### Phase 2: Reasoning Agent")
                    st.json(p2_data, expanded=False)

            else:
                st.error("Phase 2 Analysis Failed. Please check logs.")
