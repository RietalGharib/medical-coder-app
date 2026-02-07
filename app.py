import streamlit as st
import pipeline  # Imports your new backend
import os
import json
import shutil

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Paramedic AI Coder",
    page_icon="🚑",
    layout="wide"
)

# --- TITLE & SIDEBAR ---
st.title("🚑 Agentic Medical Coder")
st.markdown("""
**System Architecture:**
1.  **Phase 1 (Hybrid Extraction):** Text Layer + OCR (Tesseract) -> JSON Structure (Gemini)
2.  **Phase 2 (Reasoning):** Contextual Medical Coding (ICD-10)
""")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    # This allows you to paste the key in the UI instead of hardcoding it
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    if api_key:
        # BRIDGE: Update the variable in pipeline.py dynamically
        pipeline.API_KEY = api_key
        st.success("API Key Loaded!")
    elif not pipeline.API_KEY or "PASTE_YOUR" in pipeline.API_KEY:
        st.warning("⚠️ Please enter an API Key to proceed.")

    st.divider()
    st.info("Upload a PDF to extract patient data and generate ICD-10 codes.")

# --- MAIN LOGIC ---
uploaded_file = st.file_uploader("Upload Medical Report", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Save uploaded file to temp disk 
    # (This ensures PyMuPDF and Tesseract can read it reliably as a file path)
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.toast(f"File uploaded: {uploaded_file.name}", icon="✅")

    # Layout: Split Screen
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Document Preview")
        # Basic file details since Streamlit can't natively preview PDF text easily
        st.info(f"**Filename:** {uploaded_file.name}\n\n**Size:** {uploaded_file.size / 1024:.2f} KB")
        
        # If it's an image, show it. If PDF, show icon.
        if uploaded_file.type in ["application/pdf"]:
            st.markdown("⚠️ *PDF Preview is handled by the extraction engine.*")
        else:
            st.image(uploaded_file, caption="Paramedic Report", use_container_width=True)

    with col2:
        st.subheader("🧠 Agentic Analysis")
        run_btn = st.button("Run Analysis Pipeline", type="primary", use_container_width=True)

        if run_btn:
            if not api_key and "PASTE_YOUR" in pipeline.API_KEY:
                 st.error("Please enter an API Key in the sidebar first.")
            else:
                # --- PHASE 1: EXTRACTION ---
                with st.spinner("Phase 1: Hybrid OCR & Extraction..."):
                    # We pass the FILE PATH to the pipeline
                    p1_data = pipeline.extract_phase1(file_path)
                
                if p1_data:
                    st.success("Phase 1 Complete!")
                    
                    # Show Phase 1 Data in an Expandable Box
                    with st.expander("View Extracted Patient Data (JSON)", expanded=False):
                        st.json(p1_data)
                    
                    # --- PHASE 2: CODING ---
                    with st.spinner("Phase 2: Assigning ICD-10 Codes..."):
                        p2_data = pipeline.run_phase2_coding(p1_data)
                    
                    if p2_data:
                        st.balloons()
                        st.success("Phase 2 Complete! Codes Assigned.")
                        
                        # --- DISPLAY RESULTS NICELY ---
                        codes = p2_data.get("coding_results", [])
                        
                        if not codes:
                            st.warning("No ICD-10 codes were applicable for this case.")
                        
                        for item in codes:
                            code = item.get('icd10_code', 'N/A')
                            desc = item.get('description', 'No description')
                            reason = item.get('reasoning', '')
                            source = item.get('source_text', '')
                            
                            # Custom HTML Card for each code
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #ff4b4b; border: 1px solid #ddd;">
                                <h3 style="margin:0; color: #d63384;">{code}</h3>
                                <p style="margin:0; font-weight:bold; color: #333;">{desc}</p>
                                <hr style="margin: 8px 0;">
                                <p style="font-size: 0.9em; color: #555;"><b>Evidence:</b> "{source}"</p>
                                <p style="font-size: 0.9em; color: #0d6efd;"><b>Reasoning:</b> {reason}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # --- DOWNLOAD BUTTON ---
                        final_json = json.dumps({"patient_data": p1_data, "coding_results": p2_data}, indent=2)
                        st.download_button(
                            label="📥 Download Full Medical JSON",
                            data=final_json,
                            file_name="final_medical_report.json",
                            mime="application/json"
                        )
                    else:
                        st.error("Phase 2 (Coding) returned no data.")
                else:
                    st.error("Phase 1 (Extraction) failed. The document might be empty or unreadable.")
