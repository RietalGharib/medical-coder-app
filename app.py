import streamlit as st
import pipeline  # Imports the backend
import os
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Paramedic AI Coder", page_icon="🚑", layout="wide")

st.title("🚑 Agentic Medical Coder")
st.markdown("This tool processes medical PDFs and assigns ICD-10 codes using AI.")

# --- SIDEBAR: KEY INPUT ---
with st.sidebar:
    st.header("🔑 Configuration")
    
    # 1. Get Key from User
    user_api_key = st.text_input("Enter Gemini API Key", type="password")
    
    if not user_api_key:
        st.warning("⚠️ You must enter an API Key to run the analysis.")
    else:
        st.success("Key ready!")

# --- MAIN LOGIC ---
uploaded_file = st.file_uploader("Upload Medical Report (PDF)", type=["pdf"])

if uploaded_file:
    # Save temp file
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.toast("File Uploaded Successfully", icon="✅")

    if st.button("Run Analysis Pipeline", type="primary"):
        
        if not user_api_key:
            st.error("❌ Please enter your API Key in the sidebar first.")
        else:
            try:
                # --- PHASE 1 ---
                with st.spinner("Phase 1: Extracting Clinical Data..."):
                    # Pass key explicitly to the function
                    p1_data = pipeline.extract_phase1(file_path, api_key=user_api_key)
                
                if p1_data:
                    with st.expander("View Extracted Data"):
                        st.json(p1_data)
                    
                    # --- PHASE 2 ---
                    with st.spinner("Phase 2: Assigning ICD-10 Codes..."):
                        # Pass key explicitly to the function
                        p2_data = pipeline.run_phase2_coding(p1_data, api_key=user_api_key)
                    
                    if p2_data:
                        st.balloons()
                        st.success("Coding Complete!")
                        
                        # Display Codes
                        for item in p2_data.get("coding_results", []):
                            st.markdown(f"""
                            <div style="background:#f0f2f6;padding:10px;border-radius:5px;border-left:4px solid #ff4b4b;margin-bottom:10px">
                                <b>{item.get('icd10_code')}</b>: {item.get('description')}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Download
                        full_report = json.dumps({"patient": p1_data, "coding": p2_data}, indent=2)
                        st.download_button("Download JSON", full_report, "report.json", "application/json")
                    else:
                        st.error("Phase 2 Failed.")
                else:
                    st.error("Phase 1 Failed (Extraction returned empty).")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
