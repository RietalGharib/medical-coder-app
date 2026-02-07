import streamlit as st
import pipeline  # This imports your backend logic
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
1.  **Phase 1 (Vision):** OCR & Data Extraction (Gemini 2.0/1.5)
2.  **Phase 2 (Reasoning):** Contextual Medical Coding (ICD-10)
""")

with st.sidebar:
    st.header("Upload Report")
    uploaded_file = st.file_uploader("Drop PDF or Image here", type=["pdf", "jpg", "png", "jpeg"])
    
    # API Key Handling (Optional: Let user override)
    if not os.environ.get("GEMINI_API_KEY"):
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            # Reload pipeline to pick up key
            import importlib
            importlib.reload(pipeline)

# --- MAIN LOGIC ---
if uploaded_file:
    # 1. Save uploaded file to disk (Pipeline expects a file path)
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File uploaded: {uploaded_file.name}")

    # Display the file (Image or PDF warning)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Document Source")
        if uploaded_file.type == "application/pdf":
            st.info("PDF Preview not supported in this view, but file is processed.")
        else:
            st.image(uploaded_file, caption="Paramedic Report", use_container_width=True)

    with col2:
        st.subheader("🧠 Agentic Analysis")
        run_btn = st.button("Run Analysis Pipeline", type="primary")

        if run_btn:
            with st.spinner("Phase 1: Extracting Clinical Data..."):
                # CALL YOUR PIPELINE
                p1_data = pipeline.extract_phase1(file_path)
            
            if p1_data:
                # Show Phase 1 Data in an Expandable Box
                with st.expander("✅ Phase 1: Extracted Data (JSON)", expanded=False):
                    st.json(p1_data)
                
                with st.spinner("Phase 2: Assigning ICD-10 Codes..."):
                    p2_data = pipeline.run_phase2_coding(p1_data)
                
                if p2_data:
                    st.success("Analysis Complete!")
                    
                    # --- DISPLAY RESULTS NICELY ---
                    codes = p2_data.get("coding_results", [])
                    
                    # Create a clean UI for each code
                    for item in codes:
                        code = item.get('icd10_code', 'N/A')
                        desc = item.get('description', 'No description')
                        reason = item.get('reasoning', '')
                        source = item.get('source_text', '')
                        
                        st.markdown(f"""
                        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #ff4b4b;">
                            <h3 style="margin:0; color: #31333F;">{code} - {desc}</h3>
                            <p style="margin-top:5px;"><b>Evidence:</b> <i>"{source}"</i></p>
                            <p style="font-size: 0.9em; color: #555;"><b>AI Reasoning:</b> {reason}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Download Button
                    final_json = json.dumps({"patient": p1_data, "billing": p2_data}, indent=2)
                    st.download_button("Download Full Report JSON", final_json, "medical_report.json", "application/json")
                    
                else:
                    st.error("Phase 2 (Coding) Failed.")
            else:
                st.error("Phase 1 (Extraction) Failed.")

    # Cleanup temp files (Optional)
    # shutil.rmtree(temp_dir)
