import streamlit as st
from pipeline import extract_hybrid_content

st.set_page_config(page_title="PDF Hybrid Extractor", layout="wide")

st.title("📄 PDF Extractor (Text + OCR)")
st.markdown("""
This tool extracts **digital text** instantly and uses **OCR** to read text inside images/charts.
""")

# File Uploader
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF (Scanning text & images)..."):
        try:
            # Call the function from pipeline.py
            # We pass the file object directly
            extracted_text = extract_hybrid_content(uploaded_file)
            
            st.success("Extraction Complete!")
            
            # Show previews
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Raw Output Preview")
                st.text_area("Content", extracted_text, height=600)
            
            with col2:
                st.subheader("Download")
                st.download_button(
                    label="Download Text File",
                    data=extracted_text,
                    file_name="extracted_content.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
