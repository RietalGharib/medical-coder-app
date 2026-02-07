import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# --- CONFIGURATION ---
# If you are on Windows local, you might need to uncomment and set this:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_hybrid_content(uploaded_file):
    """
    Inputs: 
        uploaded_file: The file object from Streamlit (st.file_uploader)
    Returns: 
        A single string containing all text (Digital + OCR).
    """
    
    # Read the stream from Streamlit into PyMuPDF
    # "stream" expects bytes, "filetype" tells it it's a PDF
    bytes_data = uploaded_file.read()
    doc = fitz.open(stream=bytes_data, filetype="pdf")
    
    full_text = []

    for page_num, page in enumerate(doc):
        # 1. HARDCODE: Extract digital text (Fast & Accurate)
        text_layer = page.get_text()
        
        # 2. OCR: Extract images
        image_list = page.get_images(full=True)
        ocr_text_list = []
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Load into PIL
                image = Image.open(io.BytesIO(image_bytes))
                
                # Filter small icons/noise (Adjust 100x100 if needed)
                if image.width < 100 or image.height < 100:
                    continue
                
                # Run OCR
                text = pytesseract.image_to_string(image, lang='eng')
                if text.strip():
                    ocr_text_list.append(f"[Image Text]: {text.strip()}")
            except Exception:
                continue # Skip image if OCR fails

        # 3. MERGE
        page_content = f"--- Page {page_num + 1} ---\n{text_layer}\n"
        if ocr_text_list:
            page_content += "\n--- Image Content (OCR) ---\n" + "\n".join(ocr_text_list)
            
        full_text.append(page_content)

    return "\n".join(full_text)
