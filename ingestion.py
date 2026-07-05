# ===============================
# IMPORTS
# ===============================
import fitz  # PyMuPDF
import re
import json
import os

# ===============================
# 1. Extract text from PDF
# ===============================
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ===============================
# 2. Clean text
# ===============================
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)     # normalize newlines
    text = re.sub(r'[ \t]+', ' ', text)   # normalize spaces
    return text.strip()

# ===============================
# 3. Extract Case Name (ROBUST)
# ===============================
def extract_case_name(text):
    text = text.replace("\n", " ")
    match = re.search(
        r"([A-Z0-9 ,./()&-]+?)\s+VERSUS\s+([A-Z0-9 ,./()&-]+)",
        text,
        re.IGNORECASE
    )
    if match:
        left = match.group(1).strip()
        right = match.group(2).strip()
        left = re.sub(r'\b(APPELLANT(S)?|AND ORS\.?)\b', '', left, flags=re.IGNORECASE)
        right = re.sub(r'\b(RESPONDENT(S)?)\b', '', right, flags=re.IGNORECASE)
        left = re.sub(r'\s+', ' ', left)
        right = re.sub(r'\s+', ' ', right)
        return f"{left} vs {right}"
    return "Unknown"

# ===============================
# 4. Detect Case Type (IMPROVED)
# ===============================
def detect_case_type(text):
    match = re.search(
        r'(CIVIL|CRIMINAL|CONSTITUTIONAL)\s+APPELLATE',
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).capitalize()
    text_upper = text.upper()
    if "CRIMINAL" in text_upper:
        return "Criminal"
    elif "CIVIL" in text_upper:
        return "Civil"
    elif "CONSTITUTION" in text_upper:
        return "Constitutional"
    return "Unknown"

# ===============================
# 5. Extract metadata
# ===============================
def extract_metadata(text):
    lines = text.split("\n")
    case_title = lines[0].strip() if lines else "Unknown"
    date_match = re.search(r"\d{1,2} \w+, \d{4}", text)
    date = date_match.group(0) if date_match else "Unknown"
    year_match = re.search(r"\b(19|20)\d{2}\b", text)
    year = year_match.group(0) if year_match else "Unknown"
    court = "Unknown"
    if "SUPREME COURT OF INDIA" in text.upper():
        court = "Supreme Court of India"
    judge_match = re.search(r"Bench:\s*([A-Za-z., ]+)", text)
    judges = "Unknown"
    if judge_match:
        judges = judge_match.group(1).strip()
        judges = judges.split("202")[0]  
    case_name = extract_case_name(text)
    case_type = detect_case_type(text)
    return {
        "case_title": case_title,
        "case_name": case_name,
        "date": date,
        "year": year,
        "court": court,
        "judges": judges,
        "case_type": case_type
    }

# ===============================
# 6. Extract Judgment Text
# ===============================
def extract_judgment_text(text):
    if "JUDGMENT" in text:
        return text.split("JUDGMENT", 1)[1].strip()
    return text

# ===============================
# 7. NEW PRODUCT FEATURE: Hierarchical Layout Chunker
# ===============================
def slice_into_identifiable_paragraphs(judgment_text):
    """
    Programmatically segments continuous judgment text streams into discrete, 
    numbered paragraph blocks while filtering out legal headers and footer noise.
    """
    # Regex designed to capture standard Indian judicial paragraph enumeration formats (e.g., '1. ', '[2] ', '3(a). ')
    para_split_pattern = r'\n\s*(?:\[?\d+\]?[\.\)]|\b[A-Za-z]\b\.)\s*'
    
    raw_chunks = re.split(para_split_pattern, judgment_text)
    markers = re.findall(para_split_pattern, judgment_text)
    
    structured_chunks = []
    
    # Catch any preliminary text before the first paragraph index marker
    if raw_chunks and raw_chunks[0].strip():
        first_clean = re.sub(r'\s+', ' ', raw_chunks[0]).strip()
        if len(first_clean) > 30:  # Skip trivial residual fragments
            structured_chunks.append({
                "para_id": "PREAMBLE",
                "text": first_clean
            })
            
    # Zip together extracted text blocks with their exact structural index keys
    for i, chunk_text in enumerate(raw_chunks[1:]):
        clean_chunk = re.sub(r'\s+', ' ', chunk_text).strip()
        if len(clean_chunk) > 40:  # Enforce structural weight bounds
            # Extract clean alphanumeric paragraph marker strings
            para_marker = markers[i].strip().replace("[", "").replace("]", "").replace(".", "")
            structured_chunks.append({
                "para_id": f"PARA_{para_marker}",
                "text": clean_chunk
            })
            
    return structured_chunks

# ===============================
# 8. Process PDFs & Build Relational Database JSON
# ===============================
def run_ingestion_pipeline(pdf_folder="data/"):
    dataset = []
    flat_chunks_db = [] # Dynamic array to facilitate direct FAISS ingestion downstream
    
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        print(f"Please put your PDFs in the '{pdf_folder}' folder.")
        return

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.PDF')]
    
    for filename in pdf_files:
        print(f"Processing: {filename}")
        file_path = os.path.join(pdf_folder, filename)
        try:
            raw_text = extract_text_from_pdf(file_path)
            cleaned = clean_text(raw_text)
            metadata = extract_metadata(cleaned)
            judgment_text = extract_judgment_text(cleaned)

            # Assign properties
            case_data = metadata.copy()
            case_data["source_file"] = filename
            
            # Apply our hierarchical chunking engine
            paragraph_blocks = slice_into_identifiable_paragraphs(judgment_text)
            case_data["paragraphs"] = paragraph_blocks
            dataset.append(case_data)
            
            # Map structural relations globally to build an analytical database trace
            for block in paragraph_blocks:
                flat_chunks_db.append({
                    "text": block["text"],
                    "metadata": {
                        "source_file": filename,
                        "case_name": metadata["case_name"],
                        "para_id": block["para_id"],
                        "year": metadata["year"],
                        "court": metadata["court"]
                    }
                })
                
        except Exception as e:
            print(f"Error processing document {filename}: {e}")

    # Output Layer 1: Hierarchical Case relational view
    with open("data/processed_data.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    # Output Layer 2: Flat relational file configured for direct vector engine consumption
    with open("data/legal_chunks_ready.json", "w", encoding="utf-8") as f:
        json.dump(flat_chunks_db, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Processing complete!")
    print(f"  --> Case database stored: data/processed_data.json ({len(dataset)} entries)")
    print(f"  --> Fine-grained chunks mapped: data/legal_chunks_ready.json ({len(flat_chunks_db)} vectors ready)")

if __name__ == "__main__":
    run_ingestion_pipeline()