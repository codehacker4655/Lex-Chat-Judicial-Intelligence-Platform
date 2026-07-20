# ==========================================
# IMPORTS
# ==========================================
import fitz  # PyMuPDF
import re
import json
import os

# ==========================================
# 1. TEXT EXTRACTION & ADVANCED CLEANING
# ==========================================
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_legal_noise(text, case_name="Unknown"):
    """
    Normalizes spatial layouts and scrubs repetitive legal watermarks,
    platform tags, and running headers.
    """
    # [span_0](start_span)[span_1](start_span)Remove standard Indian Kanoon platform watermarks[span_0](end_span)[span_1](end_span)
    text = re.sub(r'Indian Kanoon\s*-\s*http://indiankanoon\.org/doc/\d+/', '', text, flags=re.IGNORECASE)
    
    # Dynamic Header Filtering: Strip out recurring case names from headers
    if case_name and case_name != "Unknown":
        normalized_name = re.escape(case_name[:30])
        text = re.sub(rf'^\s*{normalized_name}.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)

    # Strip out standalone page numbers or residual system margin tags
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Normalize structural spacing
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()

def extract_judgment_body(text):
    """
    Locates the operational core of the document, dropping preliminary 
    listing schedules, counsel appearances, and procedural pages.
    """
    judgment_markers = [r'\bJUDGMENT\b', r'\bORDER\b']
    for marker in judgment_markers:
        match = re.search(marker, text, flags=re.IGNORECASE)
        if match:
            return text[match.end():].strip()
    return text

# ==========================================
# 2. METADATA EXTRACTION LOGIC
# ==========================================
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

def detect_case_type(text):
    match = re.search(r'(CIVIL|CRIMINAL|CONSTITUTIONAL)\s+APPELLATE', text, re.IGNORECASE)
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
        judges = judge_match.group(1).strip().split("202")[0].strip()
        
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

# ==========================================
# 3. ADVANCED SUB-PARAGRAPH CHUNKER
# ==========================================
def split_by_legal_paragraphs(text):
    """
    Separates primary paragraphs and hierarchically nested subsections.
    Maps complex patterns such as: '1.', '17.', '17.10.', or '2(a).'
    """
    pattern = r'(\n\s*(?:\[?\d+(?:\.\d+)*\]?|\b[A-Za-z]\b)\.\s)'
    parts = re.split(pattern, text)

    paragraphs = []
    current_content = ""
    current_para_id = "PREAMBLE"

    for part in parts:
        marker_match = re.match(r'\n\s*\[?(\d+(?:\.\d+)*|[A-Za-z])\]?\.\s', part)
        if marker_match:
            if current_content.strip():
                paragraphs.append((current_para_id, current_content.strip()))
            current_para_id = f"PARA_{marker_match.group(1)}"
            current_content = ""
        else:
            current_content += part

    if current_content.strip():
        paragraphs.append((current_para_id, current_content.strip()))

    return paragraphs

def validate_legal_document(text):
    text_upper = text.upper()
    required_anchors = ["COURT", "JUDGMENT", "VERSUS", "SUPREME COURT OF INDIA"]
    matches = sum(1 for anchor in required_anchors if anchor in text_upper)
    return matches >= 2

# ==========================================
# 4. TARGETED PIPELINE EXECUTION
# ==========================================
def run_ingestion_pipeline(target_pdf_name=None, pdf_folder="data/"):
    """
    Processes a targeted PDF file to generate isolated fine-grained chunks, 
    preventing multi-document data overwrites.
    """
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)

    # If a targeted file is provided, process only that file. Otherwise, run on all files in folder.
    if target_pdf_name:
        pdf_files = [target_pdf_name]
    else:
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.PDF') or f.endswith('.pdf')]
    
    for filename in pdf_files:
        file_path = os.path.join(pdf_folder, filename)
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
            
        try:
            print(f"🚀 Ingesting: {filename}...")
            raw_text = extract_text_from_pdf(file_path)
            
            if not validate_legal_document(raw_text):
                print(f"❌ Skipping {filename}: Invalid Indian Court structure.")
                continue
                
            # Step 1: Pre-extract metadata using standard cleaned layout
            temp_cleaned = re.sub(r'\s+', ' ', raw_text).strip()
            metadata = extract_metadata(temp_cleaned)
            
            # Step 2: Deep Scrubbing & Body Core Segmentation
            fully_cleaned_text = clean_legal_noise(raw_text, case_name=metadata["case_name"])
            judgment_body = extract_judgment_body(fully_cleaned_text)
            
            # Step 3: Run Advanced Sub-Paragraph Splitting
            raw_paragraphs = split_by_legal_paragraphs(judgment_body)
            
            # Step 4: Map meta vectors to chunks with unique IDs
            flat_chunks_db = []
            for idx, (para_id, content) in enumerate(raw_paragraphs):
                if len(content) < 60:  # Enforce structural semantic filter
                    continue
                    
                normalized_content = re.sub(r'\s+', ' ', content).strip()
                flat_chunks_db.append({
                    "chunk_id": idx,
                    "text": normalized_content,
                    "metadata": {
                        "para_id": para_id,
                        "case_title": metadata["case_title"],
                        "case_name": metadata["case_name"],
                        "date": metadata["date"],
                        "year": metadata["year"],
                        "court": metadata["court"],
                        "judges": metadata["judges"],
                        "case_type": metadata["case_type"],
                        "source_file": filename
                    }
                })
            
            # Create a safe unique ID name for output files
            safe_id = re.sub(r'[^a-zA-Z0-9]', '_', filename)
            
            # Save the isolated case chunk JSON strictly mapping only this file's context
            output_chunk_path = f"data/chunks_{safe_id}.json"
            with open(output_chunk_path, "w", encoding="utf-8") as f:
                json.dump(flat_chunks_db, f, indent=2, ensure_ascii=False)
                
            print(f"  --> Isolated Chunks Saved: {output_chunk_path} ({len(flat_chunks_db)} paragraphs)")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    # Standard CLI fallback execution loop
    run_ingestion_pipeline()