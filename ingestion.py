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
    # 1. Scrub common platform watermarks
    text = re.sub(r'Indian Kanoon\s*-\s*http://[^\s]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://[^\s]+', '', text)
    
    # 2. Universal Citation Reassembly
    reporter_pattern = r'(\b(?:SCC|SCR|AIR|SCALE|JT|INSC|ILR|MANU(?:/[A-Z]+)?)\s*(?:\(\d+\))?)\s*\n\s*(\d+)\.'
    text = re.sub(reporter_pattern, r'\1 \2.', text, flags=re.IGNORECASE)

    # 3. Strip running headers and document titles
    if case_name and case_name != "Unknown":
        normalized_name = re.escape(case_name[:25])
        text = re.sub(rf'^\s*{normalized_name}.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)

    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()

def extract_judgment_body(text):
    """
    Uses strict multiline header anchors to locate the true judgment body.
    """
    body_pattern = r'\n\s*(?:J\s*U\s*D\s*G\s*M\s*E\s*N\s*T|O\s*R\s*D\s*E\s*R|JUDGMENT|ORDER)\s*\n'
    matches = list(re.finditer(body_pattern, text, flags=re.IGNORECASE))
    
    if matches:
        target_match = matches[0]
        for m in matches:
            if m.start() < len(text) * 0.35:
                target_match = m
        return text[target_match.end():].strip()
        
    return text

# ==========================================
# 2. METADATA EXTRACTION LOGIC
# ==========================================
def extract_case_name(text):
    single_line_text = text.replace("\n", " ")
    match = re.search(
        r"([A-Z0-9 ,./()&-]+?)\s+VERSUS\s+([A-Z0-9 ,./()&-]+)",
        single_line_text,
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

def extract_metadata(raw_text):
    """
    Takes original raw_text (preserving newlines) to accurately grab title line.
    """
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    case_title = lines[0][:100] if lines else "Unknown" # Limit title length to 100 chars
    
    date_match = re.search(r"\d{1,2} \w+, \d{4}", raw_text)
    date = date_match.group(0) if date_match else "Unknown"
    
    year_match = re.search(r"\b(19|20)\d{2}\b", raw_text)
    year = year_match.group(0) if year_match else "Unknown"
    
    court = "Unknown"
    if "SUPREME COURT OF INDIA" in raw_text.upper():
        court = "Supreme Court of India"
        
    judge_match = re.search(r"Bench:\s*([A-Za-z., ]+)", raw_text)
    judges = "Unknown"
    if judge_match:
        judges = judge_match.group(1).strip().split("202")[0].strip()
        
    case_name = extract_case_name(raw_text)
    case_type = detect_case_type(raw_text)
    
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
# 3. ADVANCED LEGAL PARAGRAPH CHUNKER
# ==========================================
def split_by_legal_paragraphs(text):
    pattern = r'(\n\s*(?:\[?\d+(?:\.\d+)*\]?|\bPara(?:graph)?\s*\d+|\(\d+\)|\b[A-Za-z]\b)\.\s)'
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    paragraphs = []
    current_content = ""
    
    current_main_para_num = 0
    current_main_para_id = "PREAMBLE"
    current_active_id = "PREAMBLE"

    for part in parts:
        marker_match = re.match(
            r'\n\s*(?:\[?(\d+(?:\.\d+)*)\]?|\bPara(?:graph)?\s*(\d+)|\((\d+)\)|([A-Za-z]))\.\s', 
            part, 
            flags=re.IGNORECASE
        )

        if marker_match:
            raw_id_str = next(g for g in marker_match.groups() if g is not None)

            if raw_id_str.isdigit():
                num_val = int(raw_id_str)

                # STRICT MONOTONIC CHECK (+1 or +2 max)
                is_valid_next_para = (
                    current_main_para_num == 0 or 
                    num_val == current_main_para_num + 1 or 
                    num_val == current_main_para_num + 2
                )

                if is_valid_next_para:
                    if current_content.strip():
                        paragraphs.append((current_active_id, current_content.strip()))
                        current_content = ""
                    
                    current_main_para_num = num_val
                    current_main_para_id = f"PARA_{num_val}"
                    current_active_id = current_main_para_id

                else:
                    if current_content.strip():
                        paragraphs.append((current_active_id, current_content.strip()))
                        current_content = ""
                    current_active_id = f"{current_main_para_id}_SUB_{num_val}"

            else:
                if current_content.strip():
                    paragraphs.append((current_active_id, current_content.strip()))
                    current_content = ""
                current_active_id = f"{current_main_para_id}_SUB_{raw_id_str.upper()}"

        else:
            current_content += part

    if current_content.strip():
        paragraphs.append((current_active_id, current_content.strip()))

    return paragraphs

def chunk_long_paragraph(content, max_chars=1500, overlap=200):
    if len(content) <= max_chars:
        return [content]
        
    sub_chunks = []
    start = 0
    while start < len(content):
        end = start + max_chars
        sub_chunks.append(content[start:end])
        start += (max_chars - overlap)
    return sub_chunks

def validate_legal_document(text):
    text_upper = text.upper()
    required_anchors = ["COURT", "JUDGMENT", "VERSUS", "SUPREME COURT OF INDIA"]
    matches = sum(1 for anchor in required_anchors if anchor in text_upper)
    return matches >= 2

# ==========================================
# 4. TARGETED PIPELINE EXECUTION
# ==========================================
def run_ingestion_pipeline(target_pdf_name=None, pdf_folder="data/"):
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)

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
                
            # ✅ Fixed: Extract metadata directly from uncollapsed raw_text
            metadata = extract_metadata(raw_text)
            
            fully_cleaned_text = clean_legal_noise(raw_text, case_name=metadata["case_name"])
            judgment_body = extract_judgment_body(fully_cleaned_text)
            
            raw_paragraphs = split_by_legal_paragraphs(judgment_body)
            
            flat_chunks_db = []
            global_chunk_idx = 0
            
            for para_id, content in raw_paragraphs:
                if len(content) < 50:  # Skip structural noise
                    continue
                    
                normalized_content = re.sub(r'\s+', ' ', content).strip()
                sub_parts = chunk_long_paragraph(normalized_content, max_chars=1500)
                
                for p_idx, sub_content in enumerate(sub_parts):
                    sub_para_id = para_id if len(sub_parts) == 1 else f"{para_id}_PART_{p_idx+1}"
                    
                    flat_chunks_db.append({
                        "chunk_id": global_chunk_idx,
                        "text": sub_content,
                        "metadata": {
                            "para_id": sub_para_id,
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
                    global_chunk_idx += 1
            
            safe_id = re.sub(r'[^a-zA-Z0-9]', '_', filename)
            output_chunk_path = f"data/chunks_{safe_id}.json"
            
            with open(output_chunk_path, "w", encoding="utf-8") as f:
                json.dump(flat_chunks_db, f, indent=2, ensure_ascii=False)
                
            print(f"  --> Isolated Chunks Saved: {output_chunk_path} ({len(flat_chunks_db)} chunks)")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    run_ingestion_pipeline()