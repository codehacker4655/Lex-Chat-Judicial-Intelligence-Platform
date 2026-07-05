import re
import json
import os

def clean_legal_noise(text, case_name="Unknown"):
    """
    Normalizes spatial layouts and applies dynamic regex expressions to scrub 
    repetitive legal watermarks, platform tags, and running headers.
    """
    # Remove standard Indian Kanoon platform watermarks
    text = re.sub(r'Indian Kanoon\s*-\s*http://indiankanoon\.org/doc/\d+/', '', text, flags=re.IGNORECASE)
    
    # Dynamic Header Filtering: If a case name is provided, dynamically strip it out if it appears as a recurring page header
    if case_name and case_name != "Unknown":
        # Create a flexible regex pattern to match variations of the case title appearing as line headers
        normalized_name = re.escape(case_name[:30]) # Track the primary descriptor segment
        text = re.sub(rf'^\s*{normalized_name}.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)

    # Strip out standalone page numbers or residual system margin tags
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Normalize structural spacing to eliminate parsing noise
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()

def extract_judgment_body(text):
    """
    Locates the operational core of the legal document, dropping preliminary listing schedules, 
    counsel appearances, and procedural index pages.
    """
    # Look for the primary judicial declaration block marker
    judgment_markers = [r'\bJUDGMENT\b', r'\bORDER\b']
    for marker in judgment_markers:
        match = re.search(marker, text, flags=re.IGNORECASE)
        if match:
            # Slice document immediately following the structural keyword boundary
            return text[match.end():].strip()
    return text

def split_by_legal_paragraphs(text):
    """
    Deep-scans the judgment stream to separate primary paragraphs and hierarchically nested subsections.
    Maps complex patterns such as: '1.', '17.', '17.10.', or sub-clauses labeled like '2(a).'
    """
    # Enforces separation at explicit alphanumeric legal paragraph indicators at line breaks
    pattern = r'(\n\s*(?:\[?\d+(?:\.\d+)*\]?|\b[A-Za-z]\b)\.\s)'
    parts = re.split(pattern, text)

    paragraphs = []
    current_content = ""
    current_para_id = "PREAMBLE" # Fallback if text appears prior to structural tracking marker 1.

    for part in parts:
        # Check if the split slice matches a structural enumeration key
        marker_match = re.match(r'\n\s*\[?(\d+(?:\.\d+)*|[A-Za-z])\]?\.\s', part)
        if marker_match:
            # Flush existing segment out if it contains meaningful content weight
            if current_content.strip():
                paragraphs.append((current_para_id, current_content.strip()))
            current_para_id = f"PARA_{marker_match.group(1)}"
            current_content = ""
        else:
            current_content += part

    # Catch the remaining trailing text chunk
    if current_content.strip():
        paragraphs.append((current_para_id, current_content.strip()))

    return paragraphs

def create_legal_chunks(input_file="data/processed_data.json", output_file="data/legal_chunks_ready.json"):
    """
    Consumes the structural database layer, refines data streams, executes legal layout partitioning, 
    and outputs the verified relation mapping JSON required for high-accuracy FAISS ingestion.
    """
    if not os.path.exists(input_file):
        print(f"Error: Ingestion database '{input_file}' not found. Run ingestion pipeline script first.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    final_chunks = []
    chunk_id_counter = 0

    for entry in data:
        raw_text = entry.get("judgment_text", "")
        case_name = entry.get("case_name", "Unknown")
        
        # Step 1: Execute spatial cleaning and text normalization
        cleaned_text = clean_legal_noise(raw_text, case_name=case_name)
        
        # Step 2: Extract core judicial body
        judgment_body = extract_judgment_body(cleaned_text)
        
        # Step 3: Parse text out into identifiable paragraph lists
        paras = split_by_legal_paragraphs(judgment_body)

        # Step 4: Map meta vectors to paragraphs
        for para_id, content in paras:
            # Enforce a Semantic Threshold Filter (Filter out trivial artifacts or legal boilerplate)
            # Short strings like "Order follows." or "Signed by:" add zero value to embedding spaces.
            if len(content) < 60:
                continue

            # Standardize multi-line blocks into clean, uniform paragraph text chunks
            normalized_content = re.sub(r'\s+', ' ', content).strip()

            final_chunks.append({
                "chunk_id": chunk_id_counter,
                "text": normalized_content,
                "metadata": {
                    "para_id": para_id,
                    "case_title": entry.get("case_title", "Unknown"),
                    "case_name": case_name,
                    "date": entry.get("date", "Unknown"),
                    "year": entry.get("year", "Unknown"),
                    "court": entry.get("court", "Supreme Court of India"),
                    "judges": entry.get("judges", "Unknown"),
                    "case_type": entry.get("case_type", "Unknown"),
                    "source_file": entry.get("source_file", "Unknown")
                }
            })
            chunk_id_counter += 1

    # Write output to our relational JSON endpoint
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Processing Pipeline Verification Complete!")
    print(f"  --> Data File Compiled: {output_file}")
    print(f"  --> Total Semantic Vectors Generated: {len(final_chunks)} citation-ready paragraphs mapped.")

if __name__ == "__main__":
    create_legal_chunks()