import re
import json
import os

def clean_legal_noise(text):
    """Removes headers, footers, and repetitive legal metadata."""
    # Remove Indian Kanoon watermarks
    text = re.sub(r'Indian Kanoon - http://indiankanoon.org/doc/\d+/', '', text)
    # Remove recurring Case Title headers (Generalized to avoid hardcoding specific names)
    text = re.sub(r'Union Of India vs Barakathullah on 22 May, 2024', '', text, flags=re.IGNORECASE)
    # Remove standalone page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Clean up excessive newlines
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def extract_judgment_body(text):
    """Filters out the bench and counsel names to start at paragraph 1."""
    if "JUDGMENT" in text:
        return text.split("JUDGMENT", 1)[1]
    return text

def split_by_legal_paragraphs(text):
    """
    Splits text by numbered paragraphs including sub-paragraphs.
    Example: '1.', '17.', '17.10.'
    """
    pattern = r'(\n\s*\d+(?:\.\d+)*\.\s)'
    parts = re.split(pattern, text)

    paragraphs = []
    current_content = ""
    current_para_id = "0"

    for part in parts:
        marker_match = re.match(r'\n\s*(\d+(?:\.\d+)*)\.\s', part)
        if marker_match:
            if current_content.strip():
                paragraphs.append((current_para_id, current_content.strip()))
            current_para_id = marker_match.group(1)
            current_content = ""
        else:
            current_content += part

    if current_content.strip():
        paragraphs.append((current_para_id, current_content.strip()))

    return paragraphs

def create_legal_chunks(input_file="data/processed_data.json", output_file="data/legal_chunks_ready.json"):
    """Main processing loop to generate citation-ready chunks with full metadata."""
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run ingestion.py first.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    final_chunks = []
    chunk_id_counter = 0

    for entry in data:
        raw_text = entry.get("judgment_text", "")
        cleaned_text = clean_legal_noise(raw_text)
        judgment_body = extract_judgment_body(cleaned_text)
        paras = split_by_legal_paragraphs(judgment_body)

        for para_id, content in paras:
            if len(content) < 50:
                continue

            # Creating final dictionary with all metadata from ingestion.py
            final_chunks.append({
                "chunk_id": chunk_id_counter,
                "text": content,
                "metadata": {
                    "para_id": para_id,
                    "case_title": entry.get("case_title", "Unknown"),
                    "case_name": entry.get("case_name", "Unknown"),
                    "date": entry.get("date", "Unknown"),
                    "year": entry.get("year", "Unknown"),
                    "court": entry.get("court", "Supreme Court of India"),
                    "judges": entry.get("judges", "Unknown"),
                    "case_type": entry.get("case_type", "Unknown"),
                    "source": entry.get("source_file", "Unknown")
                }
            })
            chunk_id_counter += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Created {len(final_chunks)} citation-ready chunks in {output_file}.")

if __name__ == "__main__":
    create_legal_chunks()