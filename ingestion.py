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
    # Modified to open via path for VS Code compatibility
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
        # clean unwanted words
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
        judges = judges.split("202")[0]  # remove noise
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
# 7. Process PDFs (VS Code Version)
# ===============================
def run_ingestion_pipeline(pdf_folder="data/"):
    dataset = []
    # Ensure data folder exists
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

            data = metadata
            data["judgment_text"] = judgment_text
            data["source_file"] = filename
            dataset.append(data)
        except Exception as e:
            print(f"Error in {filename}: {e}")

    # Save to JSON
    with open("data/processed_data.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print("\n✅ Done! File saved as data/processed_data.json")

if __name__ == "__main__":
    run_ingestion_pipeline()