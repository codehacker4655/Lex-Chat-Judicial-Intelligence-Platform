import json
import os
import re
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ================================
# 1. EMBEDDING & VECTOR STORE LOGIC
# ================================
def get_embeddings():
    # Utilizing all-mpnet-base-v2 for industry-standard semantic legal matching accuracy
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def build_or_load_vector_db(chunks_file="data/legal_chunks_ready.json", index_path="faiss_index"):
    embeddings = get_embeddings()
    
    # Check for index footprint; load locally to optimize memory and skip re-indexing
    if os.path.exists(index_path):
        print(f"📦 Loading local FAISS index from path: {index_path}...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    if not os.path.exists(chunks_file):
        raise FileNotFoundError(f"Missing chunk storage index: {chunks_file}. Execute data preparation script first.")

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunked_data = json.load(f)

    # Aligned with the exact key structure emitted by the ingestion pipeline
    documents = [
        Document(
            page_content=item["text"],
            metadata={
                "para_id": item["metadata"].get("para_id", "Unknown"),
                "case_title": item["metadata"].get("case_title", "Unknown"),
                "case_name": item["metadata"].get("case_name", "Unknown"),
                "date": item["metadata"].get("date", "Unknown"),
                "year": item["metadata"].get("year", "Unknown"),
                "court": item["metadata"].get("court", "Supreme Court of India"),
                "judges": item["metadata"].get("judges", "Unknown"),
                "case_type": item["metadata"].get("case_type", "Unknown"),
                "source_file": item["metadata"].get("source_file", "Unknown")
            }
        )
        for item in chunked_data
    ]

    print(f"🚀 Initializing FAISS indexing execution over {len(documents)} document nodes...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_path)
    print(f"💾 FAISS index compiled and saved locally to: {index_path}")
    return vectorstore

# ================================
# 2. HYBRID RETRIEVAL & POST-PROCESSING
# ================================
def get_para_val(metadata):
    """
    Extracts numerical tracking metrics from string paragraph tags (e.g., 'PARA_17.10' -> 17.1)
    to facilitate chronological post-retrieval sorting rules.
    """
    try:
        p_id_str = str(metadata.get("para_id", "0"))
        
        # Extract numerical digits or float fractions using regular expressions
        num_match = re.search(r'\d+(?:\.\d+)*', p_id_str)
        if not num_match:
            return 0.0
            
        val_str = num_match.group(0)
        if val_str.count('.') > 1:
            # Reformat trailing decimal noise out safely (e.g., 17.10.1 -> 17.101)
            parts = val_str.split('.')
            val_str = parts[0] + "." + "".join(parts[1:])
            
        return float(val_str)
    except (ValueError, TypeError):
        return 0.0

def hybrid_retrieve(vectorstore, query):
    """
    Executes a high-yield intent-aware hybrid retrieval loop combining MMR document diversity, 
    chronological parameter re-ranking, and priority heuristic token boosting.
    """
    # MMR enforces an informational diversity barrier, avoiding repetitive legal filler phrases
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30})
    retrieved_docs = retriever.invoke(query)

    # 🛑 Step A: Detect Outcome Intent
    outcome_keywords = ["outcome", "held", "decision", "result", "verdict", "order", "judgment", "ruling", "dismissed", "allowed"]
    query_lower = query.lower()
    is_outcome_query = any(word in query_lower for word in outcome_keywords)

    if is_outcome_query:
        # Route query dynamically to prioritize conclusion structural text at the end of the judgment
        print("🔍 Outcome intent signature detected: applying final conclusion ranking biases.")
        processed_docs = sorted(retrieved_docs, key=lambda x: get_para_val(x.metadata), reverse=True)
    else:
        processed_docs = retrieved_docs

    # 🛑 Step B: Dynamic Evidence Term Boosting
    critical_evidence_terms = ["weapon", "training", "sword", "knife", "bomb", "conspiracy", "assault", "murder", "intent"]
    priority_docs = []
    other_docs = []

    for d in processed_docs:
        content_lower = d.page_content.lower()
        if any(term in content_lower for term in critical_evidence_terms):
            priority_docs.append(d)
        else:
            other_docs.append(d)

    # Construct and return the custom context bundle
    return priority_docs + other_docs