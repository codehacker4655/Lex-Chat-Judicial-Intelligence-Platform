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

def build_or_load_vector_db(chunks_data=None, index_path="faiss_index"):
    """
    Loads an existing FAISS index from disk, or builds a brand new isolated 
    vector space dynamically from passed memory data chunks.
    """
    embeddings = get_embeddings()
    
    # 🔄 DYNAMIC CHECK: If this specific file's index already exists on disk, load it instantly
    if os.path.exists(index_path):
        print(f"📦 Loading existing local FAISS index from path: {index_path}...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # If the index doesn't exist, we MUST have raw chunks data passed in to build it
    if chunks_data is None:
        raise ValueError(f"FAISS index at '{index_path}' not found, and no chunks_data was provided to build it.")

    # Aligned with the exact key structure emitted by the ingestion pipeline matrix
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
        for item in chunks_data
    ]

    print(f"🚀 Initializing isolated FAISS indexing over {len(documents)} document nodes...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Saves to a unique directory matching the file (e.g., 'faiss_index_Union_Of_India')
    vectorstore.save_local(index_path)
    print(f"💾 Standalone FAISS index compiled and saved locally to: {index_path}")
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
    chronological parameter re-ranking, and context-adaptive legal token boosting.
    """
    # MMR enforces an informational diversity barrier, avoiding repetitive legal filler phrases
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30})
    retrieved_docs = retriever.invoke(query)

    # 🛑 Step A: Detect Outcome Intent
    outcome_keywords = ["outcome", "held", "decision", "result", "verdict", "order", "judgment", "ruling", "dismissed", "allowed"]
    query_lower = query.lower()
    is_outcome_query = any(word in query_lower for word in outcome_keywords)

    if is_outcome_query:
        print("🔍 Outcome intent signature detected: applying final conclusion ranking biases.")
        processed_docs = sorted(retrieved_docs, key=lambda x: get_para_val(x.metadata), reverse=True)
    else:
        processed_docs = retrieved_docs

    # 🛑 Step B: Dynamic Domain-Agnostic Legal Boosting
    # Instead of hardcoding text, we dynamically boost chunks containing specific statutory markers
    # or high-value legal structural tokens common across ALL case types.
    priority_docs = []
    other_docs = []

    for d in processed_docs:
        content = d.page_content
        
        # Heuristic 1: Contains statutory indicators like "Section 43D", "Article 21", "Notification No."
        has_statute = bool(re.search(r'\b(Section|Article|Notification|Act|Rules|No\.)\b', content, re.IGNORECASE))
        
        # Heuristic 2: Contains specific clause indicators like sub-sections or brackets (e.g., "(5)", "43D(5)")
        has_clause = bool(re.search(r'\b\d+\([a-zA-Z0-9]+\)', content))
        
        if has_statute or has_clause:
            priority_docs.append(d)
        else:
            other_docs.append(d)

    # Construct and return the custom context bundle
    return priority_docs + other_docs