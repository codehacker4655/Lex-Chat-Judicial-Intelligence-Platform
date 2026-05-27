import json
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ================================
# 1. EMBEDDING & VECTOR STORE LOGIC
# ================================
def get_embeddings():
    # Using all-mpnet-base-v2 for higher accuracy in legal semantic matching
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def build_or_load_vector_db(chunks_file="data/legal_chunks_ready.json", index_path="faiss_index"):
    embeddings = get_embeddings()
    
    # If index already exists, load it to save time
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunked_data = json.load(f)

    # Mapping all metadata from ingestion.py to the Document object
    documents = [
        Document(
            page_content=item["text"],
            metadata={
                "para_id": item["metadata"]["para_id"],
                "case_title": item["metadata"]["case_title"],
                "case_name": item["metadata"]["case_name"],
                "date": item["metadata"]["date"],
                "year": item["metadata"]["year"],
                "court": item["metadata"]["court"],
                "judges": item["metadata"]["judges"],
                "case_type": item["metadata"]["case_type"],
                "source": item["metadata"]["source"]
            }
        )
        for item in chunked_data
    ]

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

# ================================
# 2. HYBRID RETRIEVAL & POST-PROCESSING
# ================================
def get_para_val(metadata):
    try:
        p_id = str(metadata.get("para_id", "0"))
        if p_id.count('.') > 1:
            parts = p_id.split('.')
            p_id = parts[0] + "." + "".join(parts[1:])
        return float(p_id)
    except ValueError:
        return 0.0

def hybrid_retrieve(vectorstore, query):
    # MMR ensures a diverse set of paragraphs
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30})
    retrieved_docs = retriever.invoke(query)

    # Detect Outcome Intent
    outcome_keywords = ["outcome", "held", "decision", "result", "verdict", "order", "judgment", "ruling"]
    query_lower = query.lower()
    is_outcome_query = any(word in query_lower for word in outcome_keywords)

    if is_outcome_query:
        # Prioritize end of document for outcomes
        processed_docs = sorted(retrieved_docs, key=lambda x: get_para_val(x.metadata), reverse=True)
    else:
        processed_docs = retrieved_docs

    # Critical Term Boosting
    critical_evidence_terms = ["weapon", "training", "sword", "knife", "bomb", "conspiracy", "trishul"]
    priority_docs = []
    other_docs = []

    for d in processed_docs:
        if any(term in d.page_content.lower() for term in critical_evidence_terms):
            priority_docs.append(d)
        else:
            other_docs.append(d)

    return priority_docs + other_docs