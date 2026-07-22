import json
import os
import re
import numpy as np
from rank_bm25 import BM25Okapi
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
    
    # 🔄 DYNAMIC CHECK: If this specific file's index already exists on disk, load it
    if os.path.exists(index_path):
        print(f"📦 Loading existing local FAISS index from path: {index_path}...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    if chunks_data is None:
        raise ValueError(f"FAISS index at '{index_path}' not found, and no chunks_data was provided to build it.")

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
    vectorstore.save_local(index_path)
    print(f"💾 Standalone FAISS index saved to: {index_path}")
    return vectorstore

# ================================
# 2. BM25 INDEX BUILDER & HELPER
# ================================
class LegalBM25Retriever:
    """
    BM25 Sparse Keyword Searcher designed to run alongside FAISS.
    """
    def __init__(self, chunks_data):
        self.chunks_data = chunks_data
        # Tokenize text while keeping legal clause patterns intact
        self.corpus = [self._tokenize(item["text"]) for item in chunks_data]
        self.bm25 = BM25Okapi(self.corpus)

    def _tokenize(self, text):
        clean_text = re.sub(r'[^\w\s\(\)]', ' ', text.lower())
        return clean_text.split()

    def search(self, query, top_k=12):
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Skip non-matching chunks
                item = self.chunks_data[idx]
                doc = Document(
                    page_content=item["text"],
                    metadata=item["metadata"]
                )
                results.append(doc)
        return results

# ================================
# 3. RECIPROCAL RANK FUSION (RRF)
# ================================
def reciprocal_rank_fusion(dense_docs, sparse_docs, rrf_k=60, top_n=10):
    """
    Merges dense FAISS results and sparse BM25 results using RRF.
    Formula: Score = 1 / (60 + Rank)
    """
    rrf_scores = {}

    def add_docs(doc_list, weight=1.0):
        for rank, doc in enumerate(doc_list, start=1):
            source = doc.metadata.get('source_file', 'unknown')
            para = doc.metadata.get('para_id', 'para')
            # Deduplication Key incorporating file source + para ID
            para_key = f"{source}_{para}_{doc.page_content[:50]}"
            
            if para_key not in rrf_scores:
                rrf_scores[para_key] = {"doc": doc, "score": 0.0}
            rrf_scores[para_key]["score"] += weight * (1.0 / (rrf_k + rank))

    add_docs(dense_docs, weight=1.0)
    add_docs(sparse_docs, weight=1.0)

    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs[:top_n]]

# ================================
# 4. HYBRID RETRIEVAL PIPELINE
# ================================
def hybrid_retrieve(vectorstore, query, chunks_data=None, top_n=10, k_dense=12):
    """
    Executes True Hybrid Search:
    1. Dense Retrieval using FAISS (k=k_dense)
    2. Sparse Retrieval using BM25 (top_k=k_dense if chunks_data available)
    3. Reciprocal Rank Fusion (RRF) returning top_n documents.
    """

    # 1. Dense Retrieval (Semantic Search)
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_dense}
    )
    dense_docs = dense_retriever.invoke(query)

    # 2. Sparse Retrieval (Keyword Search)
    sparse_docs = []
    if chunks_data:
        bm25_searcher = LegalBM25Retriever(chunks_data)
        sparse_docs = bm25_searcher.search(query, top_k=k_dense)

    # 3. Hybrid Fusion using Reciprocal Rank Fusion (RRF)
    if sparse_docs:
        fused_docs = reciprocal_rank_fusion(
            dense_docs,
            sparse_docs,
            top_n=top_n
        )
    else:
        fused_docs = dense_docs[:top_n]

    return fused_docs