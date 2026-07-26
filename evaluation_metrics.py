import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from dotenv import load_dotenv
load_dotenv()

# Modern RAGAS Native HuggingFace Embeddings import (No deprecation warning!)
from ragas.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# =====================================================================
# 1. SET HUGGING FACE TOKEN & LLM / EMBEDDING SETUP
# =====================================================================
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") 

# Fix 1: Use ChatHuggingFace wrapper to use 'conversational' task endpoints
llm_endpoint = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    huggingfacehub_api_token=hf_token,
    temperature=0.1,
    max_new_tokens=2048
)
evaluator_llm = ChatHuggingFace(llm=llm_endpoint)

# Fix 2: Use sentence-transformers/all-mpnet-base-v2 via native RAGAS HuggingFaceEmbeddings
evaluator_embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2"
)

# =====================================================================
# 2. LOAD CHUNKS FROM ingestion.py JSON
# =====================================================================
def load_chunks_from_json():
    data_dir = "data"
    chunk_file = None
    
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.startswith("chunks_") and f.endswith(".json"):
                chunk_file = os.path.join(data_dir, f)
                break
                
    if not chunk_file or not os.path.exists(chunk_file):
        print("⚠️ Warning: Chunks JSON file not found in data/. Using fallback mapping.")
        return {}

    with open(chunk_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    para_map = {item["metadata"]["para_id"]: item["text"] for item in data}
    return para_map

chunks_db = load_chunks_from_json()

# =====================================================================
# 3. BENCHMARK RUN DATA
# =====================================================================
benchmark_runs = [
    {
        "user_input": "Summarize the conflict regarding the 444 km chargeable distance vs 334 km actual distance.",
        "response": """ISSUES: The dispute centers on the validity of the 444 km chargeable distance applied by the Union of India versus the 334 km actual distance asserted by IOCL.
REASONING: The respondent contested the 444 km chargeable distance as illegal, arguing it lacked justification and deviated from the actual distance of 334 km. The appellant defended the 444 km distance under old methodology but failed to provide the complete old distance table or explain the 110 km difference.
CONCLUSION: The conflict arises from imposing a 444 km chargeable distance despite a 334 km actual distance without providing valid documentation.""",
        "retrieved_para_ids": ['PARA_b', 'PARA_123', 'PARA_120', 'PARA_122'],
        "gold_standard_ids": ['PARA_1', 'PARA_7', 'PARA_110', 'PARA_120', 'PARA_122', 'PARA_123'],
        "reference": "The conflict centers on Railways charging freight for 444 km based on old distance tables, whereas the actual engineering distance was 333.18 km (334 km). IOCL claimed a refund for the excess 110 km erroneously charged."
    },
    {
        "user_input": "What was the difference between Category 1 (Illegal Charge) and Category 2 (Overcharge) according to paragraph 90?",
        "response": """Category 1 (Illegal Charge): Defined as a charge contrary to the law or lacking authority of law—inherently unlawful and not payable under any legal framework. Void for everyone and not subject to 6-month limitation.
Category 2 (Overcharge): An excess amount beyond what is payable in the eyes of law but partaking the same character/genus as a lawful base charge. Specific to particular facts, subject to strict 6-month limitation.""",
        "retrieved_para_ids": ['PARA_70', 'PARA_94', 'PARA_62', 'PARA_71'],
        "gold_standard_ids": ['PARA_62', 'PARA_70', 'PARA_71', 'PARA_90', 'PARA_94'],
        "reference": "Category 1 (Illegal Charge) is a levy outside or beyond the law, void ab initio without limitation bounds. Category 2 (Overcharge) is a levy in excess of what was required by law, subject to a strict 6-month limitation under Section 106(3)."
    },
    {
        "user_input": "Why was the notice under Section 106(3) held to be inapplicable for an illegal charge?",
        "response": """Section 106(3) explicitly applies only to 'overcharges' resulting from mistakes. An illegal charge is void ab initio because it lacks legal authority. Thus, no valid charge exists to which Section 106(3)'s notice requirement could attach, and imposing notice requirements would undermine judicial remedies against unlawful impositions.""",
        "retrieved_para_ids": ['PARA_98', 'PARA_121', 'PARA_18', 'PARA_59'],
        "gold_standard_ids": ['PARA_34', 'PARA_59', 'PARA_91', 'PARA_93', 'PARA_98', 'PARA_121'],
        "reference": "Section 106(3) limits claims strictly for overcharges. Illegal charges are outside the scope of Section 106(3) because they lack statutory authority from inception, so they cannot be barred by the 6-month notice period."
    }
]

# =====================================================================
# 4. PREPARE DATASET & EVALUATE
# =====================================================================
eval_dict = {
    "user_input": [],
    "response": [],
    "retrieved_contexts": [],
    "reference": []
}

print("\n-------------------------------------------------------")
print("🎯 RETRIEVAL HIT RATE vs. GOLD STANDARD CHUNKS")
print("-------------------------------------------------------")

for idx, run in enumerate(benchmark_runs, 1):
    eval_dict["user_input"].append(run["user_input"])
    eval_dict["response"].append(run["response"])
    eval_dict["reference"].append(run["reference"])
    
    retrieved_texts = [
        chunks_db.get(pid, f"Content of {pid}: Paragraph details from judgment text.") 
        for pid in run["retrieved_para_ids"]
    ]
    eval_dict["retrieved_contexts"].append(retrieved_texts)
    
    hits = set(run["retrieved_para_ids"]).intersection(set(run["gold_standard_ids"]))
    hit_rate = len(hits) / len(run["gold_standard_ids"])
    print(f"Query {idx}: Hit Rate = {hit_rate:.2%} | Matched Gold Standard IDs: {list(hits)}")

print("\n🚀 Executing RAGAS Evaluator...")
eval_dataset = Dataset.from_dict(eval_dict)

results = evaluate(
    dataset=eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)



print("\n=======================================================")
print("📊 LEX-CHAT FINAL RAGAS BENCHMARK RESULTS")
print("=======================================================\n")
print(df[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])

print("\n-------------------------------------------------------")
print(f"✅ Faithfulness (Zero-Hallucination Score):  {df['faithfulness'].mean():.2f}")
print(f"✅ Answer Relevancy:                         {df['answer_relevancy'].mean():.2f}")
print(f"✅ Context Precision (FAISS Quality):        {df['context_precision'].mean():.2f}")
print(f"✅ Context Recall (Completeness):            {df['context_recall'].mean():.2f}")
print("-------------------------------------------------------\n")

