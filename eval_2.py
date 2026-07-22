import os
from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace

# ==========================================
# 1. AUTHENTICATION & JUDGE SETUP
# ==========================================

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") 

print("⚖️ Initializing HuggingFace DeepSeek-R1 as RAGAS Judge...")

# Configure LLM Judge
llm_endpoint = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    temperature=0.01  # Low temperature for consistent evaluation scoring
)
judge_llm = LangchainLLMWrapper(ChatHuggingFace(llm=llm_endpoint))

# Configure Local Embedding Judge
judge_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

# ==========================================
# 2. DATASET PREPARATION
# ==========================================
data_samples = {
    "question": [
        "What was the exact numerical difference in distance between the chargeable distance applied by the railways and the actual recognized distance?",
        "What were the specific code words used for training with knives, iron rods, and swords?",
        "Why did the Supreme Court state that proof of a completed terrorist act under Section 15 is not required to establish an offense under Section 18 of the UAPA?",
        "Comparing both uploaded Supreme Court judgments, what were the final decisions of the Supreme Court regarding the High Court orders in each case?"
    ],
    "answer": [
        "The exact numerical difference between the chargeable distance applied by the railways and the actual recognized distance was 110 km [Para 136].",
        "The specific code words were Subject 1 for knives, Subject 2 for iron rods, and Subject 3 for swords [Para 14 Part 4].",
        "Section 18 criminalizes conspiracy, attempt, advocacy, and preparatory acts regardless of whether a completed terrorist act under Section 15 occurred [Para 18 Part 1].",
        "In Barakathullah, the Supreme Court set aside the High Court bail order [Para 22]. In IOCL, the retrieved context does not contain sufficient information regarding the final decision."
    ],
    "contexts": [
        ["a mere change in methodology would not have resulted in a difference of 110 km in the chargeable distance [Para 136]"],
        ["Training was provided on specific weapons: knives, iron rods, and swords referred to by code words Subject 1, Subject 2 and Subject 3 [PARA_14_PART_4]"],
        ["Section 18 explicitly punishes conspiring, attempting, advocating... or any act preparatory to the commission of a terrorist act [PARA_18_PART_1]"],
        ["The High Court allowed the bail... the Supreme Court set aside the order [Para 22]. This batch of 76 appeals is directed against common judgements... [Para 1]"]
    ],
    "ground_truth": [
        "The numerical difference in distance was 110 km.",
        "Subject 1 for knives, Subject 2 for iron rods, and Subject 3 for swords.",
        "Section 18 covers conspiracy and preparatory acts independently, making proof of a Section 15 completed act unnecessary.",
        "In Barakathullah, the SC set aside the bail order. In IOCL, the SC dismissed Union of India's appeal and affirmed the refund."
    ]
}

dataset = Dataset.from_dict(data_samples)

# ==========================================
# 3. RUN EVALUATION
# ==========================================
print("🚀 Running automated RAGAS evaluation across 4 metrics...")

results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=judge_llm,
    embeddings=judge_embeddings
)

# ==========================================
# 4. DISPLAY RESULTS
# ==========================================
df_results = results.to_pandas()
print("\n📊 RAGAS Evaluation Scoreboard:")
print(df_results[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])