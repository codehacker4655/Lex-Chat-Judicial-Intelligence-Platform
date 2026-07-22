<div align="center">

# ⚖️ Lex-Chat: Judicial Intelligence Platform

### Enterprise-grade Retrieval-Augmented Generation (RAG) for Legal Intelligence

Analyze, summarize, and cross-reference complex judicial documents with **paragraph-level citations**, **hybrid retrieval**, and **zero-hallucination guardrails**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=for-the-badge)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-orange?style=for-the-badge)
![SQLite](https://img.shields.io/badge/Database-SQLite-blue?style=for-the-badge&logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

</div>

---

# 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture & Tech Stack](#-architecture--tech-stack)
- [System Workflow](#-system-workflow)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Environment Variables](#-environment-variables)
- [Running the Application](#-running-the-application)
- [Running RAGAS Evaluation](#-running-ragas-evaluation)
- [Benchmark Results](#-benchmark-results)
- [Future Roadmap](#-future-roadmap)
- [License](#-license)

---

# 📖 Overview

**Lex-Chat** is an enterprise-grade **Retrieval-Augmented Generation (RAG)** platform built specifically for legal intelligence.

Instead of relying on parametric memory, the platform retrieves relevant legal paragraphs from uploaded judicial documents and generates **citation-grounded responses**, ensuring **high factual accuracy** and **minimal hallucination**.

The system combines:

- 🔍 Dense semantic retrieval
- 📚 Sparse keyword retrieval
- ⚖️ Reciprocal Rank Fusion (RRF)
- 🤖 LLM reasoning
- 📌 Paragraph-level inline citations
- 📊 Telemetry logging for continuous evaluation

---

# ✨ Key Features

### 📄 Monotonic Legal Chunking

- Custom PDF ingestion pipeline
- Preserves original judicial paragraph boundaries
- Tags every paragraph as:

```text
[PARA_1]
[PARA_2]
...
[PARA_136]
```

This eliminates mid-sentence chunk splitting and preserves legal context.

---

### 🔍 Hybrid Retrieval (FAISS + BM25 + RRF)

Retrieval combines:

- **FAISS**
  - Dense semantic similarity search

- **BM25**
  - Sparse keyword retrieval

- **Reciprocal Rank Fusion (RRF)**
  - Combines dense and sparse rankings for improved accuracy

---

### 📌 Inline Legal Citations

Every generated claim is linked directly to the supporting legal paragraph.

Example:

```text
The court held that...

[Para 14 Part 4]
```

---

### 🛡️ Deterministic Guardrails

The assistant refuses to answer questions that cannot be supported by retrieved legal context.

Benefits:

- No fabricated legal facts
- No parametric memory leakage
- Reliable enterprise deployment

---

### ⚖️ Multi-Case Comparative Analysis

Supports multiple judicial documents simultaneously.

The assistant can:

- Compare judgments
- Switch between cases
- Avoid cross-case hallucinations

---

### 📊 Telemetry & Feedback Flywheel

Every interaction logs:

- User question
- Retrieved paragraphs
- Generated response
- Latency
- User feedback

Stored inside:

```text
user_feedback_logs.db
```

These logs can later be converted into evaluation datasets.

---

# 🏗️ Architecture & Tech Stack

| Layer | Technology |
|--------|------------|
| 🎨 Frontend | Streamlit |
| 🔗 Orchestration | LangChain (`RunnableWithMessageHistory`, `ChatPromptTemplate`) |
| 📄 PDF Processing | PyMuPDF (`fitz`) |
| ✂️ Chunking | Custom Regex Paragraph Chunker |
| 🧠 Embeddings | sentence-transformers/all-mpnet-base-v2 |
| 📚 Vector Store | FAISS |
| 🔎 Sparse Retrieval | Rank-BM25 |
| ⚖️ Fusion | Reciprocal Rank Fusion (RRF) |
| 🤖 Reasoning Model | DeepSeek-R1 |
| 💾 Storage | SQLite |
| 📈 Evaluation | RAGAS |

---

# 🔄 System Workflow

```text
                User Query / PDF Upload
                         │
                         ▼
        ┌─────────────────────────────────┐
        │   PDF Ingestion & Chunking      │
        │  • PyMuPDF Extraction           │
        │  • Paragraph Tagging            │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │     Hybrid Retrieval            │
        │                                 │
        │  • FAISS                        │
        │  • BM25                         │
        │  • Reciprocal Rank Fusion       │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │      DeepSeek-R1 Reasoning      │
        │                                 │
        │  • Context Validation           │
        │  • Guardrails                   │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │ Citation-Grounded Response      │
        │ SQLite Telemetry Logging        │
        └─────────────────────────────────┘
```

---

# 📂 Project Structure

```text
Lex-Chat-Judicial-Intelligence-Platform/
│
├── app.py
├── core_logic.py
├── ingestion.py
├── evaluation_metrics.py
│
├── data/
│
├── user_feedback_logs.db
│
├── requirements.txt
├── .env.example
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/codehacker4655/Lex-Chat-Judicial-Intelligence-Platform.git

cd Lex-Chat-Judicial-Intelligence-Platform
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file.

```env
# HuggingFace Inference API

HUGGINGFACEHUB_API_TOKEN=hf_your_token

# Optional

GOOGLE_API_KEY=your_gemini_api_key
```

---

# 🚀 Running the Application

Start the Streamlit application.

```bash
streamlit run app.py
```

Open your browser:

```
http://localhost:8501
```

---

# 📊 Running RAGAS Evaluation

Run:

```bash
python evaluation_metrics.py
```

Metrics computed:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

---

# 📈 Benchmark Results

| Metric | Score | Interpretation |
|--------|-------|----------------|
| ✅ Faithfulness | **1.00** | Every generated claim is grounded in retrieved context |
| 🎯 Answer Relevancy | **0.89** | Highly relevant responses with minimal drift |
| 📚 Context Precision | **0.80** | Relevant legal paragraphs ranked effectively |
| 🔍 Context Recall | **0.80** | Strong retrieval across lengthy judgments |

---

# 🛣️ Future Roadmap

- [ ] Integrate **BGE Cross-Encoder Reranker**
- [ ] PostgreSQL + pgvector migration
- [ ] Qdrant support
- [ ] OCR for scanned legal documents
- [ ] Citation visualization
- [ ] Multi-user authentication
- [ ] Docker deployment
- [ ] Cloud hosting

---

# 📜 License

This project is distributed under the **MIT License**.

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star on GitHub!

Built with ❤️ for Legal AI & Retrieval-Augmented Generation.

</div>
