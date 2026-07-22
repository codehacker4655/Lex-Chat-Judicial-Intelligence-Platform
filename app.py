import streamlit as st
import time
import os
import re
import sqlite3
import json
from datetime import datetime
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate

# Import identical logic from other files
from core_logic import build_or_load_vector_db, hybrid_retrieve, get_embeddings

# ==========================================
# 0. USER INTERACTION & FEEDBACK LOGGING LAYER
# ==========================================
DB_FILE = "data/user_feedback_logs.db"

def init_feedback_db():
    """Initializes a local transactional database with execution metrics logging."""
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            answer TEXT,
            citations TEXT,
            retrieval_time REAL DEFAULT 0.0,
            response_time REAL DEFAULT 0.0,
            feedback_score INTEGER DEFAULT 0,
            error_category TEXT DEFAULT 'None',
            source_file TEXT DEFAULT 'Unknown'
        )
    """)
    conn.commit()
    conn.close()

def log_interaction(query, answer, citations, source_file, retrieval_time=0.0, response_time=0.0):
    """Inserts a conversation node tracking footprint with latency metrics into the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """INSERT INTO logs 
           (timestamp, query, answer, citations, source_file, retrieval_time, response_time) 
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (timestamp, query, answer, str(citations), source_file, retrieval_time, response_time)
    )
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return log_id

def update_log_feedback(log_id, score, category="None"):
    """Applies active feedback flags directly to database records."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE logs SET feedback_score = ?, error_category = ? WHERE id = ?",
        (score, category, log_id)
    )
    conn.commit()
    conn.close()

# Initialize logging layer at script instantiation
init_feedback_db()

# ==========================================
# 1. UI CONFIGURATION & SIDEBAR MANAGER
# ==========================================
st.set_page_config(page_title="Lex-Chat: Judicial Intelligence", layout="wide")
st.title("⚖️ Lex-Chat: Judicial Intelligence Platform")

if "uploaded_cases" not in st.session_state:
    st.session_state.uploaded_cases = []
if "current_active_case" not in st.session_state:
    st.session_state.current_active_case = None
if "current_chunks_data" not in st.session_state:
    st.session_state.current_chunks_data = None

with st.sidebar:
    st.header("Authentication & Settings")
    hf_token = st.text_input("Enter HuggingFace Token", type="password")
    mode = st.radio("Select Persona:", ["Professional", "Simplified"])
    
    st.divider()
    st.header("📥 Multi-Document Session Manager")
    
    uploaded_file = st.file_uploader("Upload an Indian Judgment PDF", type=["pdf"])
    
    if uploaded_file:
        if uploaded_file.name in st.session_state.uploaded_cases:
            st.warning(f"⚠️ '{uploaded_file.name}' is already active in your library!")
        else:
            os.makedirs("data", exist_ok=True)
            file_path = os.path.join("data", uploaded_file.name)
            
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            safe_file_id = re.sub(r'[^a-zA-Z0-9]', '_', uploaded_file.name)
            unique_index_path = f"faiss_index_{safe_file_id}"
            
            if st.button("🚀 Process & Index Case File"):
                with st.spinner("Processing PDF and building hybrid index..."):
                    from ingestion import run_ingestion_pipeline
                    
                    run_ingestion_pipeline(target_pdf_name=uploaded_file.name)
                    target_chunk_json = f"data/chunks_{safe_file_id}.json"
                    
                    if os.path.exists(target_chunk_json):
                        with open(target_chunk_json, "r", encoding="utf-8") as f:
                            chunks_data = json.load(f)

                        st.session_state.current_chunks_data = chunks_data
                        
                        new_vs = build_or_load_vector_db(
                            chunks_data=chunks_data,
                            index_path=unique_index_path
                        )
                        
                        st.session_state.uploaded_cases.append(uploaded_file.name)
                        
                        if len(st.session_state.uploaded_cases) > 1:
                            st.session_state.current_active_case = "✨ Search Across All Loaded Cases"
                            
                            first_case_id = re.sub(r'[^a-zA-Z0-9]', '_', st.session_state.uploaded_cases[0])
                            master_vectorstore = build_or_load_vector_db(index_path=f"faiss_index_{first_case_id}")
                            
                            merged_chunks = []
                            for remaining_case in st.session_state.uploaded_cases:
                                rem_case_id = re.sub(r'[^a-zA-Z0-9]', '_', remaining_case)
                                chunk_file = f"data/chunks_{rem_case_id}.json"
                                if os.path.exists(chunk_file):
                                    with open(chunk_file, "r", encoding="utf-8") as cf:
                                        merged_chunks.extend(json.load(cf))
                                
                                if remaining_case != st.session_state.uploaded_cases[0]:
                                    next_store = build_or_load_vector_db(index_path=f"faiss_index_{rem_case_id}")
                                    master_vectorstore.merge_from(next_store)
                                
                            st.session_state.vectorstore = master_vectorstore
                            st.session_state.current_chunks_data = merged_chunks
                        else:
                            st.session_state.vectorstore = new_vs
                            st.session_state.current_active_case = uploaded_file.name
                        
                        st.toast(f"Case footprint fully compiled: {uploaded_file.name}", icon="✅")
                        st.rerun()
                    else:
                        st.error("Ingestion failed: Ground truth chunk array missing.")

    if st.session_state.uploaded_cases:
        st.divider()
        st.subheader("📁 Active Case Library")
        
        dropdown_options = st.session_state.uploaded_cases.copy()
        if len(dropdown_options) > 1:
            dropdown_options.insert(0, "✨ Search Across All Loaded Cases")
            
        selected_case = st.selectbox(
            "Select active target context:", 
            dropdown_options,
            index=dropdown_options.index(st.session_state.current_active_case) if st.session_state.current_active_case in dropdown_options else 0
        )
        
        if selected_case != st.session_state.current_active_case:
            st.session_state.current_active_case = selected_case
            
            if selected_case == "✨ Search Across All Loaded Cases":
                with st.spinner("Merging case indices..."):
                    first_case_id = re.sub(r'[^a-zA-Z0-9]', '_', st.session_state.uploaded_cases[0])
                    master_vectorstore = build_or_load_vector_db(index_path=f"faiss_index_{first_case_id}")
                    
                    merged_chunks = []
                    for case_file in st.session_state.uploaded_cases:
                        case_id = re.sub(r'[^a-zA-Z0-9]', '_', case_file)
                        chunk_file = f"data/chunks_{case_id}.json"
                        if os.path.exists(chunk_file):
                            with open(chunk_file, "r", encoding="utf-8") as cf:
                                merged_chunks.extend(json.load(cf))
                        
                        if case_file != st.session_state.uploaded_cases[0]:
                            next_store = build_or_load_vector_db(index_path=f"faiss_index_{case_id}")
                            master_vectorstore.merge_from(next_store)
                        
                    st.session_state.vectorstore = master_vectorstore
                    st.session_state.current_chunks_data = merged_chunks
                    st.toast("Global dynamic search mode activated!", icon="✨")
                    st.rerun()
            else:
                safe_selected_id = re.sub(r'[^a-zA-Z0-9]', '_', selected_case)
                target_index_path = f"faiss_index_{safe_selected_id}"
                st.session_state.vectorstore = build_or_load_vector_db(index_path=target_index_path)
                
                chunk_file = f"data/chunks_{safe_selected_id}.json"
                if os.path.exists(chunk_file):
                    with open(chunk_file, "r", encoding="utf-8") as cf:
                        st.session_state.current_chunks_data = json.load(cf)
                        
                st.toast(f"Context shifted to: {selected_case}", icon="🔄")
                st.rerun()

    st.divider()
    admin_key = st.text_input("⚙️ Developer Console Passkey", type="password")
    
    if admin_key == "orion_dev_2026":
        st.subheader("📊 Active Analytics Pipeline")
        if os.path.exists(DB_FILE):
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT id, timestamp, query, retrieval_time, response_time, feedback_score, error_category, source_file FROM logs ORDER BY id DESC LIMIT 5")
            rows = cursor.fetchall()
            conn.close()
            
            if rows:
                for r in rows:
                    st.caption(f"**[{r[1]}]** File: `{r[7][:12]}...` | R-Time: `{r[3]:.2f}s` | L-Time: `{r[4]:.2f}s` | Score: `{r[5]}`")
            else:
                st.caption("No log vectors cached yet.")

# ==========================================
# 2. PROMPT TEMPLATES
# ==========================================
document_prompt = PromptTemplate(
    input_variables=[
        "page_content", 
        "para_id", 
        "case_name", 
        "court", 
        "year", 
        "judges", 
        "date", 
        "case_type"
    ],
    template=(
        "[SOURCE METADATA | PARAGRAPH ID: {para_id} | CASE: {case_name} | COURT: {court} | YEAR: {year} | JUDGES: {judges} | DATE: {date} | TYPE: {case_type}]\n"
        "PARA_TEXT: {page_content}"
    )
)
Prof_system_prompt = ("""
You are an expert Legal AI Assistant specializing in Indian Supreme Court judgments.

CRITICAL REFUSAL DIRECTIVE:
If the query asks about a statute, legal case, or topic that is NOT explicitly mentioned in the retrieved CONTEXT below , you MUST immediately refuse to answer and output strictly:
"The retrieved context does not contain sufficient information to answer this question."

RULES:
• Answer ONLY using the retrieved CONTEXT.
• Do NOT use outside legal knowledge, assumptions, or prior training.
• If the answer cannot be found in the retrieved context, clearly state:
  "The retrieved context does not contain sufficient information to answer this question."
• Never fabricate facts, legal conclusions, or citations.
• Every factual statement, legal proposition, observation, or conclusion MUST immediately end with one or more paragraph citations in the format [Para X].
• Never group citations at the end of a paragraph. Attach each citation directly to the sentence it supports.
• If multiple retrieved paragraphs support the same statement, cite all relevant paragraphs (e.g., [Para 12, Para 18]).
• If multiple retrieved paragraphs are relevant, combine them into one coherent answer.
• Adapt your response format to the user's question. Use a direct answer, explanation, comparison, summary, or bullet points as appropriate.
• Keep the answer focused on the user's question and avoid unnecessary information.

CONTEXT:
{context}
""")

simple_system_prompt = ("""
You are a Legal Awareness Assistant helping people understand Indian Supreme Court judgments.

CRITICAL REFUSAL DIRECTIVE:
If the question is about a topic or case not present in the provided CONTEXT, clearly state:
"The retrieved context does not contain sufficient information to answer this question."

RULES:
• Answer ONLY using the retrieved CONTEXT.
• Do NOT guess or add information that is not present in the retrieved context.
• If the answer is unavailable in the retrieved context, clearly say so.
• Never invent facts or legal conclusions.
• Use simple everyday English.
• Avoid legal jargon whenever possible. If legal terms are necessary, explain them immediately in simple language.
• Every factual statement must immediately end with paragraph citations in the format [Para X].
• Never make unsupported statements.
• If multiple retrieved paragraphs are relevant, combine them into one easy-to-understand explanation.
• Adapt your response style to the user's question. Direct answers for factual questions, detailed explanations for "why" or "how" questions, and concise summaries when requested.
• Keep the explanation clear, natural, and focused on answering the user's question.

CONTEXT:
{context}
""")

# ==========================================
# 3. CHAIN INITIALIZATION
# ==========================================
def create_legal_chain(hf_token, system_prompt_str):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_str),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1",
        task="text-generation",
        max_new_tokens=2048,
        huggingfacehub_api_token=hf_token
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    return create_stuff_documents_chain(
        llm,
        prompt,
        document_variable_name="context",
        document_prompt=document_prompt
    )

# ==========================================
# 4. CHAT INTERFACE & RUNTIME EXECUTION
# ==========================================
if hf_token:
    if "vectorstore" in st.session_state and st.session_state.current_active_case is not None:
        
        if "store" not in st.session_state: 
            st.session_state.store = {}
            
        def get_history(session_id):
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        if "messages" not in st.session_state: 
            st.session_state.messages = []
        
        col1, col2 = st.columns([8, 2])
        with col2:
            if len(st.session_state.messages) > 0:
                raw_brief = f"# Legal Analysis Brief\n*Generated on: {datetime.now().strftime('%Y-%m-%d')} via Lex-Chat Engine*\n\n"
                for m in st.session_state.messages:
                    speaker = "### User Query" if m["role"] == "user" else "### Verified Judicial Rationale"
                    raw_brief += f"{speaker}:\n{m['content']}\n\n"
                
                st.download_button(
                    label="📥 Export Brief as MD",
                    data=raw_brief,
                    file_name=f"Legal_Brief_{datetime.now().strftime('%d_%m')}.md",
                    mime="text/markdown"
                )

        for idx, m in enumerate(st.session_state.messages):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m["role"] == "assistant" and "proof_report" in m:
                    with st.expander("📊 Verification Trace Details"):
                        st.caption(m["proof_report"])

        if user_query := st.chat_input("Ask about the judgment, the judges, or for a summary..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"): 
                st.markdown(user_query)

            with st.spinner(f"Querying matching context from {st.session_state.current_active_case}..."):

                base_sys_prompt = Prof_system_prompt if mode == "Professional" else simple_system_prompt

                chain = create_legal_chain(hf_token, base_sys_prompt)

                chat_bot = RunnableWithMessageHistory(
                    chain, 
                    get_history,
                    input_messages_key="input",
                    history_messages_key="chat_history"
                )
                
                chunks_data = st.session_state.get("current_chunks_data", None)
                
                # 🚀 Search Depth: Set dynamic top_n window based on single vs multi-doc search
                is_multi_mode = (st.session_state.current_active_case == "✨ Search Across All Loaded Cases")
                search_depth = 16 if is_multi_mode else 10
                
                # ⏱️ Measure Retrieval Latency
                retrieval_start = time.perf_counter()
                docs = hybrid_retrieve(
                    st.session_state.vectorstore, 
                    user_query, 
                    chunks_data=chunks_data,
                    top_n=search_depth,
                    k_dense=search_depth
                )
                retrieval_end = time.perf_counter()
                retrieval_time = retrieval_end - retrieval_start

                # ⏱️ Measure LLM Response Latency
                response_start = time.perf_counter()
                full_answer = chat_bot.invoke(
                    {"input": user_query, "context": docs}, 
                    config={"configurable": {"session_id": st.session_state.current_active_case}}
                )
                response_end = time.perf_counter()
                response_time = response_end - response_start

                clean_answer = re.sub(r'<think>.*?</think>', '', full_answer, flags=re.DOTALL).strip()

                # Build Proof Report with explicitly mapped file sources & paragraph IDs
                proof_str = ""
                if docs:
                    doc_sources = set([f"{doc.metadata.get('source_file', 'Unknown')} ({doc.metadata.get('para_id', 'PARA')})" for doc in docs])
                    source_list_str = ", ".join(list(doc_sources))
                    proof_str = f"**Faithfulness Index:** Grounded in retrieved nodes: `{source_list_str}`.\n\n**Performance Metrics:** Retrieval: `{retrieval_time:.2f}s` | Generation: `{response_time:.2f}s`"

                # 💾 LOG EVERYTHING (Including performance metrics) TO DATABASE
                log_id = log_interaction(
                    query=user_query, 
                    answer=clean_answer, 
                    citations=proof_str, 
                    source_file=st.session_state.current_active_case,
                    retrieval_time=retrieval_time,
                    response_time=response_time
                )
                
                with st.chat_message("assistant"):
                    st.markdown(clean_answer)
                    
                    if proof_str:
                        with st.expander("📊 Evaluation & Proof Report"):
                            st.markdown(proof_str)
                    
                    st.markdown("---")
                    feedback_cols = st.columns([1, 1, 4, 4])
                    with feedback_cols[0]:
                        if st.button("👍", key=f"up_{log_id}"):
                            update_log_feedback(log_id, 1)
                            st.toast("Thanks for your feedback!", icon="✅")
                    with feedback_cols[1]:
                        if st.button("👎", key=f"down_{log_id}"):
                            update_log_feedback(log_id, -1, "General Alert")
                            st.toast("Feedback recorded. This will help us improve!", icon="⚠️")
                            
                    error_choice = st.selectbox(
                        "Report System Divergence:",
                        ["None", "Hallucinated Context", "Missing Citation Reference", "Format Discrepancy", "Incorrect Answer", "Incomplete Answer"],
                        key=f"err_{log_id}"
                    )
                    if error_choice != "None":
                        update_log_feedback(log_id, -1, error_choice)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": clean_answer,
                    "proof_report": proof_str
                })
    else:
        st.info("👋 Welcome! Please upload an Indian Supreme Court Judgment PDF in the sidebar and click **'Process & Index Case File'** to activate the localized relational RAG context.")
else:
    st.warning("Please enter your HuggingFace API token in the sidebar to activate the AI.")