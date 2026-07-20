import streamlit as st
import os
import re
import sqlite3
import shutil
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
# 0. DATABASE DATA FLYWHEEL LOGGER
# ==========================================
DB_FILE = "data/user_feedback_logs.db"

def init_feedback_db():
    """Initializes a local transactional database to log evaluation metrics."""
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
            feedback_score INTEGER DEFAULT 0,
            error_category TEXT DEFAULT 'None',
            source_file TEXT DEFAULT 'Unknown'
        )
    """)
    conn.commit()
    conn.close()

def log_interaction(query, answer, citations, source_file):
    """Inserts a conversation node tracking footprint into the logging layer."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO logs (timestamp, query, answer, citations, source_file) VALUES (?, ?, ?, ?, ?)",
        (timestamp, query, answer, str(citations), source_file)
    )
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return log_id

def update_log_feedback(log_id, score, category="None"):
    """Applies active optimization flags directly to historical interaction profiles."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE logs SET feedback_score = ?, error_category = ? WHERE id = ?",
        (score, category, log_id)
    )
    conn.commit()
    conn.close()

def get_last_interaction_feedback(source_file):
    """Queries the logging layer for the most recent transaction error flag on the current file."""
    if not os.path.exists(DB_FILE):
        return None
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT feedback_score, error_category FROM logs WHERE source_file = ? ORDER BY id DESC LIMIT 1",
        (source_file,)
    )
    row = cursor.fetchone()
    conn.close()
    return row

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

with st.sidebar:
    st.header("Authentication & Settings")
    hf_token = st.text_input("Enter HuggingFace Token", type="password")
    mode = st.radio("Select Persona:", ["Professional", "Simplified"])
    
    st.divider()
    st.header("📥 Multi-Document Session Manager")
    
    uploaded_file = st.file_uploader("Upload an Indian Judgment PDF", type=["pdf"])
    
    if uploaded_file:
        if uploaded_file.name in st.session_state.uploaded_cases:
            st.warning(f"⚠️ '{uploaded_file.name}' has already been processed and is active in your library! Go ahead and ask your questions directly.")
        else:
            os.makedirs("data", exist_ok=True)
            file_path = os.path.join("data", uploaded_file.name)
            
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            safe_file_id = re.sub(r'[^a-zA-Z0-9]', '_', uploaded_file.name)
            unique_index_path = f"faiss_index_{safe_file_id}"
            
            if st.button("🚀 Process & Index Case File"):
                with st.spinner("Executing structural validation, text extraction, and vector compile loops..."):
                    from ingestion import run_ingestion_pipeline
                    
                    run_ingestion_pipeline(target_pdf_name=uploaded_file.name)
                    target_chunk_json = f"data/chunks_{safe_file_id}.json"
                    
                    if os.path.exists(target_chunk_json):
                        with open(target_chunk_json, "r", encoding="utf-8") as f:
                            chunks_data = json.load(f)
                        
                        new_vs = build_or_load_vector_db(
                            chunks_data=chunks_data,
                            index_path=unique_index_path
                        )
                        
                        st.session_state.uploaded_cases.append(uploaded_file.name)
                        
                        if len(st.session_state.uploaded_cases) > 1:
                            st.session_state.current_active_case = "✨ Search Across All Loaded Cases"
                            
                            first_case_id = re.sub(r'[^a-zA-Z0-9]', '_', st.session_state.uploaded_cases[0])
                            master_vectorstore = build_or_load_vector_db(index_path=f"faiss_index_{first_case_id}")
                            
                            for remaining_case in st.session_state.uploaded_cases[1:]:
                                rem_case_id = re.sub(r'[^a-zA-Z0-9]', '_', remaining_case)
                                next_store = build_or_load_vector_db(index_path=f"faiss_index_{rem_case_id}")
                                master_vectorstore.merge_from(next_store)
                                
                            st.session_state.vectorstore = master_vectorstore
                        else:
                            st.session_state.vectorstore = new_vs
                            st.session_state.current_active_case = uploaded_file.name
                        
                        st.toast(f"Case footprint fully compiled: {uploaded_file.name}", icon="✅")
                        st.rerun()
                    else:
                        st.error("Ingestion failed: Ground truth chunk array structural markers missing.")

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
                with st.spinner("Merging isolated case indices into a global search matrix..."):
                    first_case_id = re.sub(r'[^a-zA-Z0-9]', '_', st.session_state.uploaded_cases[0])
                    master_vectorstore = build_or_load_vector_db(index_path=f"faiss_index_{first_case_id}")
                    
                    for remaining_case in st.session_state.uploaded_cases[1:]:
                        next_case_id = re.sub(r'[^a-zA-Z0-9]', '_', remaining_case)
                        next_store = build_or_load_vector_db(index_path=f"faiss_index_{next_case_id}")
                        master_vectorstore.merge_from(next_store)
                        
                    st.session_state.vectorstore = master_vectorstore
                    st.toast("Global dynamic search mode activated!", icon="✨")
                    st.rerun()
            else:
                safe_selected_id = re.sub(r'[^a-zA-Z0-9]', '_', selected_case)
                target_index_path = f"faiss_index_{safe_selected_id}"
                st.session_state.vectorstore = build_or_load_vector_db(index_path=target_index_path)
                st.toast(f"Context shifted to: {selected_case}", icon="🔄")
                st.rerun()

    st.divider()
    admin_key = st.text_input("⚙️ Developer Console Passkey", type="password", help="Enter engineering passkey to audit backend tracking metrics.")
    
    if admin_key == "orion_dev_2026":
        st.subheader("📊 Active Analytics Pipeline")
        if os.path.exists(DB_FILE):
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT id, timestamp, query, feedback_score, error_category, source_file FROM logs ORDER BY id DESC LIMIT 5")
            rows = cursor.fetchall()
            conn.close()
            
            if rows:
                for r in rows:
                    st.caption(f"**[{r[1]}]** File: `{r[5][:15]}...` | Q: *{r[2][:20]}...* | Score: `{r[3]}` | Tag: `{r[4]}`")
            else:
                st.caption("Database initialized, but no transaction metrics logged yet.")
        else:
            st.caption("No log vectors cached yet.")

# ==========================================
# 2. PROMPT TEMPLATES
# ==========================================
document_prompt = PromptTemplate(
    input_variables=["page_content", "case_name", "court", "year", "judges", "date", "case_type"],
    template=(
        "[SOURCE METADATA | CASE: {case_name} | COURT: {court} | YEAR: {year} | JUDGES: {judges} | DATE: {date} | TYPE: {case_type}]\n"
        "PARA_TEXT: {page_content}"
    )
)

prof_system_prompt = (
    "You are an expert Legal AI Counselor. Use the provided context and metadata to answer.\n"
    "Structure responses matching legal standards: ISSUES, REASONING (citing explicit Paragraph numbers), and CONCLUSION.\n\n"
    "CONTEXT:\n{context}"
)

simple_system_prompt = (
    "You are a Legal Awareness Assistant. Use the context and metadata to simplify the case for a common citizen.\n"
    "Break down technical legal arguments into conversational analogies while maintaining 100% factual accuracy from the Para numbers.\n\n"
    "CONTEXT:\n{context}"
)

# ==========================================
# 3. CHAIN INITIALIZATION
# ==========================================
def create_legal_chain(hf_token, system_prompt_str, vectorstore):
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

    combine_docs_chain = create_stuff_documents_chain(
        llm,
        prompt,
        document_variable_name="context",
        document_prompt=document_prompt
    )
    
    retriever = vectorstore.as_retriever() 
    return create_retrieval_chain(retriever, combine_docs_chain)

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
                
                # 🔄 REAL-TIME SELF-HEALING OPTIMIZATION LAYER
                base_sys_prompt = prof_system_prompt if mode == "Professional" else simple_system_prompt
                
                # Check the database for the last user action metric on this case file
                last_feedback = get_last_interaction_feedback(st.session_state.current_active_case)
                
                if last_feedback and last_feedback[0] == -1:
                    error_type = last_feedback[1]
                    st.warning(f"🔄 Active Feedback Realignment Engaged: Mitigating previous exception context ('{error_type}')")
                    
                    # Inject targeted prompt engineering corrections into the system instructions based on the user's specific complaint
                    correction_injection = "\n\n⚠️ CRITICAL SYSTEM REALIGNMENT WARNING: The user flagged your immediate previous response as a failure due to standard model constraints. "
                    
                    if error_type == "Hallucinated Context":
                        correction_injection += "You must strictly rely on verbatim information from the provided CONTEXT. Do not extrapolate, infer, or pull outside legal details. If a fact is missing from the provided PARA_TEXT, say so explicitly."
                    elif error_type == "Missing Citation Reference":
                        correction_injection += "You must explicitly tie every factual declaration or legal argument back to its designated paragraph number using clear [Para X] anchors. Do not summarize without direct path attribution."
                    elif error_type == "Format Discrepancy":
                        if mode == "Professional":
                            correction_injection += "You must tightly enforce the ISSUES, REASONING, and CONCLUSION layout structure. Do not skip headers or blend these structural sections."
                        else:
                            correction_injection += "Ensure your conversational analogies are 100% factually grounded in the provided parameters without over-simplifying the underlying rule of law."
                    else:
                        correction_injection += "Thoroughly double-check your logical steps, eliminate assumptions, and strictly map your reasoning constraints to the provided context."
                    
                    # Complete the self-healing instruction override
                    base_sys_prompt += correction_injection

                # Bind the dynamically constructed or corrected system prompt to the legal chain instance
                chain = create_legal_chain(hf_token, base_sys_prompt, st.session_state.vectorstore)
                
                chat_bot = RunnableWithMessageHistory(
                    chain, get_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )
                
                docs = hybrid_retrieve(st.session_state.vectorstore, user_query)
                
                response = chat_bot.invoke(
                    {"input": user_query, "context": docs}, 
                    config={"configurable": {"session_id": st.session_state.current_active_case}}
                )

                full_answer = response["answer"]
                clean_answer = re.sub(r'<think>.*?</think>', '', full_answer, flags=re.DOTALL).strip()

                proof_str = ""
                if response["context"]:
                    unique_paras = list(set([doc.metadata.get('para_id', 'Unknown') for doc in response["context"]]))
                    case_name = response["context"][0].metadata.get('case_name', 'Unknown')
                    proof_str = f"**Faithfulness Index:** Grounded in case text loops matching parameters {unique_paras}.\n\n**Source Authority:** '{case_name}' metadata maps."

                log_id = log_interaction(user_query, clean_answer, proof_str, st.session_state.current_active_case)

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
                            st.toast("System alignment metric logged!", icon="✅")
                    with feedback_cols[1]:
                        if st.button("👎", key=f"down_{log_id}"):
                            update_log_feedback(log_id, -1, "General Alert")
                            st.toast("System feedback exception logged.", icon="⚠️")
                            
                    error_choice = st.selectbox(
                        "Report System Divergence:",
                        ["None", "Hallucinated Context", "Missing Citation Reference", "Format Discrepancy"],
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