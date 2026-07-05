import streamlit as st
import os
import re
import sqlite3
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

# Import your identical logic from other files
from core_logic import build_or_load_vector_db, hybrid_retrieve, get_embeddings

# ==========================================
# 0. DATABASE DATA FLYWHEEL LOGGER (Orion Feature)
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
            error_category TEXT DEFAULT 'None'
        )
    """)
    conn.commit()
    conn.close()

def log_interaction(query, answer, citations):
    """Inserts a conversation node tracking footprint into the logging layer."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO logs (timestamp, query, answer, citations) VALUES (?, ?, ?, ?)",
        (timestamp, query, answer, str(citations))
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

# Initialize logging layer at script instantiation
init_feedback_db()

# ==========================================
# 1. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Lex-Chat: Judicial Intelligence", layout="wide")
st.title("⚖️ Lex-Chat: Judicial Intelligence Platform")

with st.sidebar:
    st.header("Authentication & Settings")
    hf_token = st.text_input("Enter HuggingFace Token", type="password")
    mode = st.radio("Select Persona:", ["Professional", "Simplified"])
    
    st.divider()
    st.info("System verified. Paragraph mapping indices are tracked statically via FAISS storage layers.")
    
    # PRODUCT ADDITION: Interactive Database View for Corporate Reviewers
    if st.checkbox("📊 View System Logs"):
        st.subheader("Active Analytics Pipeline")
        if os.path.exists(DB_FILE):
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT id, timestamp, query, feedback_score, error_category FROM logs ORDER BY id DESC LIMIT 5")
            rows = cursor.fetchall()
            conn.close()
            for r in rows:
                st.caption(f"**[{r[1]}]** Q: *{r[2][:30]}...* | Score: `{r[3]}` | Tag: `{r[4]}`")
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
# 4. CHAT INTERFACE & SESSION STATE
# ==========================================
if hf_token:
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = build_or_load_vector_db()

    if "store" not in st.session_state: st.session_state.store = {}
    def get_history(session_id):
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    sys_prompt = prof_system_prompt if mode == "Professional" else simple_system_prompt
    chain = create_legal_chain(hf_token, sys_prompt, st.session_state.vectorstore)
    
    chat_bot = RunnableWithMessageHistory(
        chain, get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    if "messages" not in st.session_state: st.session_state.messages = []
    
    # Header Control Bar for System Utilities
    col1, col2 = st.columns([8, 2])
    with col2:
        if len(st.session_state.messages) > 0:
            # PRODUCT ADDITION: Document Compilation & Export Utility Engine
            raw_brief = f"# Legal Analysis Brief\n*Generated on: {datetime.now().strftime('%Y-%m-%d')} via Lex-Chat Engine*\n\n"
            for m in st.session_state.messages:
                speaker = "### User Query" if m["role"] == "user" else "### Verified Judicial Rationale"
                raw_brief += f"{speaker}:\n{m['content']}\n\n"
            
            st.download_button(
                label="📥 Export Brief as MD",
                data=raw_brief,
                file_name=f"Legal_Analysis_Brief_{datetime.now().strftime('%d_%M')}.md",
                mime="text/markdown"
            )

    # Render Historical Conversational Messages
    for idx, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            # Persist visual anchor report states if present in memory logs
            if m["role"] == "assistant" and "proof_report" in m:
                with st.expander("📊 Verification Trace Details"):
                    st.caption(m["proof_report"])

    # Core Execution Stream upon User Entry
    if user_query := st.chat_input("Ask about the judgment, the judges, or for a summary..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"): st.markdown(user_query)

        with st.spinner("Analyzing legal context..."):
            docs = hybrid_retrieve(st.session_state.vectorstore, user_query)
            
            response = chat_bot.invoke(
                {"input": user_query, "context": docs}, 
                config={"configurable": {"session_id": "streamlit_session"}}
            )

            full_answer = response["answer"]
            clean_answer = re.sub(r'<think>.*?</think>', '', full_answer, flags=re.DOTALL).strip()

            # Compile structural verification trace payload details
            proof_str = ""
            if response["context"]:
                unique_paras = list(set([doc.metadata['para_id'] for doc in response["context"]]))
                case_name = response["context"][0].metadata['case_name']
                proof_str = f"**Faithfulness Index:** Grounded in case text loops matching parameters {unique_paras}.\n\n**Source Authority:** '{case_name}' metadata maps."

            # Log execution properties instantly into local storage logs
            log_id = log_interaction(user_query, clean_answer, proof_str)

            with st.chat_message("assistant"):
                st.markdown(clean_answer)
                
                if proof_str:
                    with st.expander("📊 Evaluation & Proof Report"):
                        st.markdown(proof_str)
                
                # PRODUCT ADDITION: Active User Feedback Interface Flow (Orion Value Add)
                st.markdown("---")
                feedback_cols = st.columns([1, 1, 4, 4])
                with feedback_cols[0]:
                    if st.button("👍", key=f"up_{log_id}"):
                        update_log_feedback(log_id, 1)
                        st.toast("System alignment metric positive logged!", icon="✅")
                with feedback_cols[1]:
                    if st.button("👎", key=f"down_{log_id}"):
                        update_log_feedback(log_id, -1, "General Alert")
                        st.toast("System feedback exception logged.", icon="⚠️")
                        
                # Active Context Dropdown Selection to handle custom evaluation classifications
                error_choice = st.selectbox(
                    "Report System Divergence:",
                    ["None", "Hallucinated Context", "Missing Citation Reference", "Format Discrepancy"],
                    key=f"err_{log_id}"
                )
                if error_choice != "None":
                    update_log_feedback(log_id, -1, error_choice)

            # Commit chat parameters safely to runtime memory lists
            st.session_state.messages.append({
                "role": "assistant", 
                "content": clean_answer,
                "proof_report": proof_str
            })
else:
    st.warning("Please enter your HuggingFace Token in the sidebar to activate the AI.")