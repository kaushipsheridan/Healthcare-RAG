"""
Streamlit Web UI for Healthcare RAG System
Allows doctors to upload documents and ask questions.
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, List
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.pdf_handler import PDFUploadHandler
from src.RAG.retrieval import RAGRetriever

# ===== Page Config =====
st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Styling =====
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
        padding: 0.75rem;
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #1557a0;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #17a2b8;
    }
    .metric-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# ===== API Config =====
API_BASE_URL = "http://localhost:8000"

# ===== Session State =====
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

if "rag_retriever" not in st.session_state:
    st.session_state.rag_retriever = None

if "pdf_handler" not in st.session_state:
    st.session_state.pdf_handler = PDFUploadHandler()

# ===== Helper Functions =====

def check_api_health() -> bool:
    """Check if FastAPI server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def query_rag(question: str) -> Dict:
    """Send question to RAG API and get answer."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=180
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"API Error: {response.status_code}",
                "message": response.text
            }
            
    except requests.exceptions.Timeout:
        return {"error": "Request timed out", "message": "Query took too long"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection Error", "message": "Cannot reach RAG API"}
    except Exception as e:
        return {"error": "Error", "message": str(e)}


def query_local_rag(question: str, rag_retriever) -> Dict:
    """Query using local RAG retriever (for uploaded PDFs)."""
    try:
        result = rag_retriever.query(question)
        return result
    except Exception as e:
        return {"error": "Error", "message": str(e)}


def get_api_stats() -> Dict:
    """Get stats from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}

# ===== Main App =====

# Header
st.title("🏥 Healthcare RAG Assistant")
st.markdown("Upload clinical documents or ask questions about pre-loaded documents.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings & Upload")
    
    # API Health Check
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Not Connected")
    
    st.divider()
    
    # Mode Selection
    st.header("📂 Document Mode")
    
    mode = st.radio(
        "Choose document source:",
        ["Use Sample Document", "Upload Custom PDF"]
    )
    
    st.divider()
    
    # PDF Upload Section
    if mode == "Upload Custom PDF":
        st.header("📤 Upload PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a clinical document (PDF format)"
        )
        
        if uploaded_file is not None:
            st.info(f"📄 File selected: {uploaded_file.name}")
            
            if st.button("📥 Ingest PDF", use_container_width=True):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Extract text
                        pdf_text = st.session_state.pdf_handler.extract_text_from_pdf(
                            uploaded_file
                        )
                        st.success(f"✅ Extracted {len(pdf_text)} characters")
                        
                        # Ingest into ChromaDB
                        success, message, chunk_count = st.session_state.pdf_handler.ingest_pdf_content(
                            pdf_text=pdf_text,
                            pdf_name=uploaded_file.name,
                            collection_name="user_uploads"
                        )
                        
                        if success:
                            st.success(f"✅ {message}")
                            st.session_state.current_pdf = uploaded_file.name
                            
                            # Create RAG retriever for uploaded PDF
                            vector_store = st.session_state.pdf_handler.get_vector_store(
                                collection_name="user_uploads"
                            )
                            
                            if vector_store:
                                # Create a temporary RAG retriever
                                class UploadedRAGRetriever:
                                    def __init__(self, vector_store):
                                        self.vector_store = vector_store
                                        from langchain_google_genai import ChatGoogleGenerativeAI
                                        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                                        self.top_k = 3
                                    
                                    def query(self, question: str) -> Dict:
                                        from datetime import datetime
                                        start_time = datetime.now()
                                        
                                        # Retrieve
                                        results = self.vector_store.similarity_search_with_relevance_scores(
                                            question, k=self.top_k
                                        )
                                        
                                        # Build prompt
                                        context_parts = []
                                        chunks = []
                                        for i, (doc, score) in enumerate(results):
                                            context_parts.append(
                                                f"--- DOCUMENT {i+1} ---\n{doc.page_content}\n"
                                            )
                                            chunks.append({
                                                "content": doc.page_content,
                                                "metadata": doc.metadata,
                                                "relevance_score": float(score),
                                                "rank": i + 1
                                            })
                                        
                                        context = "\n".join(context_parts)
                                        
                                        prompt = f"""Answer based ONLY on these documents:

{context}

Question: {question}

Answer:"""
                                        
                                        # Generate answer
                                        response = self.llm.invoke(prompt)
                                        answer = response.content
                                        
                                        end_time = datetime.now()
                                        latency_ms = (end_time - start_time).total_seconds() * 1000
                                        
                                        return {
                                            "question": question,
                                            "answer": answer,
                                            "sources": [
                                                {
                                                    "rank": chunk["rank"],
                                                    "source": chunk["metadata"].get("source"),
                                                    "chunk_number": chunk["metadata"].get("chunk_number"),
                                                    "relevance_score": chunk["relevance_score"],
                                                    "preview": chunk["content"][:200] + "..."
                                                }
                                                for chunk in chunks
                                            ],
                                            "retrieval_count": len(chunks),
                                            "latency_ms": round(latency_ms, 2),
                                            "timestamp": start_time.isoformat()
                                        }
                                
                                st.session_state.rag_retriever = UploadedRAGRetriever(vector_store)
                        else:
                            st.error(f"❌ {message}")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        if st.session_state.current_pdf:
            st.success(f"✅ Using: {st.session_state.current_pdf}")
    
    st.divider()
    
    # About
    st.header("ℹ️ About")
    st.write("""
    **RAG System** (Retrieval-Augmented Generation):
    - Retrieves relevant document chunks
    - Grounds LLM in actual documents
    - Prevents hallucinations
    - Shows sources for answers
    """)


# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask a Question")
    
    question = st.text_area(
        "Enter your question:",
        placeholder="e.g., What medications did the patient take?",
        height=100
    )
    
    if st.button("🔍 Search Documents", use_container_width=True):
        if question.strip():
            with st.spinner("🤔 Searching and generating answer..."):
                # Choose RAG source
                if mode == "Upload Custom PDF" and st.session_state.rag_retriever:
                    # Use local RAG for uploaded PDF
                    result = query_local_rag(question, st.session_state.rag_retriever)
                else:
                    # Use API RAG for sample document
                    result = query_rag(question)
            
            # Add to history
            st.session_state.query_history.append({
                "timestamp": datetime.now(),
                "question": question,
                "result": result
            })
            
            # Display result
            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                # Answer
                st.markdown("### 📝 Answer")
                st.markdown(f"""
                <div class="answer-box">
                {result['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Sources
                st.markdown("### 📚 Sources Used")
                
                if result.get('sources'):
                    for i, source in enumerate(result['sources'], 1):
                        with st.expander(
                            f"📄 Source {i}: {source.get('source', 'unknown')} "
                            f"(Relevance: {source.get('relevance_score', 0):.2%})"
                        ):
                            st.write(f"**Chunk:** {source.get('chunk_number', 'N/A')}")
                            st.write(f"**Relevance Score:** {source.get('relevance_score', 0):.4f}")
                            st.markdown("**Preview:**")
                            st.write(source.get('preview', 'N/A'))
                
                # Metadata
                st.markdown("### 📊 Query Metadata")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Latency", f"{result.get('latency_ms', 0):.0f}ms")
                
                with col2:
                    st.metric("Documents", result.get('retrieval_count', 0))
                
                with col3:
                    if result.get('sources'):
                        top_score = result['sources'][0]['relevance_score']
                        st.metric("Top Score", f"{top_score:.2%}")
        else:
            st.warning("Please enter a question!")

with col2:
    st.header("📋 History")
    
    if st.session_state.query_history:
        for i, entry in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(
                f"Q{len(st.session_state.query_history)-i}: "
                f"{entry['question'][:30]}..."
            ):
                st.write(f"**Time:** {entry['timestamp'].strftime('%H:%M:%S')}")
    else:
        st.info("No queries yet")

st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Healthcare RAG Assistant | Powered by Gemini + ChromaDB
</div>
""", unsafe_allow_html=True)