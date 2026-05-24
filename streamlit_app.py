"""
Streamlit Web UI for Healthcare RAG System
Allows doctors to upload documents and ask questions.
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, List

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

# ===== Helper Functions =====

def check_api_health() -> bool:
    """Check if FastAPI server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except Exception:
        return False


def query_rag(question: str) -> Dict:
    """
    Send question to RAG API and get answer.
    
    Args:
        question: User's question
        
    Returns:
        Response dict with answer and sources
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=180  # 3 minutes timeout
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


def get_api_stats() -> Dict:
    """Get stats from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}


def get_query_logs(limit: int = 10) -> Dict:
    """Get query logs from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/logs?limit={limit}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"logs": []}


# ===== Main App =====

# Header
st.title("🏥 Healthcare RAG Assistant")
st.markdown("Ask questions about clinical documents. Get grounded answers with sources.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Health Check
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Not Connected")
        st.write("Make sure FastAPI is running: `python -m src.API.main`")
    
    st.divider()
    
    # About
    st.header("ℹ️ About")
    st.write("""
    This is a **Retrieval-Augmented Generation (RAG)** system that:
    
    1. **Retrieves** relevant documents from a vector database
    2. **Augments** the LLM with actual document content
    3. **Generates** grounded answers (not hallucinations)
    
    **Key Features:**
    - Semantic search (understands meaning, not just keywords)
    - Source attribution (shows which docs were used)
    - Relevance scores (confidence in results)
    - Query tracking (all questions logged)
    """)
    
    st.divider()
    
    # Stats
    st.header("📊 System Stats")
    stats = get_api_stats()
    if stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", "Gemini 2.5 Flash")
        with col2:
            st.metric("Database", "ChromaDB")
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Embeddings", "Gemini")
        with col4:
            st.metric("Logging", "SQLite")

# Main Content
if not api_healthy:
    st.error("⚠️ RAG API is not running!")
    st.info("Start the API with: `python -m src.API.main`")
else:
    # Two columns layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Question")
        
        # Question input
        question = st.text_area(
            "Enter your question about the clinical documents:",
            placeholder="e.g., What medications did the patient take?",
            height=100
        )
        
        # Submit button
        if st.button("🔍 Search Documents", use_container_width=True):
            if question.strip():
                with st.spinner("🤔 Searching documents and generating answer..."):
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
                    st.write(result.get('message', ''))
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
                    
                    if result['sources']:
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(
                                f"📄 Source {i}: {source['source']} "
                                f"(Relevance: {source['relevance_score']:.2%})"
                            ):
                                st.write(f"**Chunk:** {source['chunk_number']}")
                                st.write(f"**Relevance Score:** {source['relevance_score']:.4f}")
                                st.markdown("**Preview:**")
                                st.write(source['preview'])
                    else:
                        st.info("No sources found")
                    
                    # Metadata
                    st.markdown("### 📊 Query Metadata")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Latency",
                            f"{result['latency_ms']:.0f}ms"
                        )
                    
                    with col2:
                        st.metric(
                            "Documents Retrieved",
                            result['retrieval_count']
                        )
                    
                    with col3:
                        if result['sources']:
                            top_score = result['sources'][0]['relevance_score']
                            st.metric(
                                "Top Relevance",
                                f"{top_score:.2%}"
                            )
            else:
                st.warning("Please enter a question!")
    
    with col2:
        st.header("📋 Query History")
        
        if st.session_state.query_history:
            # Show recent queries
            for i, entry in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(
                    f"Q{len(st.session_state.query_history)-i}: "
                    f"{entry['question'][:40]}..."
                ):
                    st.write(f"**Time:** {entry['timestamp'].strftime('%H:%M:%S')}")
                    if "error" not in entry['result']:
                        st.write(f"**Sources:** {entry['result']['retrieval_count']}")
                        st.write(f"**Latency:** {entry['result']['latency_ms']:.0f}ms")
        else:
            st.info("No queries yet. Ask a question to start!")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Healthcare RAG Assistant | Powered by Gemini + ChromaDB | Last Updated: 2026-05-24
</div>
""", unsafe_allow_html=True)