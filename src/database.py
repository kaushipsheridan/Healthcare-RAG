"""
Database module for logging RAG queries and responses.
Uses SQLite for simple setup (can switch to PostgreSQL later).
"""

import sqlite3
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Database file path
DB_FILE = "rag_logs.db"


def init_database():
    """
    Initialize the database with required tables.
    Call this once on startup.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create queries log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                retrieval_count INTEGER,
                top_relevance_score REAL,
                latency_ms REAL,
                sources TEXT,
                model_llm TEXT,
                model_embeddings TEXT,
                status TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Database initialized: {DB_FILE}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")


def log_query(
    question: str,
    answer: str,
    retrieval_count: int,
    top_relevance_score: float,
    latency_ms: float,
    sources: list,
    status: str = "success"
) -> bool:
    """
    Log a RAG query to the database.
    
    Args:
        question: User's question
        answer: LLM's answer
        retrieval_count: Number of chunks retrieved
        top_relevance_score: Best relevance score
        latency_ms: Query latency in milliseconds
        sources: List of source documents used
        status: Query status (success/error)
        
    Returns:
        True if logged successfully, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Convert sources list to string for storage
        sources_str = "; ".join([
            f"{s['source']} (chunk {s['chunk_number']}, score: {s['relevance_score']:.2f})"
            for s in sources
        ])
        
        cursor.execute("""
            INSERT INTO query_logs 
            (timestamp, question, answer, retrieval_count, top_relevance_score, 
             latency_ms, sources, model_llm, model_embeddings, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            question,
            answer,
            retrieval_count,
            top_relevance_score,
            latency_ms,
            sources_str,
            "gemini-2.5-flash",
            "models/gemini-embedding-001",
            status
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Query logged successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to log query: {e}")
        return False


def get_query_logs(limit: int = 10) -> list:
    """
    Retrieve recent query logs.
    
    Args:
        limit: Number of recent logs to retrieve
        
    Returns:
        List of query log dictionaries
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM query_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve logs: {e}")
        return []


def get_query_stats() -> Dict:
    """
    Get statistics about queries logged.
    
    Returns:
        Dictionary with stats
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Total queries
        cursor.execute("SELECT COUNT(*) FROM query_logs")
        total_queries = cursor.fetchone()[0]
        
        # Average latency
        cursor.execute("SELECT AVG(latency_ms) FROM query_logs WHERE status='success'")
        avg_latency = cursor.fetchone()[0] or 0
        
        # Average retrieval count
        cursor.execute("SELECT AVG(retrieval_count) FROM query_logs WHERE status='success'")
        avg_retrieval = cursor.fetchone()[0] or 0
        
        # Average top relevance score
        cursor.execute("SELECT AVG(top_relevance_score) FROM query_logs WHERE status='success'")
        avg_score = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_queries": total_queries,
            "average_latency_ms": round(avg_latency, 2),
            "average_retrieval_count": round(avg_retrieval, 2),
            "average_top_relevance_score": round(avg_score, 4)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        return {}