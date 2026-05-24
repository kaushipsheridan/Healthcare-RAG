"""
FastAPI Server for RAG System
Exposes RAG retriever as REST API endpoints
"""

import logging
from typing import Dict
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.mlflow_tracker import log_query_metrics, log_aggregated_metrics
import uvicorn

from src.RAG.retrieval import RAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Initialize FastAPI =====
app = FastAPI(
    title="Healthcare RAG API",
    description="Clinical Document Intelligence System",
    version="1.0.0"
)

# ===== Initialize RAG =====
try:
    rag_retriever = RAGRetriever()
    logger.info("✅ RAG Retriever initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG: {e}")
    rag_retriever = None


# ===== Request/Response Models =====
class QueryRequest(BaseModel):
    """Request format for /query endpoint"""
    question: str
    
    class Config:
        example = {
            "question": "What medications did the patient take?"
        }


class QueryResponse(BaseModel):
    """Response format for /query endpoint"""
    question: str
    answer: str
    sources: list
    retrieval_count: int
    latency_ms: float
    timestamp: str


# ===== Endpoints =====

@app.get("/")
async def root():
    """
    Root endpoint - API info
    """
    return {
        "name": "Healthcare RAG API",
        "version": "1.0.0",
        "status": "✅ Running",
        "endpoints": {
            "GET /": "This info",
            "GET /health": "Health check",
            "POST /query": "Query clinical documents",
            "GET /docs": "Interactive API docs (Swagger UI)"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Status of RAG system
    """
    
    if not rag_retriever or not rag_retriever.vector_store:
        return {
            "status": "❌ Unhealthy",
            "message": "RAG system not initialized. Run ingestion.py first.",
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "status": "✅ Healthy",
        "message": "RAG system ready",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query clinical documents using RAG.
    Logs all queries to database for tracking.
    """
    
    logger.info(f"📥 Received query: {request.question}")
    
    # Validate input
    if not request.question or len(request.question.strip()) == 0:
        logger.error("❌ Empty question")
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    # Check RAG system
    if not rag_retriever or not rag_retriever.vector_store:
        logger.error("❌ RAG system not initialized")
        raise HTTPException(
            status_code=503,
            detail="RAG system not ready. Please try again later."
        )
    
    try:
        # Execute RAG query
        result = rag_retriever.query(request.question)
        logger.info(f"✅ Query successful. Latency: {result['latency_ms']}ms")
        
        # ===== LOG TO DATABASE =====
        from src.database import log_query
        
        top_score = result['sources'][0]['relevance_score'] if result['sources'] else 0
        
        log_query(
            question=result['question'],
            answer=result['answer'],
            retrieval_count=result['retrieval_count'],
            top_relevance_score=top_score,
            latency_ms=result['latency_ms'],
            sources=result['sources'],
            status="success"
        )
        
        # ===== LOG TO MLFLOW =====
        log_query_metrics(
            latency_ms=result['latency_ms'],
            retrieval_count=result['retrieval_count'],
            top_relevance_score=top_score,
            question_length=len(request.question),
            answer_length=len(result['answer'])
        )
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        
        # ===== LOG ERROR =====
        from src.database import log_query
        
        log_query(
            question=request.question,
            answer=f"Error: {str(e)}",
            retrieval_count=0,
            top_relevance_score=0.0,
            latency_ms=0,
            sources=[],
            status="error"
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """
    Get RAG system statistics
    """
    
    if not rag_retriever or not rag_retriever.vector_store:
        return {
            "status": "System not initialized",
            "documents": 0,
            "chunks": 0
        }
    
    try:
        # Try to get collection info
        collection = rag_retriever.vector_store._collection
        return {
            "status": "✅ Healthy",
            "model_embeddings": "models/gemini-embedding-001",
            "model_llm": "gemini-2.5-flash",
            "top_k": rag_retriever.top_k,
            "max_context_length": rag_retriever.max_context_length,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "status": "Error retrieving stats",
            "error": str(e)
        }

@app.get("/logs")
async def get_logs(limit: int = 10):
    """
    Get recent query logs from database.
    
    Args:
        limit: Number of recent logs to return (default: 10)
        
    Returns:
        List of logged queries
    """
    from src.database import get_query_logs
    
    logs = get_query_logs(limit=limit)
    
    return {
        "count": len(logs),
        "logs": logs
    }


@app.get("/logs/stats")
async def get_logs_stats():
    """
    Get statistics about logged queries.
    
    Returns:
        Dictionary with query statistics
    """
    from src.database import get_query_stats
    
    stats = get_query_stats()
    
    return {
        "status": "✅ Stats retrieved",
        "data": stats
    }

# ===== Error Handlers =====

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors"""
    logger.error(f"❌ Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ===== Startup/Shutdown =====

@app.on_event("startup")
async def startup_event():
    """Run on server startup"""
    logger.info("🚀 FastAPI server starting...")
    
    # Initialize database
    from src.database import init_database
    init_database()
    
    # Initialize MLflow
    from src.mlflow_tracker import init_mlflow
    init_mlflow()
    
    if rag_retriever and rag_retriever.vector_store:
        logger.info("✅ RAG system ready")
    else:
        logger.warning("⚠️  RAG system not initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on server shutdown"""
    logger.info("🛑 FastAPI server shutting down...")


# ML FLOW ENDPOINTS
"""
    Start a new MLflow experiment.
    
    Use this to track different configurations:
    - Different chunk sizes
    - Different retrieval K values
    - Different models
    
    Args:
        experiment_name: Name of the experiment
        chunk_size: Document chunk size
        top_k: Number of chunks to retrieve
        
    Returns:
        Run ID for tracking
        
    Example:
        POST /mlflow/start-experiment
        {
            "experiment_name": "chunk_size_512_vs_1024",
            "chunk_size": 512,
            "top_k": 3
        }
"""
@app.post("/mlflow/start-experiment")
async def start_mlflow_experiment(
    experiment_name: str = "default_experiment",
    chunk_size: int = 512,
    top_k: int = 3
    ):
   
    from src.mlflow_tracker import start_experiment
    
    try:
        run_id = start_experiment(
            experiment_name=experiment_name,
            chunk_size=chunk_size,
            top_k=top_k
        )
        
        return {
            "status": "✅ Experiment started",
            "experiment_name": experiment_name,
            "run_id": run_id,
            "parameters": {
                "chunk_size": chunk_size,
                "top_k": top_k
            }
        }
    except Exception as e:
        logger.error(f"❌ Failed to start experiment: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start experiment: {str(e)}"
        )


@app.post("/mlflow/end-experiment")
async def end_mlflow_experiment():
    """
    End the current MLflow experiment run.
    
    Call this when done testing a configuration.
    """
    from src.mlflow_tracker import end_experiment
    from src.database import get_query_stats
    
    try:
        # Get final stats and log them
        stats = get_query_stats()
        
        if stats:
            log_aggregated_metrics(stats)
            logger.info(f"✅ Logged final metrics: {stats}")
        
        end_experiment()
        
        return {
            "status": "✅ Experiment ended",
            "final_stats": stats
        }
    except Exception as e:
        logger.error(f"❌ Failed to end experiment: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to end experiment: {str(e)}"
        )


@app.get("/mlflow/best-run")
async def get_best_run(experiment_name: str = "default_experiment"):
    """
    Get the best run from an experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Best run details (lowest latency)
        
    Example:
        GET /mlflow/best-run?experiment_name=chunk_size_512_vs_1024
    """
    from src.mlflow_tracker import get_best_experiment
    
    try:
        best = get_best_experiment(experiment_name)
        
        if not best:
            raise HTTPException(
                status_code=404,
                detail=f"No runs found for experiment: {experiment_name}"
            )
        
        return {
            "status": "✅ Best run retrieved",
            "experiment_name": experiment_name,
            "best_run": best
        }
    except Exception as e:
        logger.error(f"❌ Failed to get best run: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


@app.get("/mlflow/compare")
async def compare_mlflow_experiments(experiments: str = "exp1,exp2"):
    """
    Compare multiple experiments.
    
    Args:
        experiments: Comma-separated list of experiment names
        
    Returns:
        Comparison of experiments
        
    Example:
        GET /mlflow/compare?experiments=chunk_512,chunk_1024
    """
    from src.mlflow_tracker import compare_experiments
    
    try:
        exp_list = [e.strip() for e in experiments.split(",")]
        comparisons = compare_experiments(exp_list)
        
        if not comparisons:
            raise HTTPException(
                status_code=404,
                detail="No experiments found"
            )
        
        return {
            "status": "✅ Comparison retrieved",
            "experiments_compared": len(comparisons),
            "comparisons": comparisons
        }
    except Exception as e:
        logger.error(f"❌ Failed to compare experiments: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )

# ===== Run Server =====

if __name__ == "__main__":
    logger.info("Starting Healthcare RAG API...")
    
    uvicorn.run(
        "src.API.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )