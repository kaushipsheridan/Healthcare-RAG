"""
MLflow Tracking Module
Tracks RAG experiments, metrics, and parameters.

Helps answer: "What configuration works best?"
"""

import logging
import mlflow
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# MLflow tracking URI (default: local)
MLFLOW_TRACKING_URI = "http://localhost:5000"


def init_mlflow(tracking_uri: str = MLFLOW_TRACKING_URI):
    """
    Initialize MLflow tracking.
    
    Args:
        tracking_uri: MLflow server URI
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"✅ MLflow initialized")
        logger.info(f"   Tracking URI: {tracking_uri}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize MLflow: {e}")


def start_experiment(
    experiment_name: str,
    chunk_size: int,
    top_k: int,
    model_embeddings: str = "models/gemini-embedding-001",
    model_llm: str = "gemini-2.5-flash"
) -> str:
    """
    Start a new MLflow experiment run.
    
    Args:
        experiment_name: Name of the experiment
        chunk_size: Document chunk size in tokens
        top_k: Number of chunks to retrieve
        model_embeddings: Embedding model name
        model_llm: LLM model name
        
    Returns:
        Run ID for tracking
    """
    try:
        # Create or get experiment
        mlflow.set_experiment(experiment_name)
        
        # Start a new run
        mlflow.start_run()
        run_id = mlflow.active_run().info.run_id
        
        # Log parameters (configuration)
        mlflow.log_params({
            "chunk_size": chunk_size,
            "top_k": top_k,
            "model_embeddings": model_embeddings,
            "model_llm": model_llm
        })
        
        logger.info(f"✅ Started MLflow run: {run_id}")
        logger.info(f"   Experiment: {experiment_name}")
        logger.info(f"   Parameters: chunk_size={chunk_size}, top_k={top_k}")
        
        return run_id
        
    except Exception as e:
        logger.error(f"❌ Failed to start experiment: {e}")
        return None


def log_query_metrics(
    latency_ms: float,
    retrieval_count: int,
    top_relevance_score: float,
    question_length: int,
    answer_length: int
):
    """
    Log metrics for a single query.
    
    Args:
        latency_ms: Query latency in milliseconds
        retrieval_count: Number of chunks retrieved
        top_relevance_score: Best relevance score
        question_length: Length of question
        answer_length: Length of answer
    """
    try:
        mlflow.log_metrics({
            "latency_ms": latency_ms,
            "retrieval_count": retrieval_count,
            "top_relevance_score": top_relevance_score,
            "question_length": question_length,
            "answer_length": answer_length
        })
    except Exception as e:
        logger.error(f"❌ Failed to log metrics: {e}")


def log_aggregated_metrics(stats: Dict):
    """
    Log aggregated metrics for an experiment.
    
    Args:
        stats: Dictionary with aggregated statistics
    """
    try:
        mlflow.log_metrics({
            "avg_latency_ms": stats.get("average_latency_ms", 0),
            "avg_retrieval_count": stats.get("average_retrieval_count", 0),
            "avg_relevance_score": stats.get("average_top_relevance_score", 0),
            "total_queries": stats.get("total_queries", 0),
            "success_rate": stats.get("success_rate_percent", 0)
        })
        
        logger.info(f"✅ Logged aggregated metrics to MLflow")
        
    except Exception as e:
        logger.error(f"❌ Failed to log aggregated metrics: {e}")


def end_experiment(status: str = "FINISHED"):
    """
    End the current MLflow run.
    
    Args:
        status: Run status (FINISHED, FAILED, etc.)
    """
    try:
        mlflow.end_run()
        logger.info(f"✅ Ended MLflow run with status: {status}")
    except Exception as e:
        logger.error(f"❌ Failed to end run: {e}")


def get_best_experiment(experiment_name: str) -> Optional[Dict]:
    """
    Get the best run from an experiment based on metrics.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary with best run info
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if not experiment:
            logger.warning(f"⚠️  Experiment not found: {experiment_name}")
            return None
        
        # Get all runs for this experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.avg_latency_ms ASC"]
        )
        
        if len(runs) == 0:
            logger.warning(f"⚠️  No runs found for experiment: {experiment_name}")
            return None
        
        best_run = runs.iloc[0]
        
        return {
            "run_id": best_run.run_id,
            "params": dict(best_run.params),
            "metrics": dict(best_run.metrics),
            "status": best_run.status
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get best experiment: {e}")
        return None


def compare_experiments(experiment_names: list) -> list:
    """
    Compare multiple experiments side-by-side.
    
    Args:
        experiment_names: List of experiment names to compare
        
    Returns:
        List of experiment comparisons
    """
    try:
        comparisons = []
        
        for exp_name in experiment_names:
            best = get_best_experiment(exp_name)
            if best:
                comparison = {
                    "experiment": exp_name,
                    "best_run_id": best["run_id"],
                    "params": best["params"],
                    "avg_latency_ms": best["metrics"].get("avg_latency_ms", 0),
                    "avg_score": best["metrics"].get("avg_relevance_score", 0),
                    "success_rate": best["metrics"].get("success_rate", 0)
                }
                comparisons.append(comparison)
        
        return comparisons
        
    except Exception as e:
        logger.error(f"❌ Failed to compare experiments: {e}")
        return []