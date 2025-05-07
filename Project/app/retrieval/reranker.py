from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import CrossEncoder
import numpy as np

from ..core.config import settings

logger = logging.getLogger(__name__)

class Reranker:
    """Re-ranker for improving retrieval precision."""
    
    def __init__(self, model_name: str = settings.RERANKER_MODEL):
        """
        Initialize the re-ranker with a specific model.
        
        Args:
            model_name: Name or path of the re-ranking model
        """
        try:
            self.model = CrossEncoder(model_name)
            logger.info(f"Loaded re-ranker model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading re-ranker model {model_name}: {e}")
            raise
    
    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Re-rank search results based on query relevance.
        
        Args:
            query: Query text
            results: List of retrieval results to re-rank
            top_k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Re-ranked results
        """
        try:
            if not results:
                return []
            
            # Prepare input pairs for the re-ranker
            input_pairs = [(query, result["text"]) for result in results]
            
            # Get scores from the re-ranker
            scores = self.model.predict(input_pairs)
            
            # Add scores to results
            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i])
            
            # Sort results by re-ranking score (descending)
            reranked_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
            
            # Return top-k results
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error re-ranking results: {e}")
            raise