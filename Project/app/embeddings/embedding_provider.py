from typing import List, Dict, Any, Optional, Union
import logging
import numpy as np
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import os
import uuid

from ..core.config import settings
from ..document_processing.processor import DocumentChunk

logger = logging.getLogger(__name__)

class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        pass

class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using local sentence-transformers models."""
    
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        """
        Initialize the embedding provider with a specific model.
        
        Args:
            model_name: Name or path of the sentence-transformers model
        """
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {e}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
           
            embeddings = self.model.encode(texts, normalize_embeddings=True)

            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

class EmbeddingService:
    """Service for managing document embeddings."""
    
    def __init__(self):
        """Initialize the embedding service with the configured provider."""
        self.provider_type = settings.EMBEDDING_PROVIDER
        
        if self.provider_type == "local":
            self.provider = LocalEmbeddingProvider()
        else:
            # Placeholder for other providers (OpenAI, etc.)
            raise ValueError(f"Unsupported embedding provider: {self.provider_type}")
    
    def embed_document_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Embed document chunks and prepare them for storage.
        
        Args:
            chunks: List of document chunks to embed
            
        Returns:
            Dict containing document ID, embeddings, and metadata
        """
        try:
        
            texts = [chunk.text for chunk in chunks]
            
    
            embeddings = self.provider.get_embeddings(texts)
      
            document_id = str(uuid.uuid4())
            
          
            chunk_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_data.append({
                    "id": f"{document_id}_{i}",
                    "text": chunk.text,
                    "embedding": embedding,
                    "metadata": chunk.metadata
                })
            
            return {
                "document_id": document_id,
                "chunks": chunk_data
            }
            
        except Exception as e:
            logger.error(f"Error embedding document: {e}")
            raise