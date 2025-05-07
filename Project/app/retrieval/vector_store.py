# app/retrieval/vector_store.py
from typing import List, Dict, Any, Optional, Union
import logging
import os
import chromadb
from chromadb.utils import embedding_functions
import uuid
import json

from ..core.config import settings
from ..document_processing.processor import DocumentChunk

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for document embeddings using ChromaDB."""
    
    def __init__(self, collection_name: str = "documents"):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to use
        """
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Initialized vector store with collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_document(self, document_data: Dict[str, Any]) -> str:
        """
        Add a document to the vector store.
        
        Args:
            document_data: Document data including embeddings and metadata
            
        Returns:
            str: Document ID
        """
        try:
            document_id = document_data["document_id"]
            chunks = document_data["chunks"]
            
            # Prepare data for batch addition
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for chunk in chunks:
                ids.append(chunk["id"])
                embeddings.append(chunk["embedding"])
                
                # Prepare metadata (ensure it's JSON serializable)
                metadata = chunk["metadata"].copy()
                
                # Ensure document_id is added to metadata for filtering
                metadata["document_id"] = document_id
                
                for key, value in metadata.items():
                    # Convert lists to strings
                    if isinstance(value, list):
                        metadata[key] = str(value)
                    # Convert other non-primitive types to strings
                    elif not isinstance(value, (str, int, float, bool, type(None))):
                        metadata[key] = str(value)
                
                metadatas.append(metadata)
                documents.append(chunk["text"])
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"Added document to vector store: {document_id} with {len(chunks)} chunks")
            
            return document_id
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {e}")
            raise
    
    def query(self, query_text: str, embedding: List[float], document_id: Optional[str] = None, 
          n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_text: Query text
            embedding: Query embedding vector
            document_id: Optional document ID to filter results
            n_results: Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of query results
        """
        try:
            # Set up filter if document_id is provided
            filter_condition = None
            if document_id:
                # Use explicit metadata filtering with ChromaDB
                filter_condition = {"document_id": document_id}
                
            # Query the collection with explicit filter
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where=filter_condition  # Use ChromaDB's native filtering
            )
            
            # Process and return results
            processed_results = []
            
            if not results or not results['ids'] or not results['ids'][0]:
                # Log more details about the empty results
                logger.warning(f"Empty results from ChromaDB. Document ID: {document_id}, Query: {query_text[:50]}...")
                return []
            
            for i in range(len(results['ids'][0])):
                # Extract result data
                result_id = results['ids'][0][i]
                result_text = results['documents'][0][i]
                result_metadata = results['metadatas'][0][i]
                result_distance = results['distances'][0][i] if 'distances' in results else None
                
                # Process metadata - convert string representations back to lists
                for key, value in result_metadata.items():
                    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                        try:
                            import ast
                            result_metadata[key] = ast.literal_eval(value)
                        except (SyntaxError, ValueError):
                            pass
                
                processed_results.append({
                    "id": result_id,
                    "text": result_text,
                    "metadata": result_metadata,
                    "distance": result_distance
                })
            
            logger.info(f"Query returned {len(processed_results)} results")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            raise