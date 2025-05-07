# app/document_processing/hierarchical_chunker.py
from typing import List, Dict, Any, Optional
import logging
from .semantic_chunker import SemanticChunker
from .processor import DocumentChunk

logger = logging.getLogger(__name__)

class HierarchicalChunker:
    """
    Creates a hierarchical structure of chunks with parent-child relationships.
    Parent chunks provide broader context, while child chunks provide detailed information.
    """
    
    def __init__(
        self,
        parent_max_size: int = 2000,
        child_max_size: int = 500,
        overlap_size: int = 100
    ):
        """
        Initialize the hierarchical chunker.
        
        Args:
            parent_max_size: Maximum size of parent chunks
            child_max_size: Maximum size of child chunks
            overlap_size: Overlap size between chunks
        """
        self.parent_chunker = SemanticChunker(
            max_chunk_size=parent_max_size,
            min_chunk_size=parent_max_size // 4,
            overlap_size=overlap_size
        )
        
        self.child_chunker = SemanticChunker(
            max_chunk_size=child_max_size,
            min_chunk_size=child_max_size // 4,
            overlap_size=overlap_size // 2
        )
        
    def create_hierarchical_chunks(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create a hierarchical structure of chunks.
        
        Args:
            text: Input text
            metadata: Metadata to include with chunks
            
        Returns:
            Dictionary with parent and child chunks
        """
        if metadata is None:
            metadata = {}
            
      
        parent_chunks = self.parent_chunker.create_chunks(text, metadata)
     
        all_child_chunks = []
        
        for i, parent in enumerate(parent_chunks):
            parent_text = parent["text"]
            parent_metadata = parent["metadata"].copy()
            

            parent_metadata["parent_index"] = i
            
       
            child_chunks = self.child_chunker.create_chunks(parent_text, parent_metadata)
            
            all_child_chunks.extend(child_chunks)
            
        logger.info(f"Created hierarchical structure with {len(parent_chunks)} parents and {len(all_child_chunks)} children")
        
        return {
            "parents": parent_chunks,
            "children": all_child_chunks
        }
    
    def create_document_chunks(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """
        Create document chunks compatible with the existing API.
        
        Args:
            text: Input text
            metadata: Metadata to include with chunks
            
        Returns:
            List of DocumentChunk objects
        """
        if metadata is None:
            metadata = {}
   
        chunks = self.create_hierarchical_chunks(text, metadata)
        

        doc_chunks = []
        
   
        for i, parent in enumerate(chunks["parents"]):
            parent_metadata = parent["metadata"].copy()
            parent_metadata["chunk_type"] = "parent"
            parent_metadata["chunk_id"] = f"parent_{i}"
            
            doc_chunks.append(DocumentChunk(parent["text"], parent_metadata))
        

        for i, child in enumerate(chunks["children"]):
            child_metadata = child["metadata"].copy()
            child_metadata["chunk_type"] = "child"
            child_metadata["chunk_id"] = f"child_{i}"
            
            doc_chunks.append(DocumentChunk(child["text"], child_metadata))
            
        return doc_chunks