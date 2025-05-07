from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import os
import logging
from ..core.config import settings

logger = logging.getLogger(__name__)

class Document:
    """Class representing a processed document with text and metadata."""
    
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

class DocumentChunk:
    """Class representing a chunk of a document with text and metadata."""
    
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

class BaseDocumentProcessor(ABC):
    """Abstract base class for document processors."""
    
    @abstractmethod
    def process(self, file_path: str) -> Document:
        """
        Process a document file and extract text and metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document: A Document object with extracted text and metadata
        """
        pass
    
    @abstractmethod
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Split a document into chunks for embedding.
        
        Args:
            document: Document object to chunk
            
        Returns:
            List[DocumentChunk]: List of document chunks
        """
        pass
    
    def get_processor_for_file(file_path: str) -> 'BaseDocumentProcessor':
        """
        Factory method to get the appropriate processor for a file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            BaseDocumentProcessor: An instance of the appropriate processor
        """
        from .pdf_processor import PDFProcessor
        from .docx_processor import DocxProcessor
        from .txt_processor import TxtProcessor
        
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return PDFProcessor()
        elif ext == '.docx':
            return DocxProcessor()
        elif ext == '.txt':
            return TxtProcessor()
        else:
            raise ValueError(f"Unsupported file format: {ext}")