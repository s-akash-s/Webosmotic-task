import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os
import logging
from typing import Dict, List, Any, Optional
import tempfile
from pathlib import Path

from .processor import BaseDocumentProcessor, Document, DocumentChunk
from ..core.config import settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class PDFProcessor(BaseDocumentProcessor):
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len
        )
    
    def process(self, file_path: str) -> Document:
        """
        Process a PDF file and extract text and metadata.
        If the PDF is scanned (has no text), OCR will be applied.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Document: A Document object with extracted text and metadata
        """
        logger.info(f"Processing PDF file: {file_path}")
        
        try:
            with open(file_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                metadata = reader.metadata if reader.metadata else {}
                
                doc_metadata = {
                    "source": os.path.basename(file_path),
                    "file_path": file_path,
                    "file_type": "pdf",
                    "page_count": len(reader.pages),
                    "title": metadata.get('/Title', ''),
                    "author": metadata.get('/Author', ''),
                    "creation_date": metadata.get('/CreationDate', ''),
                }
                
                text_content = []
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    
                    if not page_text or page_text.isspace():
                        logger.info(f"Page {page_num+1} has no text, applying OCR")
                        page_text = self._process_scanned_page(file_path, page_num)
                    
                    text_content.append(f"--- Page {page_num+1} ---\n{page_text}")
                
                document_text = "\n\n".join(text_content)
                
                if not document_text or document_text.isspace():
                    logger.info("Document appears to be fully scanned, applying OCR to all pages")
                    document_text = self._process_scanned_document(file_path)
                
                return Document(document_text, doc_metadata)
                
        except Exception as e:
            logger.error(f"Error processing PDF file: {e}")
            raise
    
    def _process_scanned_page(self, pdf_path: str, page_num: int) -> str:
        """Apply OCR to a specific page of a PDF."""
        try:
            images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
            
            if not images:
                return ""
            
            return pytesseract.image_to_string(images[0], lang=settings.OCR_LANGUAGE)
            
        except Exception as e:
            logger.error(f"Error performing OCR on page {page_num}: {e}")
            return ""
    
    def _process_scanned_document(self, pdf_path: str) -> str:
        """Apply OCR to an entire PDF document."""
        try:
            images = convert_from_path(pdf_path)
            
            text_content = []
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang=settings.OCR_LANGUAGE)
                text_content.append(f"--- Page {i+1} ---\n{page_text}")
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Error performing OCR on document: {e}")
            return ""
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Split a document into chunks for embedding.
        
        Args:
            document: Document object to chunk
            
        Returns:
            List[DocumentChunk]: List of document chunks
        """
        logger.info(f"Chunking document: {document.metadata.get('source', 'unknown')}")
        
        try:
            chunks = self.text_splitter.split_text(document.text)
            
            doc_chunks = []
            for i, chunk_text in enumerate(chunks):
                
                page_markers = [f"--- Page {p} ---" for p in range(1, document.metadata.get("page_count", 1) + 1)]
                
          
                pages = []
                for page_num, marker in enumerate(page_markers, 1):
                    if marker in chunk_text:
                        pages.append(page_num)
                
                if not pages:
                    total_length = len(document.text)
                    chunk_start = document.text.find(chunk_text)
                    relative_position = chunk_start / total_length if total_length > 0 else 0
                    estimated_page = max(1, min(
                        round(relative_position * document.metadata.get("page_count", 1)), 
                        document.metadata.get("page_count", 1)
                    ))
                    pages = [estimated_page]
                
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    "chunk_id": i,
                    "pages": pages,
                    "page": pages[0] if pages else 1, 
                })
                
                doc_chunks.append(DocumentChunk(chunk_text, chunk_metadata))
            
            return doc_chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise