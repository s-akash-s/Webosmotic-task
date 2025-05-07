import os
import logging
from typing import Dict, List, Any, Optional
import re
import chardet
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .processor import BaseDocumentProcessor, Document, DocumentChunk
from ..core.config import settings

logger = logging.getLogger(__name__)

class TxtProcessor(BaseDocumentProcessor):
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len
        )
    
    def process(self, file_path: str) -> Document:
        """
        Process a TXT file and extract text and metadata.
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            Document: A Document object with extracted text and metadata
        """
        logger.info(f"Processing TXT file: {file_path}")
        
        try:
       
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding_result = chardet.detect(raw_data)
                encoding = encoding_result['encoding']
            
            
            with open(file_path, 'r', encoding=encoding) as file:
                text_content = file.read()
        
            file_stats = os.stat(file_path)
            
   
            doc_metadata = {
                "source": os.path.basename(file_path),
                "file_path": file_path,
                "file_type": "txt",
                "file_size": file_stats.st_size,
                "creation_date": str(file_stats.st_ctime),
                "modification_date": str(file_stats.st_mtime),
            }

            chars_per_page = 3000
            page_count = max(1, len(text_content) // chars_per_page)
            doc_metadata["page_count"] = page_count
            
         
            processed_text = ""
            for i in range(page_count):
                start_idx = i * chars_per_page
                end_idx = min((i + 1) * chars_per_page, len(text_content))
                page_text = text_content[start_idx:end_idx]
                processed_text += f"--- Page {i+1} ---\n{page_text}\n\n"
            
            return Document(processed_text, doc_metadata)
            
        except Exception as e:
            logger.error(f"Error processing TXT file: {e}")
            raise
    
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
         
                page_numbers = []
                page_pattern = re.compile(r"--- Page (\d+) ---")
                for match in page_pattern.finditer(chunk_text):
                    page_numbers.append(int(match.group(1)))
                
          
                if not page_numbers:
            
                    total_length = len(document.text)
                    chunk_start = document.text.find(chunk_text)
                    relative_position = chunk_start / total_length if total_length > 0 else 0
                    estimated_page = max(1, min(
                        round(relative_position * document.metadata.get("page_count", 1)), 
                        document.metadata.get("page_count", 1)
                    ))
                    page_numbers = [estimated_page]
                
             
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    "chunk_id": i,
                    "pages": page_numbers,
                    "page": page_numbers[0] if page_numbers else 1,  
                })
                
                doc_chunks.append(DocumentChunk(chunk_text, chunk_metadata))
            
            return doc_chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise