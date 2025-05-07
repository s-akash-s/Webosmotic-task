# tests/test_chunking.py
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.document_processing.hierarchical_chunker import HierarchicalChunker
from app.document_processing.processor import Document
from app.document_processing.txt_processor import TxtProcessor
from app.document_processing.pdf_processor import PDFProcessor
from app.document_processing.docx_processor import DocxProcessor

def test_hierarchical_chunking():
    """Test the hierarchical chunking strategy"""
    logger.info("Testing hierarchical chunking")
    
    # Sample text with page markers
    sample_text = """
    --- Page 1 ---
    # Chapter 1: Introduction
    
    This is the introduction to our document. This section provides an overview of the topic.
    
    ## Section 1.1: Background
    
    Here we discuss the background of the topic. This includes historical context and foundational concepts.
    
    --- Page 2 ---
    ## Section 1.2: Objectives
    
    The main objectives of this document are:
    1. To explain the concept clearly
    2. To provide examples
    3. To demonstrate the application
    
    --- Page 3 ---
    # Chapter 2: Methodology
    
    In this chapter, we outline the methodology used in our approach.
    
    ## Section 2.1: Data Collection
    
    Data was collected through various means including surveys, interviews, and document analysis.
    
    --- Page 4 ---
    ## Section 2.2: Analysis
    
    The analysis was conducted using both qualitative and quantitative methods.
    """
    
    # Initialize chunker
    chunker = HierarchicalChunker()
    
    chunks = chunker.create_hierarchical_chunks(sample_text, {"source": "test.txt"})
    
    # Log results
    logger.info(f"Created {len(chunks['parents'])} parent chunks and {len(chunks['children'])} child chunks")
    
    # Print parent chunks
    for i, parent in enumerate(chunks['parents']):
        logger.info(f"Parent {i}:")
        logger.info(f"Text (first 100 chars): {parent['text'][:100]}...")
        logger.info(f"Metadata: {parent['metadata']}")
    
    # Print some child chunks
    for i, child in enumerate(chunks['children'][:3]):
        logger.info(f"Child {i}:")
        logger.info(f"Text (first 100 chars): {child['text'][:100]}...")
        logger.info(f"Metadata: {child['metadata']}")
    
    return chunks

def test_document_processors():
    """Test the document processors with hierarchical chunking"""
    logger.info("Testing document processors with hierarchical chunking")
    
    # Create a sample TXT file
    sample_file = "test_sample.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write("""
        # Document Title
        
        This is a sample document to test the document processors with hierarchical chunking.
        
        ## Section 1
        
        This is the first section of the document. It contains some sample text that will be chunked.
        The chunking should preserve the semantic structure of the document.
        
        ## Section 2
        
        This is the second section of the document. It also contains sample text for chunking.
        The hierarchical chunker should create parent chunks for larger context and child chunks for specific information.
        """)
    
    # Test all processors
    processors = [
        ("TXT", TxtProcessor()),
        # Add other processors as needed, if test files are available
    ]
    
    for name, processor in processors:
        logger.info(f"Testing {name} processor")
        
        # Process file
        document = processor.process(sample_file)
        logger.info(f"Processed document: {document.metadata['source']}")
        
        # Chunk document
        chunks = processor.chunk(document)
        
        # For hierarchical chunkers
        if hasattr(chunks, '__len__'):
            logger.info(f"Created {len(chunks)} chunks")
            # Display a sample of chunks
            for i, chunk in enumerate(chunks[:2]):
                logger.info(f"Chunk {i}:")
                logger.info(f"Text (first 100 chars): {chunk.text[:100]}...")
                logger.info(f"Metadata: {chunk.metadata}")
        else:
            logger.info("Chunking method returned a non-list result")
    
    # Clean up
    os.remove(sample_file)

if __name__ == "__main__":
    test_hierarchical_chunking()
    test_document_processors()