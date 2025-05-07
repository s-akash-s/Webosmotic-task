# test_document_processors.py
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the app directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Import our components
from app.document_processing.processor import BaseDocumentProcessor
from app.document_processing.pdf_processor import PDFProcessor
from app.document_processing.docx_processor import DocxProcessor
from app.document_processing.txt_processor import TxtProcessor

def test_processor(file_path):
    """Test processing a specific file"""
    logger.info(f"Testing processor for file: {file_path}")
    
    try:
        # Get the appropriate processor
        processor = BaseDocumentProcessor.get_processor_for_file(file_path)
        logger.info(f"Using processor: {processor.__class__.__name__}")
        
        # Process the document
        document = processor.process(file_path)
        logger.info(f"Document processed successfully")
        logger.info(f"Metadata: {document.metadata}")
        logger.info(f"Text length: {len(document.text)} characters")
        
        # Print first 200 characters of the document
        logger.info(f"Text sample: {document.text[:200]}...")
        
        # Test chunking
        chunks = processor.chunk(document)
        logger.info(f"Document chunked into {len(chunks)} chunks")
        
        # Print info about the first chunk
        if chunks:
            logger.info(f"First chunk text sample: {chunks[0].text[:100]}...")
            logger.info(f"First chunk metadata: {chunks[0].metadata}")
        
        return True
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return False

if __name__ == "__main__":
    # Create test directory if it doesn't exist
    test_dir = "test_files"
    os.makedirs(test_dir, exist_ok=True)
    
    # Test each processor type
    # Note: You'll need to provide actual test files in these locations
    test_files = [
        "Free_Test_Data_100KB_PDF.pdf",    # Replace with actual PDF file path
        "sample-files.com-basic-text.docx",   # Replace with actual DOCX file path
        "Part_1.txt"     # Replace with actual TXT file path
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            result = test_processor(file_path)
            logger.info(f"Test result for {file_path}: {'Success' if result else 'Failure'}")
        else:
            logger.warning(f"Test file not found: {file_path}")