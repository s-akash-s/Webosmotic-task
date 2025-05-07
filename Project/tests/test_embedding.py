# test_embedding.py
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
from app.embeddings.embedding_provider import EmbeddingService

def test_embedding_service(file_path):
    """Test embedding a document"""
    logger.info(f"Testing embedding for file: {file_path}")
    
    try:
        # Get the appropriate processor
        processor = BaseDocumentProcessor.get_processor_for_file(file_path)
        
        # Process the document
        document = processor.process(file_path)
        
        # Chunk the document
        chunks = processor.chunk(document)
        logger.info(f"Document chunked into {len(chunks)} chunks")
        
        # Initialize embedding service
        embedding_service = EmbeddingService()
        
        # Generate embeddings
        document_data = embedding_service.embed_document_chunks(chunks)
        logger.info(f"Generated embeddings for {len(document_data['chunks'])} chunks")
        logger.info(f"Document ID: {document_data['document_id']}")
        
        # Check first embedding
        first_embedding = document_data['chunks'][0]['embedding']
        logger.info(f"First embedding dimension: {len(first_embedding)}")
        logger.info(f"First embedding sample: {first_embedding[:5]}...")
        
        return document_data
    except Exception as e:
        logger.error(f"Error embedding document: {e}")
        return None

if __name__ == "__main__":
    # Test with a sample file
    test_file = "Part_1.txt"  # Replace with an actual file path
    
    if os.path.exists(test_file):
        result = test_embedding_service(test_file)
        if result:
            logger.info("Embedding test successful")
        else:
            logger.error("Embedding test failed")
    else:
        logger.warning(f"Test file not found: {test_file}")