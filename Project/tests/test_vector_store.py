# test_vector_store.py
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Import our components
from app.document_processing.processor import BaseDocumentProcessor
from app.embeddings.embedding_provider import EmbeddingService
from app.retrieval.vector_store import VectorStore

def test_vector_store(file_path):
    """Test storing and retrieving document embeddings"""
    logger.info(f"Testing vector store with file: {file_path}")
    
    try:
        # Get the appropriate processor
        processor = BaseDocumentProcessor.get_processor_for_file(file_path)
        
        # Process the document
        document = processor.process(file_path)
        
        # Chunk the document
        chunks = processor.chunk(document)
        
        # Initialize embedding service
        embedding_service = EmbeddingService()
        
        # Generate embeddings
        document_data = embedding_service.embed_document_chunks(chunks)
        
        # Initialize vector store
        vector_store = VectorStore()
        
        # Add document to vector store
        document_id = vector_store.add_document(document_data)
        logger.info(f"Added document to vector store with ID: {document_id}")
        
        # Test query
        test_query = "sample"  # Replace with a relevant query for your document
        query_embedding = embedding_service.provider.get_embeddings([test_query])[0]
        
        results = vector_store.query(
            query_text=test_query,
            embedding=query_embedding,
            document_id=document_id,
            n_results=3
        )
        
        logger.info(f"Query returned {len(results)} results")
        
        if results:
            logger.info(f"First result text sample: {results[0]['text'][:100]}...")
        
        return results
    except Exception as e:
        logger.error(f"Error in vector store test: {e}")
        return None

if __name__ == "__main__":
    # Test with a sample file
    test_file = "D:/Webosmotic-task/Part_1.txt"  # Replace with an actual file path
    
    if os.path.exists(test_file):
        results = test_vector_store(test_file)
        if results:
            logger.info("Vector store test successful")
        else:
            logger.error("Vector store test failed")
    else:
        logger.warning(f"Test file not found: {test_file}")