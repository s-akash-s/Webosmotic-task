# test_reranker.py
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
from app.retrieval.reranker import Reranker

def test_reranker(file_path):
    """Test the re-ranker with retrieved results"""
    logger.info(f"Testing re-ranker with file: {file_path}")
    
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
        
        # Test query
        test_query = "sample"  # Replace with a relevant query for your document
        query_embedding = embedding_service.provider.get_embeddings([test_query])[0]
        
        # Get initial results
        results = vector_store.query(
            query_text=test_query,
            embedding=query_embedding,
            document_id=document_id,
            n_results=5
        )
        
        logger.info(f"Initial query returned {len(results)} results")
        
        # Initialize re-ranker
        reranker = Reranker()
        
        # Re-rank results
        reranked_results = reranker.rerank(test_query, results, top_k=3)
        
        logger.info(f"Re-ranker returned {len(reranked_results)} results")
        
        # Display re-ranking scores
        for i, result in enumerate(reranked_results):
            logger.info(f"Result {i+1} score: {result.get('rerank_score', 'N/A')}")
            logger.info(f"Result {i+1} text: {result['text'][:50]}...")
        
        return reranked_results
    except Exception as e:
        logger.error(f"Error in re-ranker test: {e}")
        return None

if __name__ == "__main__":
    # Test with a sample file
    test_file = "D:/Webosmotic-task/Part_1.txt"  # Replace with an actual file path
    
    if os.path.exists(test_file):
        results = test_reranker(test_file)
        if results:
            logger.info("Re-ranker test successful")
        else:
            logger.error("Re-ranker test failed")
    else:
        logger.warning(f"Test file not found: {test_file}")