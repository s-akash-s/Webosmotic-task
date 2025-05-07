# test_llm.py
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.llm.llm_provider import OllamaProvider

def test_llm_service():
    """Test the LLM service with a simple query and context"""
    logger.info("Testing LLM service")
    
    try:
        # Initialize LLM service
        llm_service = OllamaProvider()
        
        # Sample query and context
        query = "What is the main topic of the text?"
        
        context = [
            {
                "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.",
                "metadata": {
                    "source": "sample.txt",
                    "page": 1
                }
            },
            {
                "text": "Machine learning is a subset of AI that focuses on developing systems that learn from data without being explicitly programmed.",
                "metadata": {
                    "source": "sample.txt",
                    "page": 2
                }
            }
        ]
        
        # Generate response
        response = llm_service.generate_response(
            query=query,
            context=context,
            conversation_history=None,
            require_citations=True
        )
        
        logger.info(f"LLM response: {response['answer']}")
        logger.info(f"Citations: {response['citations']}")
        
        return response
    except Exception as e:
        logger.error(f"Error in LLM test: {e}")
        return None

if __name__ == "__main__":
    response = test_llm_service()
    if response:
        logger.info("LLM service test successful")
    else:
        logger.error("LLM service test failed")