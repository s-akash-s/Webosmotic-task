# test_api.py
import sys
import os
import logging
from fastapi.testclient import TestClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Import our application
from app import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    logger.info("Root endpoint test passed")

def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    logger.info("Health endpoint test passed")

def test_embedding_endpoint():
    """Test the embedding endpoint with a sample file"""
    test_file = "Part_1.txt"
    
    if os.path.exists(test_file):
        with open(test_file, "rb") as f:
            response = client.post(
                "/api/embedding",
                files={"document": (os.path.basename(test_file), f, "text/plain")}
            )
        
        logger.info(f"Embedding endpoint response: {response.json()}")
        assert response.status_code in (200, 201)
        assert response.json()["status"] == "success"
        assert "document_id" in response.json()
        
        # Save document_id for query test
        document_id = response.json()["document_id"]
        logger.info(f"Embedding endpoint test passed, document_id: {document_id}")
        return document_id
    else:
        logger.warning(f"Test file not found: {test_file}")
        return None

def test_query_endpoint(document_id):
    """Test the query endpoint with a sample query"""
    if not document_id:
        logger.warning("No document_id provided, skipping query test")
        return
    
    query_data = {
        "query": "What is the main topic?",
        "document_id": document_id,
        "require_citations": True
    }
    
    response = client.post("/api/query", json=query_data)
    
    logger.info(f"Query endpoint response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "response" in response.json()
    assert "conversation_id" in response.json()
    
    # Test follow-up query with conversation_id
    conversation_id = response.json()["conversation_id"]
    
    follow_up_data = {
        "query": "Tell me more about it",
        "document_id": document_id,
        "require_citations": True,
        "conversation_id": conversation_id
    }
    
    follow_up_response = client.post("/api/query", json=follow_up_data)
    
    logger.info(f"Follow-up query response: {follow_up_response.json()}")
    assert follow_up_response.status_code == 200
    assert follow_up_response.json()["status"] == "success"
    
    logger.info("Query endpoint test passed")

if __name__ == "__main__":
    test_root_endpoint()
    test_health_endpoint()
    document_id = test_embedding_endpoint()
    test_query_endpoint(document_id)