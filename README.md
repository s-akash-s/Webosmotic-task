# Document Intelligence RAG Chatbot System

A powerful and intelligent document processing and querying system that can handle various document formats and provide accurate, contextual responses with proper citations.

## Features

- **Multi-format Document Support**: Process PDF (with OCR for scanned content), DOCX, and TXT files
- **Intelligent Document Processing**: Extract text and metadata from various document formats
- **Advanced Chunking Strategies**: Contextual chunking with metadata including page numbers and document names
- **Semantic Search**: Find the most relevant information using state-of-the-art embedding models
- **Re-ranking Capabilities**: Improve search precision with dedicated re-ranking models
- **Conversational Memory**: Track and utilize conversation history for enhanced responses
- **Proper Citations**: Provide page-level citations for all responses to ensure transparency

## Technology Stack

- **FastAPI**: High-performance API framework for the backend
- **Streamlit**: User-friendly frontend interface
- **LangChain**: Framework for building LLM applications
- **Sentence Transformers**: For embeddings and re-ranking
- **ChromaDB**: Vector database for efficient retrieval
- **Ollama**: Integration with open-source large language models
- **PyTesseract**: OCR capability for scanned documents
- **PyPDF2 & python-docx**: Document format processing

## System Architecture

The system follows a modular architecture:

1. **Document Processing Pipeline**: Handles document ingestion, OCR, and text extraction
2. **Embedding Layer**: Transforms document chunks into vector representations
3. **Vector Database**: Stores and indexes embedded chunks for efficient retrieval
4. **Retrieval System**: Finds and re-ranks relevant document chunks based on user queries
5. **LLM Integration**: Generates accurate responses using context from retrieved chunks
6. **API Layer**: Exposes functionality through well-defined REST endpoints
7. **UI Layer**: Provides an intuitive interface for users to interact with the system

## Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://github.com/ollama/ollama) for local LLM support (or access to a remote Ollama server)
- For OCR functionality: Tesseract OCR must be installed

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd document-intelligence-rag-system
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Update configuration settings in `app/core/config.py` if needed

4. Run the application:
   ```
   python run_app.py
   ```

5. Access the application:
   - API: http://localhost:8000
   - UI: http://localhost:8501

## API Documentation

### Document Embedding API

Embeds a document for later querying.

- **Endpoint**: `POST /api/embedding`
- **Request**: Multi-part form data with uploaded document file
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Document embedded successfully.",
    "document_id": "12345"
  }
  ```

### Document Query API

Queries an embedded document and returns relevant answers with citations.

- **Endpoint**: `POST /api/query`
- **Request**:
  ```json
  {
    "query": "What is the main argument in the document?",
    "document_id": "12345",
    "require_citations": true,
    "conversation_id": null
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "response": {
      "answer": "The main argument is about the impact of technology on education.",
      "citations": [
        {
          "page": 12,
          "document_name": "sample_document.pdf"
        }
      ]
    },
    "conversation_id": "abc123xyz"
  }
  ```

## Testing

Run the test suite with:
```
python -m pytest tests/
```

## License

[MIT License](LICENSE)
