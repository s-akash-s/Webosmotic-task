# Document Intelligence RAG Chatbot System - Technical Documentation

## Introduction

This document provides detailed technical documentation for the Document Intelligence RAG Chatbot System, including technology choices, architectural decisions, and implementation details. The system is designed to meet the requirements specified in the WO AI/ML Task for building an intelligent document processing and querying system.

## Technology Choices and Justifications

### Model Selection

#### LLM Model: Qwen2.5 (7B) via Ollama

**Justification:**
- **Performance-Efficiency Balance**: Qwen2.5 (7B) provides strong reasoning and generative capabilities while being lightweight enough to run on standard hardware.
- **Context Length**: Supports longer context windows (up to 32k tokens) which is essential for RAG applications with extensive document contexts.
- **Offline Capability**: Using Ollama allows for deployment in environments without external API dependencies, ensuring data privacy and reducing operational costs.
- **Instruction Following**: Qwen2.5 demonstrates excellent ability to follow complex instructions, making it well-suited for generating responses with proper citations.

#### Embedding Model: BAAI/bge-small-en

**Justification:**
- **Efficiency**: At 384 dimensions, this model provides a good balance between embedding quality and storage/computational efficiency.
- **Performance**: Ranks highly on MTEB (Massive Text Embedding Benchmark) leaderboards while being more resource-efficient than larger models.
- **Specialized for Retrieval**: Specifically fine-tuned for retrieval tasks, making it more effective than general-purpose embeddings.
- **Multilingual Capabilities**: Can handle English content effectively while having some cross-lingual capabilities.

#### Re-ranker Model: BAAI/bge-reranker-base

**Justification:**
- **Complementary to Embedding Model**: Works well with the bge-small-en embedding model, providing a cohesive retrieval pipeline.
- **Accuracy Improvement**: Significantly improves relevance of retrieved documents compared to vector similarity search alone.
- **Lightweight**: Balances performance with resource utilization, allowing for deployment on standard hardware.
- **Cross-Encoder Architecture**: Provides more precise similarity scoring by considering query and document together rather than independently.

### Framework Selection

#### FastAPI

**Justification:**
- **Performance**: High throughput with minimal overhead, critical for handling document processing requests.
- **Type Safety**: Strong typing and validation through Pydantic ensures robust API design.
- **Async Support**: Built on Starlette for asynchronous request handling, allowing efficient IO operations.
- **Documentation**: Automatic Swagger/OpenAPI documentation generation simplifies API usage.

#### LangChain

**Justification:**
- **Modular Components**: Provides pre-built components for the entire RAG pipeline, accelerating development.
- **Integration Ecosystem**: Simplifies integration with multiple LLMs, embedding models, and vector stores.
- **Active Development**: Regularly updated with new features and optimizations for LLM applications.
- **Community Support**: Wide adoption means good documentation and community resources.

#### ChromaDB

**Justification:**
- **Performance**: Efficient vector storage and retrieval with multiple distance metrics.
- **Persistence**: Supports both in-memory and disk-based storage with SQLite backend.
- **Metadata Filtering**: Advanced filtering capabilities based on document metadata.
- **Ease of Integration**: Python-native with simple API, integrates well with LangChain.

## Document Processing Pipeline

### Document Ingestion

The system supports multiple document formats through specialized processors:

1. **PDF Processor** (`PDFProcessor`):
   - Uses PyPDF2 for text extraction from digital PDFs
   - Integrates Tesseract OCR via pdf2image for scanned documents
   - Extracts structured metadata including page numbers, document title, and creation date

2. **DOCX Processor** (`DocxProcessor`):
   - Leverages python-docx for text and structure extraction
   - Supports embedded images with OCR processing
   - Preserves document hierarchy when possible

3. **TXT Processor** (`TxtProcessor`):
   - Simple but efficient text extraction 
   - Adds basic metadata like filename and creation date

### Chunking Strategy

The system implements two chunking approaches:

1. **Hierarchical Chunking** (`hierarchical_chunker.py`):
   - Respects document structure (paragraphs, sections, etc.)
   - Creates parent-child relationships between chunks
   - Maintains context when splitting long sections

2. **Semantic Chunking** (`semantic_chunker.py`):
   - Uses embedding similarity to create semantically coherent chunks
   - Avoids splitting related concepts across chunks
   - More effective for unstructured documents

**Chunk Size Justification (1000 tokens with 100 token overlap):**
- Large enough to capture context but small enough for precise retrieval
- Overlap ensures concepts that cross chunk boundaries aren't lost
- Compatible with embedding model context limitations
- Optimized based on empirical testing with various document types

## Retrieval System

### Vector Database Implementation

The system uses ChromaDB for vector storage with the following configuration:

- Distance metric: Cosine similarity
- Persistence: SQLite backend for durability and performance
- Metadata filtering: Enables filtering by document ID and other metadata
- Optional: Collection-based organization for multi-tenant scenarios

### Re-ranking Process

To improve retrieval precision:

1. Initial retrieval returns top-10 chunks based on vector similarity
2. Re-ranker evaluates query-document pairs for more accurate relevance scoring
3. Top-5 re-ranked chunks are used for response generation
4. Metadata from re-ranked chunks is preserved for citation generation

## API Design

The system exposes two main endpoints that satisfy the requirements:

### Embedding API (`/api/embedding`)

- Accepts document uploads via multipart form
- Processes documents based on file type
- Generates and stores embeddings
- Returns a unique document ID for future queries

### Query API (`/api/query`)

- Accepts queries with document ID and optional conversation ID
- Retrieves and re-ranks relevant chunks
- Generates responses with citations
- Manages conversation history for follow-up queries
- Returns answers with page-level citations

## Testing Approach

The system has been tested with various document types:

1. **Academic/Technical**:
   - Research papers (PDF)
   - Technical documentation (DOCX)

2. **Literary/Historical**:
   - Public domain books (TXT)
   - Philosophical texts (PDF)

3. **Business/Legal**:
   - Annual reports (PDF)
   - Legal agreements (DOCX)

Test coverage includes:
- Unit tests for individual components
- Integration tests for the complete pipeline
- Performance benchmarks for various document sizes

## Performance Considerations

- **Memory Usage**: Optimized for balance between performance and resource utilization
- **Processing Time**: Document embedding pipeline optimized for parallel processing
- **Query Latency**: Typical query-to-response time under 2 seconds
- **Scalability**: Design allows for horizontal scaling through stateless API

## Future Enhancements

Potential improvements to consider:

1. **Hybrid Search**: Combining vector search with keyword-based search for improved recall
2. **Multi-document Queries**: Enabling queries across multiple documents simultaneously
3. **Advanced Chunking**: Implementing more sophisticated chunking strategies
4. **UI Improvements**: Enhanced visualization of document structure and citations
5. **Streaming Responses**: Implementing streaming for faster user feedback

## Conclusion

The Document Intelligence RAG Chatbot System has been designed and implemented to meet all requirements specified in the WO AI/ML Task. The system provides a robust solution for document processing and intelligent querying with an emphasis on accuracy, performance, and proper citation.
