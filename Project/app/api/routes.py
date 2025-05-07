from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import logging
import os
import tempfile
import uuid
from typing import List, Optional, Dict, Any
import shutil

from .models import EmbeddingRequest, EmbeddingResponse, QueryRequest, QueryResponse, ErrorResponse
from ..core.config import settings
from ..document_processing.processor import BaseDocumentProcessor
from ..embeddings.embedding_provider import EmbeddingService
from ..retrieval.vector_store import VectorStore
from ..retrieval.reranker import Reranker
from ..llm.llm_provider import OllamaProvider
from app.api.conversation_manage import ConversationManager

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
vector_store = VectorStore()
embedding_service = EmbeddingService()
reranker = Reranker()
llm_service = OllamaProvider()
conversation_manager = ConversationManager()

# Helper function to get uploaded file path
async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save an uploaded file to disk.
    
    Args:
        upload_file: Uploaded file
        
    Returns:
        str: Path to the saved file
    """
    try:
        # Create a unique filename
        file_ext = os.path.splitext(upload_file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(settings.DOCUMENT_UPLOAD_FOLDER, unique_filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

@router.post("/api/embedding", response_model=EmbeddingResponse)
async def embed_document(document: UploadFile = File(...)):
    """
    Embed a document.
    
    Args:
        document: Uploaded document file
        
    Returns:
        EmbeddingResponse: Response with document ID
    """
    try:
        # Save the uploaded file
        file_path = await save_upload_file(document)
        logger.info(f"Saved uploaded file: {file_path}")
        
        # Get the appropriate document processor
        try:
            processor = BaseDocumentProcessor.get_processor_for_file(file_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Process the document
        processed_doc = processor.process(file_path)
        logger.info(f"Processed document: {processed_doc.metadata.get('source', 'unknown')}")
        
        # Chunk the document
        doc_chunks = processor.chunk(processed_doc)
        logger.info(f"Generated {len(doc_chunks)} chunks")
        
        # Generate embeddings
        document_data = embedding_service.embed_document_chunks(doc_chunks)
        logger.info(f"Generated embeddings for {len(document_data['chunks'])} chunks")
        
        # Store in vector database
        document_id = vector_store.add_document(document_data)
        logger.info(f"Stored document in vector database: {document_id}")
        
        # Return success response
        return {
            "status": "success",
            "message": "Document embedded successfully.",
            "document_id": document_id
        }
        
    except Exception as e:
        logger.error(f"Error embedding document: {e}")
        error_detail = str(e) if not isinstance(e, HTTPException) else e.detail
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Failed to embed document.",
                "error_details": error_detail
            }
        )

@router.post("/api/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query a document.
    
    Args:
        request: Query request
        
    Returns:
        QueryResponse: Response with answer and citations
    """
    try:
        # Extract request parameters
        query = request.query
        document_id = request.document_id
        require_citations = request.require_citations
        conversation_id = request.conversation_id
        
        logger.info(f"Received query request: {query} for document: {document_id}")
        
        # Generate query embedding
        query_embedding = embedding_service.provider.get_embeddings([query])[0]
        
        # Retrieve relevant chunks from vector store
        retrieved_chunks = vector_store.query(
            query_text=query,
            embedding=query_embedding,
            document_id=document_id,
            n_results=10
        )
        
        if not retrieved_chunks:
            logger.warning(f"No relevant chunks found for query: {query}")
            return {
                "status": "success",
                "response": {
                    "answer": "I couldn't find any relevant information in the document to answer your question.",
                    "citations": []
                },
                "conversation_id": conversation_id or conversation_manager.create_conversation(document_id)
            }
        
        # Re-rank the retrieved chunks
        reranked_chunks = reranker.rerank(query, retrieved_chunks, top_k=5)
        logger.info(f"Re-ranked {len(reranked_chunks)} chunks")
        
        # Get conversation history if conversation_id is provided
        conversation_history = None
        if conversation_id:
            history = conversation_manager.get_conversation_history(conversation_id)
            if history:
                conversation_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in history
                ]
        
        # Generate response with LLM
        response = llm_service.generate_response(
            query=query,
            context=reranked_chunks,
            conversation_history=conversation_history,
            require_citations=require_citations
        )
        
        # Create or update conversation
        if not conversation_id:
            conversation_id = conversation_manager.create_conversation(document_id)
        
        # Add messages to conversation
        conversation_manager.add_message(conversation_id, "user", query)
        conversation_manager.add_message(conversation_id, "assistant", response["answer"], response)
        
        # Return response
        return {
            "status": "success",
            "response": response,
            "conversation_id": conversation_id
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_detail = str(e) if not isinstance(e, HTTPException) else e.detail
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Failed to process query.",
                "error_details": error_detail
            }
        )