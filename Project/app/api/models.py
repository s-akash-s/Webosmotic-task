from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class EmbeddingRequest(BaseModel):
    document: str = Field(..., description="Filename of the document to embed")

class EmbeddingResponse(BaseModel):
    status: str
    message: str
    document_id: str

class Citation(BaseModel):
    page: int
    document_name: str

class Answer(BaseModel):
    answer: str
    citations: List[Citation]

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    document_id: str = Field(..., description="ID of the document to query")
    require_citations: bool = Field(True, description="Whether to include citations")
    conversation_id: Optional[str] = Field(None, description="ID for conversation tracking")

class QueryResponse(BaseModel):
    status: str
    response: Optional[Answer] = None
    message: Optional[str] = None
    conversation_id: Optional[str] = None
    error_details: Optional[str] = None

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    error_details: Optional[str] = None