from pydantic_settings import BaseSettings
from typing import Optional, List, Dict
from pathlib import Path
import os

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Document Intelligence RAG System"
    
    # LLM Settings
    LLM_MODEL: str = "qwen2.5:7b"  
    LLM_PROVIDER: str = "ollama"     
    LLM_API_KEY: Optional[str] = None
    LLM_TEMPERATURE: float = 0.1
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en"
    EMBEDDING_PROVIDER: str = "local"  # 'local', 'openai', etc.
    EMBEDDING_DIMENSION: int = 384
    
    # Re-ranker Settings
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_PROVIDER: str = "local"
    
    # Vector DB Settings
    VECTOR_DB: str = "chroma"
    VECTOR_DB_PATH: str = "./data/vector_db"
    
    # Document Processing
    DOCUMENT_UPLOAD_FOLDER: str = "./data/documents"
    DOCUMENT_PROCESSED_FOLDER: str = "./data/processed"
    OCR_LANGUAGE: str = "eng"
    
    # Chunking Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    
    # Base path for the project
    BASE_PATH: Path = Path(__file__).resolve().parent.parent.parent
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

os.makedirs(settings.DOCUMENT_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(settings.DOCUMENT_PROCESSED_FOLDER, exist_ok=True)
os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)