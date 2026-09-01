import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """Application Configuration Settings."""
    
    # App Info
    APP_NAME: str = "DocuMind RAG Assistant"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API Keys
    GEMINI_API_KEY: str = ""
    
    # AI Models
    LLM_MODEL_NAME: str = "gemini-2.5-flash"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Directory Paths
    UPLOADS_DIR: Path = BASE_DIR / "data" / "uploads"
    INDEX_DIR: Path = BASE_DIR / "data" / "index"
    
    # Retrieval Configuration
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.25
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
