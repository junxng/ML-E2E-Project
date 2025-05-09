import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "EC2 Ubuntu"
    
    # LangChain Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # LangSmith for tracing (optional)
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGCHAIN_TRACING: str = os.getenv("LANGSMITH_TRACING", "false")
    
    # Qdrant Settings
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION", "pdf_collection")
    
    # PDF Storage
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    PDF_STORAGE_PATH: Path = BASE_DIR / "data" / "pdfs"
    KNOWLEDGE_BASE_FILE: Path = BASE_DIR / "data" / "knowledge_base.json"
    
    # Application Settings
    MAX_PDF_SIZE_MB: int = int(os.getenv("MAX_PDF_SIZE_MB", "10"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    def __init__(self):
        # Create directories if they don't exist
        os.makedirs(self.PDF_STORAGE_PATH, exist_ok=True)
    
settings = Settings()
