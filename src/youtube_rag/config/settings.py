import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Central configuration for all API keys and app settings."""
    
    # OpenRouter
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "youtube-rag")
    
    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Text Splitting
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Retrieval
    TOP_K_RESULTS: int = 5
    
    def validate(self):
        """Validates that all required API keys are present."""
        missing = []
        if not self.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        if not self.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

settings = Settings()