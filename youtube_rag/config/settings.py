import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_secret(key: str, default: str = ""):
    """Get secret from Streamlit Cloud (st.secrets) or local .env file."""
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except:
        return os.getenv(key, default)


class Settings:
    """Central config — reads from Streamlit Cloud secrets or local .env."""

    OPENROUTER_API_KEY: str = get_secret("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    PINECONE_API_KEY: str = get_secret("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = get_secret("PINECONE_INDEX_NAME", "youtube-rag")

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    TOP_K_RESULTS: int = 5
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50


settings = Settings()