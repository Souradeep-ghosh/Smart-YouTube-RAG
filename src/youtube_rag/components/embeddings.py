from langchain_huggingface import HuggingFaceEmbeddings
from src.youtube_rag.config.settings import settings


class EmbeddingModel:
    """Handles HuggingFace embedding model initialization and encoding."""

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self._embeddings = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Returns the HuggingFace embeddings model.
        Uses lazy loading — model is only downloaded on first call.
        """
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "normalize_embeddings": True,  # cosine similarity ready
                    "batch_size": 32,
                },
            )
        return self._embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embeds a single query string.
        Used during retrieval to embed the user's question.
        """
        return self.get_embeddings().embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of document strings.
        Used during indexing to embed transcript chunks.
        """
        return self.get_embeddings().embed_documents(texts)