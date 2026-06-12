from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone
from src.youtube_rag.config.settings import settings
from src.youtube_rag.components.embeddings import EmbeddingModel


class VectorStore:
    """Handles Pinecone vector store operations — indexing and retrieval."""

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.index_name = settings.PINECONE_INDEX_NAME
        self._vector_store = None

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    def index_documents(self, documents: list[Document], namespace: str) -> PineconeVectorStore:
        """
        Embeds and indexes a list of documents into Pinecone.
        Uses video_id as namespace to separate different videos.

        Args:
            documents: List of chunked transcript Documents
            namespace: Pinecone namespace (video_id)

        Returns:
            PineconeVectorStore instance
        """
        embeddings = self.embedding_model.get_embeddings()

        self._vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=self.index_name,
            namespace=namespace,
        )
        return self._vector_store

    def load_existing(self, namespace: str) -> PineconeVectorStore:
        """
        Loads an existing Pinecone index without re-indexing.
        Used when the same video is queried again.

        Args:
            namespace: Pinecone namespace (video_id)

        Returns:
            PineconeVectorStore instance
        """
        embeddings = self.embedding_model.get_embeddings()

        self._vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=embeddings,
            namespace=namespace,
        )
        return self._vector_store

    def namespace_exists(self, namespace: str) -> bool:
        """
        Checks if a namespace (video) already exists in Pinecone.
        Avoids re-indexing the same video twice.
        """
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            return namespace in stats.get("namespaces", {})
        except Exception:
            return False

    def get_retriever(self, namespace: str, top_k: int = None):
        """
        Returns a LangChain retriever for semantic search.

        Args:
            namespace: Pinecone namespace (video_id)
            top_k: Number of chunks to retrieve (default from settings)
        """
        k = top_k or settings.TOP_K_RESULTS
        vector_store = self.load_existing(namespace)
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )