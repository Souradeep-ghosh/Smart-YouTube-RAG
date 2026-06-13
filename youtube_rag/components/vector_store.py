from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore as LCVectorStore
from youtube_rag.config.settings import settings
from youtube_rag.components.embeddings import EmbeddingModel


class VectorStore:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.index_name = settings.PINECONE_INDEX_NAME
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = pc.Index(self.index_name)

    def index_documents(self, documents: list[Document], namespace: str):
        embeddings = self.embedding_model.get_embeddings()
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        vectors = embeddings.embed_documents(texts)
        to_upsert = [
            (f"{namespace}-{i}", vectors[i], {**metadatas[i], "text": texts[i]})
            for i in range(len(texts))
        ]
        self.index.upsert(vectors=to_upsert, namespace=namespace)

    def namespace_exists(self, namespace: str) -> bool:
        try:
            stats = self.index.describe_index_stats()
            return namespace in stats.get("namespaces", {})
        except Exception:
            return False

    def get_retriever(self, namespace: str, top_k: int = 5):
        embedding_model = self.embedding_model

        class PineconeRetriever:
            def invoke(self, query: str) -> list[Document]:
                vector = embedding_model.embed_query(query)
                results = self.index.query(
                    vector=vector,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True
                )
                return [
                    Document(
                        page_content=match["metadata"].get("text", ""),
                        metadata=match["metadata"]
                    )
                    for match in results["matches"]
                ]

        retriever = PineconeRetriever()
        retriever.index = self.index
        return retriever