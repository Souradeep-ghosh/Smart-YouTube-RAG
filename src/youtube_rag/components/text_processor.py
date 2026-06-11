from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextProcessor:
    """Handles text chunking and preprocessing of transcript documents."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )

    def merge_transcript_segments(self, documents: list[Document]) -> list[Document]:
        """
        Merges all transcript segments into a single document
        before chunking — preserves context across segments.
        """
        if not documents:
            raise ValueError("No documents to process.")

        # Merge all text with timestamps preserved in metadata
        full_text = " ".join([doc.page_content for doc in documents])
        metadata = documents[0].metadata  # carry over video metadata

        return [Document(page_content=full_text, metadata=metadata)]

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Splits merged transcript into chunks for embedding.
        Preserves video metadata in each chunk.
        """
        merged = self.merge_transcript_segments(documents)
        chunks = self.splitter.split_documents(merged)

        # Add chunk index to metadata for traceability
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)