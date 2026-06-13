from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


class RAGChain:
    """
    Handles Q&A and Summarization chains using retrieved transcript chunks.
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.output_parser = StrOutputParser()

    # ─────────────────────────────────────────
    # Q&A Chain
    # ─────────────────────────────────────────

    def get_qa_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert assistant that answers questions 
based strictly on the provided YouTube video transcript context.

Rules:
- Answer only from the context provided
- If the answer is not in the context, say "I couldn't find that in the video."
- Be concise and clear
- Quote relevant parts when helpful

Context from video transcript:
{context}
"""),
            ("human", "{question}")
        ])

    def format_docs(self, docs: list[Document]) -> str:
        """Formats retrieved chunks into a single context string."""
        return "\n\n".join([
            f"[Chunk {i+1}]: {doc.page_content}"
            for i, doc in enumerate(docs)
        ])

    def build_qa_chain(self, retriever):
        """
        Builds a RAG Q&A chain:
        question → retrieve chunks → format → LLM → answer
        """
        prompt = self.get_qa_prompt()

        chain = (
            {
                "context": retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | self.output_parser
        )
        return chain

    # ─────────────────────────────────────────
    # Summarization Chain
    # ─────────────────────────────────────────

    def get_summarization_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert at summarizing YouTube video transcripts.

Create a well-structured summary in markdown format with:
- **🎯 Main Topic**: One sentence about what the video is about
- **📌 Key Points**: 5-7 bullet points of the most important ideas
- **💡 Key Takeaways**: 2-3 actionable insights or conclusions
- **🏷️ Topics Covered**: comma-separated list of topics

Be concise, informative and well-organized.
"""),
            ("human", "Please summarize this YouTube video transcript:\n\n{transcript}")
        ])

    def build_summarization_chain(self):
        """
        Builds a direct summarization chain:
        full transcript → LLM → structured summary
        """
        prompt = self.get_summarization_prompt()
        chain = prompt | self.llm | self.output_parser
        return chain

    def summarize(self, transcript_text: str) -> str:
        """
        Summarizes the full transcript text.

        Args:
            transcript_text: Full plain text of the video transcript

        Returns:
            Markdown formatted summary string
        """
        chain = self.build_summarization_chain()
        return chain.invoke({"transcript": transcript_text})

    def answer_question(self, question: str, retriever) -> str:
        """
        Answers a question using RAG over the transcript chunks.

        Args:
            question: User's question string
            retriever: Pinecone retriever for the current video

        Returns:
            Answer string from LLM
        """
        chain = self.build_qa_chain(retriever)
        return chain.invoke(question)