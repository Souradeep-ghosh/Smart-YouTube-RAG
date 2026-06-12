from langchain_openai import ChatOpenAI
from src.youtube_rag.config.settings import settings


class LLMModel:
    """Handles OpenRouter LLM initialization via OpenAI-compatible API."""

    def __init__(self, model_name: str = None, api_key: str = None):
        self.model_name = model_name or "openrouter/free"
        self.api_key = api_key or settings.OPENROUTER_API_KEY

    def get_llm(self) -> ChatOpenAI:
        """
        Returns a LangChain ChatOpenAI instance pointed at OpenRouter.
        OpenRouter is fully compatible with the OpenAI API format.
        """
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is missing. "
                "Please provide it in the sidebar or set OPENROUTER_API_KEY in .env"
            )

        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=settings.OPENROUTER_BASE_URL,
            temperature=0.3,
            max_tokens=2048,
            default_headers={
                "HTTP-Referer": "https://github.com/Souradeep-ghosh/Smart-YouTube-RAG",
                "X-Title": "Smart YouTube RAG Assistant",
            }
        )