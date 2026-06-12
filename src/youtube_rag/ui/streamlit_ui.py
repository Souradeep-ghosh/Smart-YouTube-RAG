import streamlit as st
from configparser import ConfigParser
import os


class StreamlitUI:
    """Handles all Streamlit UI rendering and sidebar controls."""

    def __init__(self):
        self.config = ConfigParser()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "..", "config", "uiconfigfile.ini")
        self.config.read(config_path)

    def get_page_title(self) -> str:
        return self.config.get("DEFAULT", "PAGE_TITLE")

    def get_app_description(self) -> str:
        return self.config.get("DEFAULT", "APP_DESCRIPTION")

    def get_model_options(self) -> list:
        return self.config.get("DEFAULT", "MODEL_OPTIONS").split(", ")

    def setup_page(self):
        """Sets up the Streamlit page config and header."""
        st.set_page_config(
            page_title="🎥 " + self.get_page_title(),
            page_icon="🎥",
            layout="wide"
        )
        st.title("🎥 " + self.get_page_title())
        st.caption(self.get_app_description())

    def render_sidebar(self) -> dict:
        """
        Renders the sidebar controls and returns user inputs as a dict.
        """
        user_controls = {}

        with st.sidebar:
            st.header("⚙️ Configuration")
            st.divider()

            # API Keys
            st.subheader("🔑 API Keys")
            user_controls["openrouter_api_key"] = st.text_input(
                "OpenRouter API Key",
                type="password",
                help="Get your free key at openrouter.ai"
            )
            if not user_controls["openrouter_api_key"]:
                st.warning("🚧 Enter your OpenRouter API Key to proceed. "
                           "Get one free at: https://openrouter.ai/settings/keys")

            user_controls["pinecone_api_key"] = st.text_input(
                "Pinecone API Key",
                type="password",
                help="Get your free key at pinecone.io"
            )
            if not user_controls["pinecone_api_key"]:
                st.warning("🚧 Enter your Pinecone API Key to proceed. "
                           "Get one free at: https://app.pinecone.io")

            st.divider()

            # Model Selection
            st.subheader("🤖 Model")
            model_options = self.get_model_options()
            user_controls["selected_model"] = st.selectbox(
                "Select LLM",
                model_options,
                help="All models are free via OpenRouter"
            )

            st.divider()

            # Advanced Settings
            st.subheader("🔧 Advanced Settings")
            user_controls["chunk_size"] = st.slider(
                "Chunk Size",
                min_value=200,
                max_value=1500,
                value=500,
                step=100,
                help="Size of transcript chunks for retrieval"
            )
            user_controls["top_k"] = st.slider(
                "Top K Results",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of chunks to retrieve per question"
            )

            st.divider()

            # About section
            st.subheader("ℹ️ About")
            st.markdown("""
            Built with:
            - 🦜 LangChain + Pinecone
            - 🤗 HuggingFace Embeddings
            - 🚀 OpenRouter Free LLMs
            - 🎈 Streamlit
            """)

        return user_controls

    def render_video_input(self) -> str:
        """Renders the YouTube URL input field."""
        st.subheader("🔗 Enter YouTube URL")
        url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL with available transcripts"
        )
        return url

    def render_mode_tabs(self):
        """Renders the Summary and Q&A mode tabs."""
        return st.tabs(["📄 Summarize Video", "💬 Ask Questions"])

    def render_video_info(self, title: str, video_id: str, chunk_count: int):
        """Renders video metadata after successful processing."""
        st.success(f"✅ Video processed successfully!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📹 Video", title[:30] + "..." if len(title) > 30 else title)
        with col2:
            st.metric("🧩 Chunks Indexed", chunk_count)
        with col3:
            st.metric("🔍 Vector Store", "Pinecone ✅")