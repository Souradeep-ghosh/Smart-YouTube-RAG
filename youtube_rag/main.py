import streamlit as st
from youtube_rag.ui.streamlit_ui import StreamlitUI
from youtube_rag.ui.display_result import DisplayResult
from youtube_rag.components.transcript_loader import TranscriptLoader
from youtube_rag.components.text_processor import TextProcessor
from youtube_rag.components.vector_store import VectorStore
from youtube_rag.components.llm import LLMModel
from youtube_rag.components.rag_chain import RAGChain


def load_app():
    """Main orchestrator — loads and runs the Smart YouTube RAG Assistant."""

    # ── UI Setup ──────────────────────────────────────────
    ui = StreamlitUI()
    display = DisplayResult()
    ui.setup_page()
    user_controls = ui.render_sidebar()

    # ── Validate API Keys ─────────────────────────────────
    openrouter_key = user_controls.get("openrouter_api_key")
    pinecone_key = user_controls.get("pinecone_api_key")

    if not openrouter_key or not pinecone_key:
        st.info("👈 Please enter your API keys in the sidebar to get started.")
        return

    # ── Override settings with user-provided keys ─────────
    import os
    os.environ["OPENROUTER_API_KEY"] = openrouter_key
    os.environ["PINECONE_API_KEY"] = pinecone_key

    # ── YouTube URL Input ─────────────────────────────────
    url = ui.render_video_input()

    if not url:
        st.info("👆 Paste a YouTube URL above to get started.")
        return

    # ── Process Video Button ──────────────────────────────
    process_btn = st.button("🚀 Process Video", type="primary", use_container_width=True)

    if process_btn:
        try:
            # Step 1: Load transcript
            with st.spinner("📥 Fetching transcript..."):
                loader = TranscriptLoader()
                documents, video_id, title = loader.load_transcript(url)
                st.session_state["video_id"] = video_id
                st.session_state["title"] = title
                st.session_state["documents"] = documents
                st.session_state["video_loaded"] = False

            # Step 2: Process and chunk transcript
            with st.spinner("✂️ Processing and chunking transcript..."):
                processor = TextProcessor(
                    chunk_size=user_controls.get("chunk_size", 500),
                    chunk_overlap=50
                )
                chunks = processor.split_documents(documents)
                full_text = processor.get_full_transcript_text(documents)
                st.session_state["full_text"] = full_text
                st.session_state["chunks"] = chunks

            # Step 3: Index into Pinecone
            with st.spinner("📌 Indexing into Pinecone vector store..."):
                from youtube_rag.config.settings import settings
                settings.PINECONE_API_KEY = pinecone_key
                vector_store = VectorStore()

                if vector_store.namespace_exists(video_id):
                    st.info("⚡ Video already indexed — loading from Pinecone cache.")
                else:
                    vector_store.index_documents(chunks, namespace=video_id)

                st.session_state["video_loaded"] = True

            # Step 4: Show video info
            ui.render_video_info(title, video_id, len(chunks))

        except Exception as e:
            import traceback
            display.display_error(f"Failed to process video: {str(e)}\n\n{traceback.format_exc()}")
            return

    # ── Main Tabs ─────────────────────────────────────────
    if st.session_state.get("video_loaded"):
        tab_summary, tab_qa = ui.render_mode_tabs()

        # Initialize components
        llm_model = LLMModel(
            model_name=user_controls.get("selected_model"),
            api_key=openrouter_key
        )
        llm = llm_model.get_llm()
        rag = RAGChain(llm=llm)

        # ── Tab 1: Summarization ──────────────────────────
        with tab_summary:
            st.subheader("📄 Video Summary")
            if st.button("✨ Generate Summary", type="primary", use_container_width=True):
                with st.spinner("🤖 Generating summary..."):
                    try:
                        summary = rag.summarize(st.session_state["full_text"])
                        st.session_state["summary"] = summary
                    except Exception as e:
                        display.display_error(f"Summarization failed: {str(e)}")

            if st.session_state.get("summary"):
                display.display_summary(
                    st.session_state["summary"],
                    st.session_state.get("title", "Video")
                )

        # ── Tab 2: Q&A ────────────────────────────────────
        with tab_qa:
            st.subheader("💬 Ask Questions About the Video")
            display.display_clear_chat_button()

            # Build retriever for current video
            vector_store = VectorStore()
            retriever = vector_store.get_retriever(
                namespace=st.session_state["video_id"],
                top_k=user_controls.get("top_k", 5)
            )

            # Q&A callback
            def answer_question(question: str) -> str:
                return rag.answer_question(question, retriever)

            display.display_qa_interface(
                video_loaded=True,
                qa_callback=answer_question
            )