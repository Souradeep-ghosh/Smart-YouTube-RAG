import streamlit as st


class DisplayResult:
    """Handles rendering of RAG results in the Streamlit UI."""

    def display_summary(self, summary: str, title: str):
        """
        Renders the video summary in a clean formatted layout.

        Args:
            summary: Markdown formatted summary string from LLM
            title: Video title string
        """
        st.subheader(f"📄 Summary: {title}")
        st.divider()
        st.markdown(summary)
        st.divider()

        # Download button for summary
        st.download_button(
            label="⬇️ Download Summary",
            data=summary,
            file_name=f"{title[:30].replace(' ', '_')}_summary.md",
            mime="text/markdown"
        )

    def display_qa_interface(self, video_loaded: bool, qa_callback):
        """
        Renders the Q&A chat interface.

        Args:
            video_loaded: Whether a video has been processed
            qa_callback: Function to call with the user's question
        """
        if not video_loaded:
            st.info("👆 Please process a video first using the URL input above.")
            return

        # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display existing chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        question = st.chat_input("Ask anything about the video...")

        if question:
            # Display user message
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching transcript and generating answer..."):
                    try:
                        answer = qa_callback(question)
                        st.markdown(answer)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer
                        })
                    except Exception as e:
                        error_msg = f"❌ Error generating answer: {str(e)}"
                        st.error(error_msg)

    def display_processing_status(self, message: str):
        """Shows a processing spinner message."""
        return st.spinner(f"⏳ {message}")

    def display_error(self, message: str):
        """Renders a styled error message."""
        st.error(f"❌ {message}")

    def display_clear_chat_button(self):
        """Renders a button to clear chat history."""
        if st.session_state.get("chat_history"):
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
                
                