import streamlit as st
from analyze import get_chat_response
from langchain_community.chat_models import ChatOllama
from langchain_core.vectorstores import VectorStoreRetriever
from logger import setup_logging
from typing import Optional, Tuple

logger = setup_logging()


def cleanup_chat_resources(st) -> None:
    """Clean up chat-related resources."""
    if hasattr(st.session_state, "ollama"):
        try:
            # Clean up Ollama instance
            if st.session_state.ollama is not None:
                del st.session_state.ollama
                st.session_state.ollama = None
                logger.info("Cleaned up Ollama instance")

        except Exception as e:
            logger.error(f"Error cleaning up Ollama: {e}")


def render_chatbot():
    st.subheader("Chatbot Feature")
    user_input = st.text_input("Ask a question:")
    submit = st.button("Submit Query")

    # Clean up when switching away from chat
    if (
        not submit
        and hasattr(st.session_state, "last_chat_active")
        and st.session_state.last_chat_active
    ):
        cleanup_chat_resources(st)
        st.session_state.last_chat_active = False

    return user_input if submit else None


def handle_chat(
    st,
    user_input: str,
    job_retriever: VectorStoreRetriever,
    resume_retriever: VectorStoreRetriever,
    ollama: ChatOllama,
) -> None:
    """Handle chat interactions with proper resource management."""
    try:
        st.session_state.last_chat_active = True

        response = get_chat_response(
            user_input, job_retriever, resume_retriever, ollama
        ).content

        if not hasattr(st.session_state, "chat_history"):
            st.session_state.chat_history = []

        st.session_state.chat_history.append({"role": "User", "content": user_input})
        st.session_state.chat_history.append({"role": "Assistant", "content": response})

        for chat in st.session_state.chat_history:
            st.write(f"{chat['role']}:\n{chat['content']}")

    except Exception as e:
        logger.error(f"Error in chat handling: {e}")
        st.error("An error occurred while processing your request. Please try again.")

    finally:
        # Force cleanup of any embeddings created during chat
        if hasattr(job_retriever, "vectorstore") and hasattr(
            job_retriever.vectorstore, "_embeddings"
        ):
            job_retriever.vectorstore._embeddings.cleanup()
        if hasattr(resume_retriever, "vectorstore") and hasattr(
            resume_retriever.vectorstore, "_embeddings"
        ):
            resume_retriever.vectorstore._embeddings.cleanup()
