import streamlit as st
from analyze import get_chat_response
from langchain_community.chat_models import ChatOllama
from langchain_core.vectorstores import VectorStoreRetriever
from logger import setup_logging

logger = setup_logging()


def render_chatbot():
    st.subheader("Chatbot Feature")
    user_input = st.text_input("Ask a question:")

    submit = st.button("Submit Query")

    return user_input if submit else None


def handle_chat(
    st,
    user_input: str,
    job_retriever: VectorStoreRetriever,
    resume_retriever: VectorStoreRetriever,
    ollama: ChatOllama,
) -> None:
    """Handle chat interactions and update the chat history."""
    response = get_chat_response(
        user_input, job_retriever, resume_retriever, ollama
    ).content

    st.session_state.chat_history.append({"role": "User", "content": user_input})
    st.session_state.chat_history.append({"role": "Assistant", "content": response})

    print(f"chatbot chat_history after: {st.session_state.chat_history[0]}")

    for chat in st.session_state.chat_history:
        st.write(f"{chat['role']}:\n{chat['content']}")
