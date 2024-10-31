import os
from dataclasses import dataclass
from typing import Optional, List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
import config
import logging

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    result: str = ""
    job_retriever: Optional[Chroma] = None
    resume_retriever: Optional[Chroma] = None
    job_text: str = ""
    resume_text: str = ""
    chat_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.chat_history is None:
            self.chat_history = []

    def add_chat_message(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})

    def save_analysis_results(
        self, result: str, job_retriever: Chroma, resume_retriever: Chroma
    ):
        self.result = result.content
        self.job_retriever = job_retriever
        self.resume_retriever = resume_retriever
        logger.info(f"result: {result.content}")


def initialize_state(st) -> None:
    """Initialize all session state variables"""
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()


def reset_state_analysis(st) -> None:
    """Reset all session state variables related to analysis"""
    # TODO: is resetting chat history on new analysis desired behavior? Should we inform the user before hand?
    st.session_state.app_state.chat_history = []
    st.session_state.app_state.result = ""
    st.session_state.app_state.job_retriever = None
    st.session_state.app_state.resume_retriever = None


def new_ollama_instance() -> ChatOllama:
    """Create a new instance of the ChatOllama model"""
    return ChatOllama(
        model="mistral:v0.3",
        temperature=0.3,
        base_url=config.app_config.OLLAMA_BASE_URL,
        num_ctx=config.app_config.CONTEXT_WINDOW,
    )
