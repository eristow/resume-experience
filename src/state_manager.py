import os
from dataclasses import dataclass
from typing import Optional, List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
import config
import logging
from logger import setup_logging

logger = setup_logging()


def save_analysis_results(
    st, result: str, job_retriever: Chroma, resume_retriever: Chroma
):
    st.session_state.result = result.content
    st.session_state.job_retriever = job_retriever
    st.session_state.resume_retriever = resume_retriever
    logger.info(f"result: {result.content}")


def reset_state_analysis(st) -> None:
    """Reset all session state variables related to analysis"""
    st.session_state.chat_history = []
    st.session_state.result = ""
    st.session_state.job_retriever = None
    st.session_state.resume_retriever = None
    st.session_state.analysis_confirmed = False


def new_ollama_instance() -> ChatOllama:
    """Create a new instance of the ChatOllama model"""
    return ChatOllama(
        model="mistral:v0.3",
        temperature=0.3,
        base_url=config.app_config.OLLAMA_BASE_URL,
        num_ctx=config.app_config.CONTEXT_WINDOW,
    )
