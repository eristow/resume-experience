import pytest
from unittest.mock import Mock, patch
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage
from components.chatbot import handle_chat
from streamlit.testing.v1 import AppTest


class TestHandleChat:
    @pytest.fixture
    def mock_chat_response(self):
        with patch("components.chatbot.get_chat_response") as mock:
            mock.return_value = AIMessage(content="Test response")
            yield mock

    @pytest.fixture
    def mock_streamlit(self):
        with patch("components.chatbot.st") as mock:

            class SessionState:
                chat_history = []

            mock = Mock()
            mock.session_state = SessionState()
            yield mock

    @pytest.fixture
    def mock_dependencies(self):
        job_retriever = Mock()
        resume_retriever = Mock()
        ollama = Mock(spec=ChatOllama)
        return job_retriever, resume_retriever, ollama

    def test_handle_chat_basic_interaction(
        self, mock_chat_response, mock_streamlit, mock_dependencies
    ):
        job_retriever, resume_retriever, ollama = mock_dependencies
        user_input = "Test question"

        with patch("streamlit.write") as mock_write:
            handle_chat(
                mock_streamlit, user_input, job_retriever, resume_retriever, ollama
            )

            print(f"chat_history: {mock_streamlit.session_state.chat_history}")

            assert len(mock_streamlit.session_state.chat_history) == 2
            assert mock_streamlit.session_state.chat_history[0] == {
                "role": "User",
                "content": "Test question",
            }
            assert mock_streamlit.session_state.chat_history[1] == {
                "role": "Assistant",
                "content": "Test response",
            }

    def test_handle_chat_multiple_interactions(
        self, mock_chat_response, mock_streamlit, mock_dependencies
    ):
        job_retriever, resume_retriever, ollama = mock_dependencies
        inputs = ["First question", "Second question"]

        with patch("streamlit.write") as mock_write:
            for user_input in inputs:
                handle_chat(
                    mock_streamlit, user_input, job_retriever, resume_retriever, ollama
                )

            assert len(mock_streamlit.session_state.chat_history) == 4
            assert mock_chat_response.call_count == 2

    def test_handle_chat_calls_dependencies(
        self, mock_chat_response, mock_streamlit, mock_dependencies
    ):
        job_retriever, resume_retriever, ollama = mock_dependencies
        user_input = "Test question"

        handle_chat(mock_streamlit, user_input, job_retriever, resume_retriever, ollama)

        mock_chat_response.assert_called_once_with(
            user_input, job_retriever, resume_retriever, ollama
        )
