import pytest
from state_manager import (
    AppState,
    initialize_state,
    reset_state_analysis,
    new_ollama_instance,
)
from unittest.mock import MagicMock, patch
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama


class TestStateManager:
    def test_app_state_init(self):
        state = AppState()
        assert state.result == ""
        assert state.job_retriever is None
        assert state.resume_retriever is None
        assert state.job_text == ""
        assert state.resume_text == ""
        assert state.chat_history == []
        assert state.initialized is False

    def test_app_state_add_chat_message(self):
        state = AppState()
        state.add_chat_message("user", "test message")
        assert len(state.chat_history) == 1
        assert state.chat_history[0]["role"] == "user"
        assert state.chat_history[0]["content"] == "test message"

    def test_app_state_save_analysis_results(self):
        state = AppState()
        mock_result = MagicMock()
        mock_result.content = "test result"
        mock_job_retriever = MagicMock(spec=Chroma)
        mock_resume_retriever = MagicMock(spec=Chroma)

        state.save_analysis_results(
            mock_result, mock_job_retriever, mock_resume_retriever
        )

        assert state.result == "test result"
        assert state.job_retriever == mock_job_retriever
        assert state.resume_retriever == mock_resume_retriever

    def test_initialize_state(self):
        mock_st = MagicMock()
        mock_st.session_state.return_value = {}

        initialize_state(mock_st)

        assert isinstance(mock_st.session_state.app_state, AppState)

    def test_reset_state_analysis(self):
        mock_st = MagicMock()
        mock_st.session_state.app_state = AppState()
        mock_st.session_state.app_state.chat_history = ["test message"]
        mock_st.session_state.app_state.result = "test result"
        mock_st.session_state.app_state.job_retriever = MagicMock()
        mock_st.session_state.app_state.resume_retriever = MagicMock()

        reset_state_analysis(mock_st)

        assert mock_st.session_state.app_state.chat_history == []
        assert mock_st.session_state.app_state.result == ""
        assert mock_st.session_state.app_state.job_retriever is None
        assert mock_st.session_state.app_state.resume_retriever is None

    @patch("state_manager.ChatOllama")
    def test_new_ollama_instance(self, mock_chat_ollama):
        mock_instance = MagicMock(spec=ChatOllama)
        mock_chat_ollama.return_value = mock_instance

        result = new_ollama_instance()

        mock_chat_ollama.assert_called_once_with(
            model="mistral:v0.3",
            temperature=0.3,
            base_url="http://ollama:11434",
            num_ctx=8192,
        )
        assert result == mock_instance

    def test_app_state_multiple_chat_messages(self):
        state = AppState()
        messages = [("user", "Hello"), ("assistant", "Hi"), ("user", "How are you?")]

        for role, content in messages:
            state.add_chat_message(role, content)

        assert len(state.chat_history) == 3
        for i, (role, content) in enumerate(messages):
            assert state.chat_history[i]["role"] == role
            assert state.chat_history[i]["content"] == content
