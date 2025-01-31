import pytest
from state_manager import (
    save_analysis_results,
    reset_state_analysis,
    new_ollama_instance,
)
from unittest.mock import MagicMock, patch
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama


class TestStateManager:
    def test_save_analysis_results(self):
        mock_st = MagicMock()
        mock_st.session_state.return_value = {}

        mock_result = MagicMock()
        mock_result.content = "test result"
        mock_job_retriever = MagicMock(spec=Chroma)
        mock_resume_retriever = MagicMock(spec=Chroma)

        save_analysis_results(
            mock_st, mock_result, mock_job_retriever, mock_resume_retriever
        )

        assert mock_st.session_state.result == "test result"
        assert mock_st.session_state.job_retriever == mock_job_retriever
        assert mock_st.session_state.resume_retriever == mock_resume_retriever

    def test_reset_state_analysis(self):
        mock_st = MagicMock()
        mock_st.session_state.chat_history = ["test message"]
        mock_st.session_state.result = "test result"
        mock_st.session_state.job_retriever = MagicMock()
        mock_st.session_state.resume_retriever = MagicMock()

        reset_state_analysis(mock_st)

        assert mock_st.session_state.chat_history == []
        assert mock_st.session_state.result == ""
        assert mock_st.session_state.job_retriever is None
        assert mock_st.session_state.resume_retriever is None

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
