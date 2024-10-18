from streamlit.testing.v1 import AppTest
from tools import (
    extract_text,
    extract_text_from_file,
    analyze_inputs,
    extract_text_from_image,
)
import custom_embeddings
import os
import pytest
from unittest.mock import Mock, patch, call
from langchain_community.chat_models import ChatOllama


class TestAnalyzeInputs:
    """Testing analyze_inputs function."""

    @pytest.fixture
    def mock_chat_ollama(self):
        with patch("langchain_community.chat_models.ChatOllama") as mock_chat:
            mock_instance = Mock()
            mock_chat.return_value = mock_instance
            yield mock_chat

    @pytest.fixture
    def mock_embeddings(self):
        mock_emb = Mock(name="MockedEmbeddings")
        return mock_emb

    @pytest.fixture
    def mock_vectorstore(self):
        mock_vs = Mock()
        mock_retriever = Mock()
        mock_vs.as_retriever.return_value = mock_retriever
        return mock_vs

    @pytest.fixture
    def mock_process_text(self, mock_vectorstore):
        with patch("tools.process_text") as mock_process:
            mock_process.return_value = mock_vectorstore
            yield mock_process

    @pytest.fixture
    def mock_custom_embeddings(self, mock_embeddings):
        with patch("tools.CustomEmbeddings") as mock_custom:
            mock_custom.return_value = mock_embeddings
            yield mock_custom

    def test_analyze_inputs_happy(
        self,
        mock_chat_ollama,
        mock_process_text,
        mock_custom_embeddings,
        mock_vectorstore,
    ):
        job_text = "Sample job description"
        resume_text = "Sample resume"

        response, job_retriever, resume_retriever = analyze_inputs(
            job_text, resume_text
        )

        assert response.content[:8].strip() == "Overall"
        assert job_retriever == mock_vectorstore.as_retriever()
        assert resume_retriever == mock_vectorstore.as_retriever()

        assert mock_process_text.call_count == 2

        mock_custom_embeddings.assert_called_once_with(model_name="./models/mistral")

    def test_analyze_inputs_with_invalid_vectorstore(
        self, mock_chat_ollama, mock_custom_embeddings, mock_embeddings
    ):
        with patch("tools.process_text", return_value=None):
            response = analyze_inputs("job", "resume")
            assert response == "Failed to process the files."

    @pytest.mark.parametrize(
        "job_text,resume_text",
        [
            ("", "resume"),
            ("job", ""),
            ("", ""),
        ],
    )
    def test_analyze_inputs_with_missing_inputs(
        self, job_text, resume_text, mock_chat_ollama, mock_custom_embeddings
    ):
        job_text = None
        resume_text = None

        response = analyze_inputs(job_text, resume_text)
        assert response == "Failed to process the files."

    def test_analyze_inputs_full_chain(
        self,
        mock_chat_ollama,
        mock_process_text,
        mock_custom_embeddings,
        mock_vectorstore,
        mock_embeddings,
    ):
        job_text = "Sample job description"
        resume_text = "Sample resume"

        response, job_retriever, resume_retriever = analyze_inputs(
            job_text, resume_text
        )

        mock_custom_embeddings.assert_called_once_with(model_name="./models/mistral")
        assert mock_process_text.call_count == 2

        mock_process_text.assert_has_calls(
            [call(job_text, mock_embeddings), call(resume_text, mock_embeddings)],
            any_order=True,
        )

        assert mock_vectorstore.as_retriever.call_count == 2


class TestExtractText:
    """Testing extract_text function."""

    @pytest.fixture(scope="session", autouse=True)
    def cleanup_temp_dir(self):
        yield
        os.remove("tests/temp/test.pdf")

    def test_extract_job_ad(self):
        file = open("tests/test.pdf", "r")

        assert extract_text("job ad", file, "./tests/temp/") == "Test image text."


class TestExtractTextFromFile:
    """Testing extract_text_from_file function."""

    def test_extract_text_from_pdf(self):
        file = open("tests/test.pdf", "r")

        assert extract_text_from_file(file, "tests/test.pdf") == "Test image text."

    def test_extract_text_from_doc(self):
        file = open("tests/test.docx", "r")

        assert extract_text_from_file(file, "tests/test.docx") == "Test doc text.\n"

    def test_blank_file(self):
        file = open("tests/blank.pdf", "r")

        assert extract_text_from_file(file, "tests/blank.pdf") == None


class TestExtractTextFromImage:
    """Testing extract_text_from_image function."""

    def test_extract_text_from_image(self):
        assert extract_text_from_image("tests/test.pdf") == "Test image text."

    def test_blank_image(self):
        assert extract_text_from_image("tests/blank.pdf") == ""