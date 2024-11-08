from analyze import (
    analyze_inputs,
    process_text,
    get_chat_response,
)

import pytest
from unittest.mock import Mock, patch, call
from langchain_community.embeddings import FakeEmbeddings


def mock_chat_ollama(input_data):
    class FakeResponse:
        def __init__(self, content):
            self.content = content

    return FakeResponse("Overall the resume is a good match for the job description.")


class TestAnalyzeInputs:
    """Testing analyze_inputs function."""

    @pytest.fixture
    def mock_verify_input_size(self):
        with patch("analyze.verify_input_size") as mock_verify:
            mock_verify.return_value = 100
            yield mock_verify

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
        with patch("analyze.process_text") as mock_process:
            mock_process.return_value = mock_vectorstore
            yield mock_process

    @pytest.fixture
    def mock_custom_embeddings(self, mock_embeddings):
        with patch("analyze.CustomEmbeddings") as mock_custom:
            mock_custom.return_value = mock_embeddings
            yield mock_custom

    @pytest.fixture
    def mock_tokenizer(self):
        with patch("analyze.AutoTokenizer") as mock_tokenizer:
            mock_instance = Mock()
            mock_instance.model_max_length = 32000
            mock_instance.encode.return_value = [1, 2, 3]
            mock_tokenizer.from_pretrained.return_value = mock_instance
            yield mock_tokenizer

    @pytest.fixture
    def mock_request_id(self):
        with patch("analyze.context_manager.create_request_context") as mock_request_id:
            mock_request_id.return_value = "1234"
            yield mock_request_id

    def test_analyze_inputs_valid(
        self,
        mock_verify_input_size,
        mock_process_text,
        mock_custom_embeddings,
        mock_vectorstore,
    ):
        job_text = "Sample job description"
        resume_text = "Sample resume"

        response, job_retriever, resume_retriever = analyze_inputs(
            job_text, resume_text, mock_chat_ollama
        )

        assert response.content[:8].strip() == "Overall"
        assert job_retriever == mock_vectorstore.as_retriever()
        assert resume_retriever == mock_vectorstore.as_retriever()

        assert mock_process_text.call_count == 2

        mock_custom_embeddings.assert_called_once_with(model_name="./models/mistral")

    def test_analyze_inputs_with_invalid_vectorstore(
        self,
        mock_verify_input_size,
        mock_custom_embeddings,
        mock_embeddings,
    ):
        with patch("analyze.process_text", return_value=None):
            response, job_retriever, resume_retriever = analyze_inputs(
                "job",
                "resume",
                mock_chat_ollama,
            )
            assert "Failed:" in response
            assert job_retriever is None
            assert resume_retriever is None

    @pytest.mark.parametrize(
        "job_text,resume_text",
        [
            ("", "resume"),
            ("job", ""),
            ("", ""),
        ],
    )
    def test_analyze_inputs_with_missing_inputs(
        self,
        job_text,
        resume_text,
        mock_verify_input_size,
        mock_custom_embeddings,
    ):
        job_text = None
        resume_text = None

        response, job_retriever, resume_retriever = analyze_inputs(
            job_text, resume_text, mock_chat_ollama
        )
        assert "Failed:" in response
        assert job_retriever is None
        assert resume_retriever is None

    def test_analyze_inputs_full_chain(
        self,
        mock_process_text,
        mock_custom_embeddings,
        mock_vectorstore,
        mock_embeddings,
        mock_tokenizer,
        mock_request_id,
    ):
        job_text = "Sample job description"
        resume_text = "Sample resume"

        response, job_retriever, resume_retriever = analyze_inputs(
            job_text, resume_text, mock_chat_ollama
        )

        mock_custom_embeddings.assert_called_once_with(model_name="./models/mistral")
        assert mock_process_text.call_count == 2

        mock_process_text.assert_has_calls(
            [
                call(job_text, mock_embeddings, "1234_job"),
                call(resume_text, mock_embeddings, "1234_resume"),
            ],
            any_order=True,
        )

        assert mock_vectorstore.as_retriever.call_count == 2

    def test_analyze_inputs_with_large_input(
        self,
        mock_custom_embeddings,
        mock_embeddings,
    ):
        with patch(
            "analyze.verify_input_size", side_effect=ValueError("Input too large")
        ):
            response, job_retriever, resume_retriever = analyze_inputs(
                "large job text",
                "large resume text",
                mock_chat_ollama,
            )
            assert "Failed: Input too large" in response
            assert job_retriever is None
            assert resume_retriever is None

    def test_analyze_inputs_with_processing_error(
        self,
        mock_verify_input_size,
        mock_custom_embeddings,
        mock_embeddings,
    ):
        with patch("analyze.process_text", side_effect=Exception("Processing error")):
            response, job_retriever, resume_retriever = analyze_inputs(
                "job",
                "resume",
                mock_chat_ollama,
            )
            assert "Failed: Unable to process the files." in response
            assert job_retriever is None
            assert resume_retriever is None


class TestProcessText:
    @pytest.fixture
    def mock_fake_embeddings(self):
        mock_embeddings = Mock(spec=FakeEmbeddings)
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_embeddings.get_tokenizer = Mock(return_value=mock_tokenizer)
        mock_embeddings.model_name = Mock(return_value="fake_model")
        return mock_embeddings

    @pytest.fixture
    def mock_chroma(self):
        with patch("analyze.Chroma") as mock_chroma:
            mock_chroma.from_texts_return_value = [1, 2, 3]
            yield mock_chroma

    def test_process_text_valid(self, mock_fake_embeddings, mock_chroma):
        text = "Sample text"
        vectorstore = process_text(text, mock_fake_embeddings, "1234")
        assert vectorstore is not None

    def test_process_text_with_invalid_text(self, mock_fake_embeddings):
        text = ""
        vectorstore = process_text(text, mock_fake_embeddings, "1234")
        assert vectorstore is None

    def test_process_text_with_invalid_embeddings(self):
        text = "Sample text"
        embeddings = None
        vectorstore = process_text(text, embeddings, "1234")
        assert vectorstore is None


class TestGetChatResponse:
    @pytest.fixture
    def mock_vectorstore(self):
        mock_vs = Mock()
        mock_retriever = Mock()
        mock_vs.as_retriever.return_value = mock_retriever
        return mock_vs

    def test_get_chat_response_valid(self, mock_vectorstore):
        response = get_chat_response(
            "Sample text",
            mock_vectorstore.as_retriever(),
            mock_vectorstore.as_retriever(),
            mock_chat_ollama,
        )

        assert response is not None
        assert (
            response.content
            == "Overall the resume is a good match for the job description."
        )

    def test_get_chat_response_with_invalid_text(self, mock_vectorstore):
        response = get_chat_response(
            "",
            mock_vectorstore.as_retriever(),
            mock_vectorstore.as_retriever(),
            mock_chat_ollama,
        )

        assert response is None

    def test_get_chat_response_with_invalid_retrievers(self, mock_vectorstore):
        response = get_chat_response(
            "Sample text",
            None,
            None,
            mock_chat_ollama,
        )

        assert response is None
