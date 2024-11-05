from extract_text import (
    extract_text_from_uploaded_files,
    extract_text,
    extract_text_from_file,
    extract_text_from_image,
)
import custom_embeddings
import os
import pytest
from unittest.mock import Mock, patch, call
from langchain_community.chat_models import ChatOllama
from streamlit.runtime.uploaded_file_manager import UploadedFile


class TestExtractTextFromUploadedFiles:
    """Testing extract_text_from_uploaded_files function."""

    def test_extract_text_from_uploaded_files_valid(self):
        job_file = Mock(spec=UploadedFile)
        resume_file = Mock(spec=UploadedFile)
        job_file.name = "job.pdf"
        resume_file.name = "resume.pdf"

        with patch("extract_text.extract_text", return_value="Sample text"):
            job_text, resume_text = extract_text_from_uploaded_files(
                job_file, resume_file, "./tests/temp/"
            )
            assert job_text == "Sample text"
            assert resume_text == "Sample text"

    def test_extract_text_from_uploaded_files_none_files(self):
        job_text, resume_text = extract_text_from_uploaded_files(
            None, None, "./tests/temp/"
        )
        assert job_text is None
        assert resume_text is None


class TestExtractText:
    """Testing extract_text function."""

    @pytest.fixture(scope="session", autouse=True)
    def cleanup_temp_dir(self):
        yield
        if os.path.exists("tests/temp/test.pdf"):
            os.remove("tests/temp/test.pdf")

    def test_extract_job(self):
        file = open("tests/test.pdf", "r")

        assert extract_text("job", file, "./tests/temp/") == "Test image text."

    def test_extract_text_invalid_type(self):
        file = Mock(spec=UploadedFile)
        file.name = "test.pdf"

        result = extract_text("invalid", file, "./tests/temp/")
        assert result is None

    def test_extract_text_file_missing_file(self):
        file = Mock(spec=UploadedFile)
        file.name = "nonexistent.pdf"

        result = extract_text("job", file, "./tests/temp/")
        assert result is None

    def test_extract_text_unsupported_extension(self):
        file = Mock(spec=UploadedFile)
        file.name = "test.xyz"

        result = extract_text("job", file, "./tests/temp/")
        assert result is None


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

    def test_extract_text_from_file_unsupported_type(self):
        file = Mock(spec=UploadedFile)
        file.name = "test.xyz"

        result = extract_text_from_file(file, "test.xyz")
        assert result is None

    @patch("docx.Document")
    def test_extract_text_from_file_corrupted_docx(self, mock_docx):
        file = Mock(spec=UploadedFile)
        file.name = "test.docx"
        mock_docx.side_effect = Exception("Corrupted file")

        result = extract_text_from_file(file, "test.docx")
        assert result is None


class TestExtractTextFromImage:
    """Testing extract_text_from_image function."""

    def test_extract_text_from_image(self):
        assert extract_text_from_image("tests/test.pdf") == "Test image text."

    def test_blank_image(self):
        assert extract_text_from_image("tests/blank.pdf") == ""

    @patch("pdf2image.convert_from_path")
    def test_extract_text_from_image_error(self, mock_convert):
        mock_convert.side_effect = Exception("PDF conversion error")

        result = extract_text_from_image("test.pdf")
        assert result == ""

    def test_extract_text_from_missing_image(self):
        assert extract_text_from_image("tests/nonexistent.pdf") == ""
