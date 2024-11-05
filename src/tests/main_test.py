from streamlit.testing.v1 import AppTest
import pytest
from unittest.mock import Mock, patch
import streamlit as st
import os
import shutil


class TestMainApp:
    @pytest.fixture
    def mock_state(self):
        with patch("streamlit.session_state", {}) as mock_state:
            yield mock_state

    @pytest.fixture
    def cleanup_temp(self):
        yield
        if os.path.exists("temp"):
            shutil.rmtree("temp")

    def test_app_runs_without_exception(self):
        at = AppTest.from_file("main.py", default_timeout=10)
        at.run()

        assert not at.exception

    # def test_initialize_app(self, mock_state):
    #     with patch("main.intialize_state") as mock_init:
    #         from main import initialize_app

    #         initialize_app()
    #         mock_init.assert_called_once()
    #         assert st.session_state.initialized == True

    # def test_cleanup(self, cleanup_temp):
    #     from main import cleanup

    #     os.makedirs("temp", exist_ok=True)
    #     cleanup()
    #     assert not os.path.exists("temp")

    # def test_log_texts(self, caplog):
    #     from main import log_texts

    #     job_text = "test job" * 20  # More than 100 chars
    #     resume_text = "test resume" * 20  # More than 100 chars

    #     log_texts(job_text, resume_text)
    #     assert len(caplog.records) == 2
    #     assert "job_text" in caplog.records[0].message
    #     assert "resume_text" in caplog.records[1].message

    # def test_app_with_file_upload(self):
    #     at = AppTest.from_file("main.py")

    #     job_file = Mock()
    #     job_file.name = "test_job.pdf"
    #     resume_file = Mock()
    #     resume_file.name = "test_resume.pdf"

    #     with patch(
    #         "components.file_upload.render_file_upload",
    #         return_value=(job_file, resume_file),
    #     ):
    #         with patch(
    #             "extract_text.extract_text_from_uploaded_files",
    #             return_value=("job text", "resume text"),
    #         ):
    #             at.run()
    #             assert not at.exception

    # def test_app_analyze_flow(self):
    #     at = AppTest.from_file("main.py")

    #     st.session_state.app_state = Mock()
    #     st.session_state.app_state.job_text = "test job"
    #     st.session_state.app_state.resume_text = "test resume"

    #     with patch("analyze.analyze_inputs", return_value=("result", Mock(), Mock())):
    #         at.run()
    #         button = at.get_widget("button", "Analyze")
    #         button.click()
    #         assert not at.exception

    # def test_app_chat_flow(self):
    #     at = AppTest.from_file("main.py")

    #     st.session_state.app_state = Mock()
    #     st.session_state.app_state.job_text = "test job"
    #     st.session_state.app_state.resume_text = "test resume"
    #     st.session_state.app_state.job_retriever = Mock()
    #     st.session_state.app_state.resume_retriever = Mock()

    #     with patch("components.chatbot.handle_chat") as mock_chat:
    #         at.run()
    #         text_input = at.get_widget("text_input", "Ask a question")
    #         text_input.input("test question").run()
    #         mock_chat.assert_called_once()
    #         assert not at.exception

    # def test_app_error_handling(self):
    #     at = AppTest.from_file("main.py")

    #     with patch(
    #         "components.file_upload.render_file_upload", return_value=(None, None)
    #     ):
    #         at.run()
    #         assert "Please upload" in at.get_text_elements()[1]
    #         assert not at.exception

    # @pytest.mark.parametrize(
    #     "job_text,resume_text",
    #     [
    #         ("", "test"),
    #         ("test", ""),
    #         ("", ""),
    #     ],
    # )
    # def test_app_invalid_inputs(self, job_text, resume_text):
    #     at = AppTest.from_file("main.py")

    #     st.session_state.app_state = Mock()
    #     st.session_state.app_state.job_text = job_text
    #     st.session_state.app_state.resume_text = resume_text

    #     at.run()

    #     buttons = [el for el in at.button if el.label == "Analyze"]
    #     if buttons:
    #         buttons[0].click()

    #     text = [el for el in at.text if "Please upload" in str(el.value)]
    #     print(at.get("text"))
    #     assert text
