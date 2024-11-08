import atexit
import os
import shutil
import streamlit as st
from io import BytesIO
import logging
from datetime import datetime
from dotenv import load_dotenv
import threading
import uuid
from extract_text import extract_text_from_uploaded_files
from analyze import analyze_inputs
import config
from state_manager import initialize_state, reset_state_analysis, new_ollama_instance
from components.file_upload import render_file_upload
from components.text_display import render_text_display
from components.output_experience import render_output_experience
from components.chatbot import render_chatbot, handle_chat


def setup_logging():
    """Configure logging with a custom fomatter that includes a unique request ID."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.getLevelName(config.app_config.LOG_LEVEL))

    thread_local = threading.local()
    thread_local.request_id = str(uuid.uuid4())[:8]

    class RequestIdFilter(logging.Filter):
        def filter(self, record):
            record.request_id = getattr(thread_local, "request_id", "")
            return True

    formatter = logging.Formatter(
        "%(asctime)s [%(request_id)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    request_id_filter = RequestIdFilter()
    stream_handler.addFilter(request_id_filter)

    logger.handlers.clear()

    logger.addHandler(stream_handler)

    return logger


logger = setup_logging()


def log_texts(job_text: str, resume_text: str) -> None:
    """Log the job and resume texts."""
    job_text_log = job_text[:100] if len(job_text) > 100 else job_text
    resume_text_log = resume_text[:100] if len(resume_text) > 100 else resume_text

    logger.info(f"job_text: %s", job_text_log)
    logger.info(f"resume_text: %s", resume_text_log)


def initialize_app():
    """Initialize the app on first run."""
    if not hasattr(st.session_state, "initialized"):
        load_dotenv()
        initialize_state(st)
        app_state = st.session_state.app_state

        logger.info("Starting the Resume Experience Analyzer app...")
        os.makedirs(config.app_config.TEMP_DIR, exist_ok=True)
        st.session_state.initialized = True


def cleanup():
    """Clean up resources when the session ends."""
    if os.path.exists(config.app_config.TEMP_DIR):
        shutil.rmtree(config.app_config.TEMP_DIR)
        logger.info("Closing the Resume Experience Analyzer app...")


atexit.register(lambda: shutil.rmtree(config.app_config.TEMP_DIR, ignore_errors=True))


# Streamlit UI components
def main():
    """
    Main function for the Resume Experience Analyzer application.

    This function handles the user interface and logic for uploading job descriptions and resumes,
    analyzing the inputs, and displaying the results.
    """
    initialize_app()

    app_state = st.session_state.app_state
    ollama = new_ollama_instance()

    st.title("Resume Experience Analyzer")

    st.write(
        "This app will compare a job description to a resume and extract the number of years of relevant work experience from the resume."
    )
    job_file, resume_file = render_file_upload()

    if job_file is None and resume_file is None:
        st.write(
            "Please upload a job description and a resume to auto-populate the fields below."
        )
    else:
        app_state.job_text, app_state.resume_text = extract_text_from_uploaded_files(
            job_file, resume_file, config.app_config.TEMP_DIR
        )

    render_text_display(app_state.job_text, app_state.resume_text)

    if st.button("Analyze"):
        start_time = datetime.now()
        reset_state_analysis(st)

        job_text = app_state.job_text
        resume_text = app_state.resume_text

        with st.spinner("Processing (this can take a few minutes)..."):
            if not job_text or not resume_text:
                st.write("Please upload a job description and a resume first.")
                return

            log_texts(job_text, resume_text)

            result, job_retriever, resume_retriever = analyze_inputs(
                job_text,
                resume_text,
                ollama,
            )

            if isinstance(result, str) and "Failed" in result:
                st.error(result)
                return

            app_state.save_analysis_results(result, job_retriever, resume_retriever)
            logger.info(f"Time spent analyzing: {datetime.now() - start_time}")

    render_output_experience(app_state.result)

    user_input = render_chatbot()

    if user_input is not None:
        start_time = datetime.now()
        job_text = app_state.job_text
        resume_text = app_state.resume_text
        job_retriever = app_state.job_retriever
        resume_retriever = app_state.resume_retriever

        if (job_text == "") or (resume_text == ""):
            st.write("Please upload and analyze a job description and a resume first.")
            return

        handle_chat(user_input, job_retriever, resume_retriever, ollama)

        logger.info(f"Time spent creating chat response: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()
