import atexit
import os
import shutil
import streamlit as st
import uuid
from io import BytesIO
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from extract_text import extract_text_from_uploaded_files
from analyze import analyze_inputs
import config
from state_manager import (
    initialize_state,
    reset_state_analysis,
    new_ollama_instance,
    save_analysis_results,
)
from components.file_upload import render_file_upload
from components.text_display import render_text_display
from components.output_experience import render_output_experience
from components.chatbot import render_chatbot, handle_chat
from components.job_input import render_job_input
from logger import setup_logging


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
        # initialize_state(st)
        # app_state = st.session_state.app_state
        st.session_state.result = ""
        st.session_state.job_retriever = None
        st.session_state.resume_retriever = None
        st.session_state.job_text = ""
        st.session_state.chat_history = None
        st.session_state.initialized = False
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.job_rows = [
            {"job": None, "start_date": None, "end_date": None, "description": ""}
        ]
        st.session_state.job_info = []

        logger.info("Starting the Resume Experience Analyzer app...")
        os.makedirs(config.app_config.TEMP_DIR, exist_ok=True)
        # st.session_state.app_state.initialized = True
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

    # app_state = st.session_state.app_state
    ollama = new_ollama_instance()

    st.title("Resume Experience Analyzer")

    st.write(
        "This app will compare a job description to a resume and extract the number of years of relevant work experience from the resume."
    )

    with st.container(border=True):

        job_file = render_file_upload()

        if job_file is None:
            st.write(
                "Please upload a job description to auto-populate the text area below."
            )
        else:
            st.session_state.job_text = extract_text_from_uploaded_files(
                job_file, config.app_config.TEMP_DIR
            )

        render_text_display(st.session_state.job_text)

        render_job_input()

    if st.button("Analyze", use_container_width=True):
        start_time = datetime.now()
        reset_state_analysis(st)

        job_info = ""

        # TODO: figure out how to handle overlapping dates...
        for i in range(len(st.session_state.job_rows)):
            data = st.session_state.job_rows[i]
            logger.info(f"data {i}: {data}")

            job_length = relativedelta(data["end_date"], data["start_date"])

            job_info += f"JOB {i + 1} | "
            job_info += f"{data["job"]} | "
            job_info += f"{job_length.years} years, {job_length.months} months, {job_length.days} days | "
            job_info += f"{data["description"]}\n"

        logger.info(f"job_info: {job_info}")

        with st.spinner("Processing (this can take a few minutes)..."):
            if not st.session_state.job_text or not job_info:
                st.write("Please upload a job description and a resume first.")
                return

            log_texts(
                st.session_state.job_text,
                job_info,
            )

            new_result, new_job_retriever, new_resume_retriever = analyze_inputs(
                st,
                st.session_state.job_text,
                job_info,
                ollama,
            )

            if isinstance(new_result, str) and "Failed" in new_result:
                st.error(new_result)
                return

            logger.info(f"new_job_retriever: {new_job_retriever}")
            logger.info(
                f"new_job_retriever content: {new_job_retriever.vectorstore.similarity_search("What is the role?")}"
            )
            logger.info(f"new_resume_retriever: {new_resume_retriever}")
            logger.info(
                f"new_resume_retriever content: {new_resume_retriever.vectorstore.similarity_search("What is the name?")}"
            )

            # app_state.save_analysis_results(
            #     new_result, new_job_retriever, new_resume_retriever
            # )
            save_analysis_results(
                st, new_result, new_job_retriever, new_resume_retriever
            )
            logger.info(f"Time spent analyzing: {datetime.now() - start_time}")

    # render_output_experience(app_state.result)
    render_output_experience(st.session_state.result)

    if st.session_state.job_retriever:
        st.write(st.session_state.job_retriever)
        st.write(
            f"job_retriever content: {st.session_state.job_retriever.vectorstore.similarity_search("What is the role?")}"
        )

    if st.session_state.resume_retriever:
        st.write(st.session_state.resume_retriever)
        st.write(
            f"resume_retriever content: {st.session_state.resume_retriever.vectorstore.similarity_search("What is the name?")}"
        )

    user_input = render_chatbot()

    if user_input is not None:
        start_time = datetime.now()
        # job_text = st.session_state.job_text
        # resume_text = st.session_state.resume_text
        # job_retriever = st.session_state.job_retriever
        # resume_retriever = st.session_state.resume_retriever

        if (
            (st.session_state.job_text == "")
            or (st.session_state.job_retriever == None)
            or (st.session_state.resume_retriever == None)
        ):
            st.write(
                "Please input the job description, the job experience from the resume, and perform an analysis first."
            )
            return

        logger.info(f"st.session_state {st.session_state}")
        logger.info(f"job_retriever: {st.session_state.job_retriever}")
        logger.info(
            f"job_retriever content: {st.session_state.job_retriever.vectorstore.similarity_search("What is the role?")}"
        )
        logger.info(f"resume_retriever: {st.session_state.resume_retriever}")
        logger.info(
            f"resume_retriever content: {st.session_state.resume_retriever.vectorstore.similarity_search("What is the name?")}"
        )

        handle_chat(
            user_input,
            st.session_state.job_retriever,
            st.session_state.resume_retriever,
            ollama,
        )

        logger.info(f"Time spent creating chat response: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()
