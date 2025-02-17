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
    reset_state_analysis,
    new_ollama_instance,
    save_analysis_results,
)
from components.file_upload import render_file_upload
from components.text_display import render_text_display
from components.output_experience import render_output_experience
from components.chatbot import render_chatbot, handle_chat, cleanup_chat_resources
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
        st.session_state.using_dev_data = False
        st.session_state.enable_dev_features = config.app_config.ENABLE_DEV_FEATURES
        logger.info(
            f"init: enable_dev_features: {st.session_state.enable_dev_features}"
        )
        st.session_state.extracting_text = False
        st.session_state.analysis_confirmed = False

        logger.info("Starting the Resume Experience Analyzer app...")
        os.makedirs(config.app_config.TEMP_DIR, exist_ok=True)
        # st.session_state.app_state.initialized = True
        st.session_state.initialized = True


def cleanup():
    """Clean up resources when the session ends."""
    cleanup_chat_resources(st)
    if os.path.exists(config.app_config.TEMP_DIR):
        shutil.rmtree(config.app_config.TEMP_DIR)
        logger.info("Closing the Resume Experience Analyzer app...")


atexit.register(lambda: shutil.rmtree(config.app_config.TEMP_DIR, ignore_errors=True))


@st.dialog("Confirm Analysis Start")
def confirm_start_analysis():
    st.error(
        "Running analysis will clear your current chatbot history! Are you sure you want to begin an analysis?"
    )
    col1, col2, col3 = st.columns([2, 1, 1])

    with col2:
        if st.button("Cancel"):
            st.session_state.analysis_confirmed = False
            st.rerun()

    with col3:
        if st.button("Confirm", type="primary"):
            st.session_state.analysis_confirmed = True
            st.rerun()


# Streamlit UI components
def main():
    """
    Main function for the Resume Experience Analyzer application.

    This function handles the user interface and logic for uploading job ads and resumes,
    analyzing the inputs, and displaying the results.
    """
    initialize_app()

    ollama = new_ollama_instance()

    st.title("Resume Experience Analyzer")

    # st.write(
    #     "This app will compare a job ad to a resume and extract the number of years of relevant work experience from the resume."
    # )
    st.info(
        "Please do not interact with the app while it is processing the analysis or chatbot."
    )

    with st.container(border=True):

        job_file = render_file_upload()

        if job_file is None:
            st.write("Please upload a job ad to auto-populate the text area below.")
        else:
            st.session_state.extracting_text = True
            st.session_state.job_text = extract_text_from_uploaded_files(
                job_file, config.app_config.TEMP_DIR
            )
            st.session_state.extracting_text = False

        render_text_display(st.session_state.job_text)

        render_job_input()

    if st.button(
        "Analyze",
        use_container_width=True,
        disabled=st.session_state.analysis_confirmed
        or st.session_state.extracting_text,
    ):
        confirm_start_analysis()

    if st.session_state.analysis_confirmed:
        start_time = datetime.now()

        job_info = ""

        # Validate job rows
        valid_jobs = [
            row
            for row in st.session_state.job_rows
            if "job" in row
            and "start_date" in row
            and "end_date" in row
            and "is_part_time" in row
            and "description" in row
        ]

        # Create job info string
        job_info = ""
        job_lengths = []

        for i, data in enumerate(valid_jobs):
            job_length = relativedelta(data["end_date"], data["start_date"])
            if data["is_part_time"]:
                job_length = job_length / 2
            job_lengths.append(job_length)

            job_info += f"JOB {i + 1} | "
            job_info += f"{data['job']} | "
            job_info += f"Length of job: {job_length.years} years, {job_length.months} months, {job_length.days} days | "
            job_info += f"{data['description']}\n"

        total_job_lengths = None
        logger.info(f"job_lengths: {job_lengths}")
        for length in job_lengths:
            logger.info(f"length: {length}")
            if total_job_lengths is None:
                total_job_lengths = length
            else:
                total_job_lengths += length

        total_job_info = (
            f"Total length of jobs: {total_job_lengths.years} years, {total_job_lengths.months} months, {total_job_lengths.days} days | \n"
            + job_info
        )

        logger.info(f"total_job_info: {total_job_info}")

        with st.spinner("Processing (this can take a few minutes)..."):
            if not st.session_state.job_text or not total_job_info:
                st.write("Please upload a job ad and a resume first.")
                return

            log_texts(
                st.session_state.job_text,
                total_job_info,
            )

            new_result, new_job_retriever, new_resume_retriever = analyze_inputs(
                st.session_state.job_text,
                total_job_info,
                ollama,
            )

            if isinstance(new_result, str) and "Failed" in new_result:
                st.error(new_result)
                return

            logger.info(f"new_job_retriever: {new_job_retriever}")
            logger.info(f"new_resume_retriever: {new_resume_retriever}")

            # app_state.save_analysis_results(
            #     new_result, new_job_retriever, new_resume_retriever
            # )
            save_analysis_results(
                st, new_result, new_job_retriever, new_resume_retriever
            )
            logger.info(f"Time spent analyzing: {datetime.now() - start_time}")

    render_output_experience(st.session_state.result)

    # if st.session_state.job_retriever:
    #     st.write(st.session_state.job_retriever)
    #     st.write(
    #         f"job_retriever content: {st.session_state.job_retriever.vectorstore.similarity_search("What is the role?")}"
    #     )

    # if st.session_state.resume_retriever:
    #     st.write(st.session_state.resume_retriever)
    #     st.write(
    #         f"resume_retriever content: {st.session_state.resume_retriever.vectorstore.similarity_search("What is the name?")}"
    #     )

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
                "Please input the job ad, the job experience from the resume, and perform an analysis first."
            )
            return

        handle_chat(
            st,
            user_input,
            st.session_state.job_retriever,
            st.session_state.resume_retriever,
            ollama,
        )

        logger.info(f"Time spent creating chat response: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()
