"""
main.py

This module provides the main function for the Resume Experience Analyzer application. It handles the user interface and logic for uploading job descriptions and resumes, analyzing the inputs, and displaying the results.
"""

import os
import streamlit as st
from tools import (
    extract_text,
    extract_text_from_file,
    analyze_inputs,
    get_chat_response,
    CONTEXT_WINDOW,
)
import logging
from datetime import datetime
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
import gc
import torch

TEMP_DIR = "/tmp"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
ollama = ChatOllama(
    model="mistral:v0.3",
    temperature=0.3,
    base_url=OLLAMA_BASE_URL,
    num_ctx=CONTEXT_WINDOW,
)

if "result" not in st.session_state:
    st.session_state.result = ""
if "job_retriever" not in st.session_state:
    st.session_state.job_retriever = None
if "resume_retriever" not in st.session_state:
    st.session_state.resume_retriever = None
if "job_text" not in st.session_state:
    st.session_state.job_text = ""
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Streamlit UI components
def main():
    """
    Main function for the Resume Experience Analyzer application.

    This function handles the user interface and logic for uploading job descriptions and resumes,
    analyzing the inputs, and displaying the results.

    Parameters:
    None

    Returns:
    None
    """
    st.title("Resume Experience Analyzer")

    st.write(
        "This app will compare a job description to a resume and extract the number of years of relevant work experience from the resume."
    )

    col1, col2 = st.columns(2)

    job_text = ""
    resume_text = ""
    result = ""

    with st.form("input_file_form"):
        with col1:
            job_file = st.file_uploader(
                "Upload Job Description", type=["pdf", "doc", "docx"], key="job_file"
            )

        with col2:
            resume_file = st.file_uploader(
                "Upload Resume", type=["pdf", "doc", "docx"], key="resume_file"
            )

        if st.form_submit_button("Extract Text"):
            start_time = datetime.now()
            if job_file is None and resume_file is None:
                st.write("Please upload a job description and a resume first.")
                return

            job_text = extract_text("job", job_file, TEMP_DIR)
            resume_text = extract_text("resume", resume_file, TEMP_DIR)
            st.session_state["job_text"] = job_text
            st.session_state["resume_text"] = resume_text

            logger.info(f"Time spent extracting text: {datetime.now() - start_time}")

    col3, col4 = st.columns(2)
    col3.text_area("Job Description", value=st.session_state["job_text"], height=300)
    col4.text_area("Resume Text", value=st.session_state["resume_text"], height=300)

    if st.button("Analyze"):
        start_time = datetime.now()
        st.session_state["result"] = ""
        if "job_retriever" in st.session_state:
            del st.session_state["job_retriever"]
        if "resume_retriever" in st.session_state:
            del st.session_state["resume_retriever"]

        # Force garbage collection
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        job_text = st.session_state["job_text"]
        resume_text = st.session_state["resume_text"]

        with st.spinner("Processing (this can take a few minutes)..."):
            if not job_text or not resume_text:
                st.write("Please upload a job description and a resume first.")
                return

            job_text_log = job_text[:100] if len(job_text) > 100 else job_text
            resume_text_log = (
                resume_text[:100] if len(resume_text) > 100 else resume_text
            )
            logger.info(f"job_text: %s", job_text_log)
            logger.info(f"resume_text: %s", resume_text_log)

            result, job_retriever, resume_retriever = analyze_inputs(
                job_text, resume_text, ollama
            )

            if isinstance(result, str) and "Failed" in result:
                st.error(result)
                return

            st.session_state["result"] = result.content
            logger.info(f"result_split: {result.content.split("|")}")
            st.session_state["job_retriever"] = job_retriever
            st.session_state["resume_retriever"] = resume_retriever

            logger.info(f"Time spent analyzing: {datetime.now() - start_time}")

    # Display output experience
    st.subheader("Output Experience")
    if st.session_state["result"] != "":
        split_result = st.session_state["result"].split("|")
        split_result = [x.strip() for x in split_result]
        for result in split_result:
            st.write(result)

    # Chatbot feature
    st.subheader("Chatbot Feature")
    user_input = st.text_input("Ask a question:")

    if st.button("Submit Query"):
        start_time = datetime.now()
        job_text = st.session_state["job_text"]
        resume_text = st.session_state["resume_text"]

        if (job_text == "") or (resume_text == ""):
            st.write("Please upload and analyze a job description and a resume first.")
            return

        st.session_state["chat_history"].append({"role": "User", "content": user_input})

        job_retriever = st.session_state["job_retriever"]
        resume_retriever = st.session_state["resume_retriever"]
        response = get_chat_response(
            user_input, job_retriever, resume_retriever, ollama
        ).content
        st.session_state["chat_history"].append(
            {"role": "Assistant", "content": response}
        )

        for chat in st.session_state["chat_history"]:
            st.write(f"{chat['role']}:\n{chat['content']}")

        logger.info(f"Time spent creating chat response: {datetime.now() - start_time}")


if __name__ == "__main__":
    # Create a temporary directory to save uploaded files
    os.makedirs(TEMP_DIR, exist_ok=True)

    main()

    # Clean up the temporary directory after the app closes
    import shutil

    shutil.rmtree(TEMP_DIR)
