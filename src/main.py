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
)
import logging
from datetime import datetime

TEMP_DIR = "./temp_dir"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if "result" not in st.session_state:
    st.session_state.result = ""
if "job_ad_retriever" not in st.session_state:
    st.session_state.job_ad_retriever = None
if "resume_retriever" not in st.session_state:
    st.session_state.resume_retriever = None
if "job_ad_text" not in st.session_state:
    st.session_state.job_ad_text = ""
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

    job_ad_text = ""
    resume_text = ""
    result = ""

    with st.form("input_file_form"):
        with col1:
            job_ad_file = st.file_uploader(
                "Upload Job Description", type=["pdf", "doc", "docx"], key="job_ad_file"
            )

        with col2:
            resume_file = st.file_uploader(
                "Upload Resume", type=["pdf", "doc", "docx"], key="resume_file"
            )

        if st.form_submit_button("Extract Text"):
            start_time = datetime.now()
            if job_ad_file is None and resume_file is None:
                st.write("Please upload a job description and a resume first.")
                return

            job_ad_text = extract_text("job ad", job_ad_file, TEMP_DIR)
            resume_text = extract_text("resume", resume_file, TEMP_DIR)
            st.session_state["job_ad_text"] = job_ad_text
            st.session_state["resume_text"] = resume_text

            logger.info(f"Time spent extracting text: {datetime.now() - start_time}")


    col3, col4 = st.columns(2)
    col3.text_area("Job Description", value=st.session_state["job_ad_text"], height=300)
    col4.text_area("Resume Text", value=st.session_state["resume_text"], height=300)

    if st.button("Analyze"):
        start_time = datetime.now()
        st.session_state["result"] = ""
        st.session_state["job_ad_retriever"] = None
        st.session_state["resume_retriever"] = None

        job_ad_text = st.session_state["job_ad_text"]
        resume_text = st.session_state["resume_text"]

        with st.spinner("Processing (this can take a few minutes)..."):
            logger.info("job_ad_text: ", job_ad_text[:100])
            logger.info("resume_text: ", resume_text[:100])
            if (job_ad_text == "") or (resume_text == ""):
                st.write("Please upload a job description and a resume first.")
                return

            result, job_ad_retriever, resume_retriever = analyze_inputs(
                job_ad_text, resume_text
            )
            st.session_state["result"] = result.content
            logger.info(f"result_split: {result.content.split("|")}")
            st.session_state["job_ad_retriever"] = job_ad_retriever
            st.session_state["resume_retriever"] = resume_retriever

            logger.info(f"Time spent analyzing: {datetime.now() - start_time}")

    # Display output experience
    st.subheader("Output Experience")
    if st.session_state["result"] != "":
        split_result = st.session_state["result"].split("|")
        split_result = [x.strip() for x in split_result]
        st.write(split_result[0])
        st.write(split_result[1])
        st.write(split_result[2])

    # Chatbot feature
    st.subheader("Chatbot Feature")
    user_input = st.text_input("Ask a question:")

    if st.button("Submit Query"):
        start_time = datetime.now()
        job_ad_text = st.session_state["job_ad_text"]
        resume_text = st.session_state["resume_text"]

        if (job_ad_text == "") or (resume_text == ""):
            st.write("Please upload and analyze a job description and a resume first.")
            return

        st.session_state["chat_history"].append({"role": "User", "content": user_input})

        job_ad_retriever = st.session_state["job_ad_retriever"]
        resume_retriever = st.session_state["resume_retriever"]
        response = get_chat_response(
            user_input, job_ad_retriever, resume_retriever
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
