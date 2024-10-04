"""
main.py

This module provides the main function for the Resume Experience Analyzer application. It handles the user interface and logic for uploading job descriptions and resumes, analyzing the inputs, and displaying the results.
"""

import os
import streamlit as st
from tools import extract_text_from_file, analyze_inputs, get_chat_response

TEMP_DIR = "./temp_dir"

# Initialize chat history
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
    job_ad_retriever = None
    resume_retriever = None

    st.title("Resume Experience Analyzer")

    st.write(
        "This app will compare a job description to a resume and extract the number of years of relevant work experience from the resume."
    )

    col1, col2 = st.columns(2)

    job_ad_text = ""
    resume_text = ""

    with col1:
        job_file = st.file_uploader(
            "Upload Job Description", type=["pdf", "doc", "docx"], key="job_file"
        )
        if job_file:
            job_file_path = os.path.join(TEMP_DIR, job_file.name)
            with open(job_file_path, "wb") as temp_file:
                temp_file.write(job_file.getbuffer())
            job_ad_text = extract_text_from_file(job_file, job_file_path)

    with col2:
        resume_file = st.file_uploader(
            "Upload Resume", type=["pdf", "doc", "docx"], key="resume_file"
        )
        if resume_file:
            resume_file_path = os.path.join(TEMP_DIR, resume_file.name)
            with open(resume_file_path, "wb") as temp_file:
                temp_file.write(resume_file.getbuffer())
            resume_text = extract_text_from_file(resume_file, resume_file_path)

    col1.text_area("Job Description", value=job_ad_text, height=300)
    col2.text_area("Resume Text", value=resume_text, height=300)

    if st.button("Analyze"):
        with st.spinner("Processing (this can take a few minutes)..."):
            result, job_ad_retriever_temp, resume_retriever_temp = analyze_inputs(
                job_file_path, resume_file_path
            )
            st.session_state["result"] = result.content
            st.session_state["job_ad_retriever"] = job_ad_retriever_temp
            st.session_state["resume_retriever"] = resume_retriever_temp

    if "result" in st.session_state:
        st.write(st.session_state["result"])

    # Display output experience
    # st.subheader("Output Experience")
    # st.write("Overall Experience: X years")
    # st.write("Relevant Experience: X years")
    # st.write("Notes: ...")

    # Chatbot feature
    st.subheader("Chatbot Feature")
    user_input = st.text_input("Ask a question:")

    if st.button("Submit Query"):
        if ("job_ad_retriever" not in st.session_state) or (
            "resume_retriever" not in st.session_state
        ):
            st.write("Please upload and analyze a job description and a resume first.")
            return

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]

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


if __name__ == "__main__":
    # Create a temporary directory to save uploaded files
    os.makedirs(TEMP_DIR, exist_ok=True)

    main()

    # Clean up the temporary directory after the app closes
    import shutil

    shutil.rmtree(TEMP_DIR)
