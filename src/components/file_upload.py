import streamlit as st
from typing import Tuple, Optional
from streamlit.runtime.uploaded_file_manager import UploadedFile
from io import BytesIO


def render_file_upload() -> Tuple[Optional[UploadedFile], Optional[UploadedFile]]:
    """Render file upload section and return uploaded files"""
    col1, col2 = st.columns(2)

    with st.form("input_file_form"):
        with col1:
            job_file = st.file_uploader(
                "Upload Job Description", type=["pdf", "doc", "docx"], key="job_file"
            )

        with col2:
            resume_file = st.file_uploader(
                "Upload Resume", type=["pdf", "doc", "docx"], key="resume_file"
            )

        submit = st.form_submit_button("Extract Text")

    return (job_file, resume_file) if submit else (None, None)
