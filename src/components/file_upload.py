import streamlit as st
from typing import Tuple, Optional
from streamlit.runtime.uploaded_file_manager import UploadedFile
from io import BytesIO


def render_file_upload() -> Optional[UploadedFile]:
    """Render file upload section and return uploaded files"""

    with st.form("input_file_form", border=False):
        job_file = st.file_uploader(
            "Upload Job Ad",
            type=["pdf", "doc", "docx"],
            key="job_file",
            disabled=st.session_state.analysis_confirmed
            or st.session_state.extracting_text,
        )

        submit = st.form_submit_button(
            "Extract Text",
            disabled=st.session_state.analysis_confirmed
            or st.session_state.extracting_text,
        )

    return job_file if submit else None
