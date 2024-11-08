import streamlit as st


def render_text_display(job_text: str, resume_text: str) -> None:
    """Render text display section"""
    col3, col4 = st.columns(2)
    col3.text_area("Job Description", value=job_text, height=300)
    col4.text_area("Resume Text", value=resume_text, height=300)
