import streamlit as st


def render_text_display(job_text: str) -> None:
    """Render text display section"""
    st.text_area("Job Description", value=job_text, height=300)
