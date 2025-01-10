import streamlit as st
from logger import setup_logging

logger = setup_logging()


def create_job_row(num):
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            job_input = st.text_input(f"Job {num}")
        with col2:
            start_date_input = st.date_input(f"Start Date {num}")
        with col3:
            end_date_input = st.date_input(f"End Date {num}")
        job_description = st.text_area(f"Job Description {num}")

        if st.button(f"Delete row {num}", type="primary", icon="❌"):
            return {}

    return {
        "job": job_input,
        "start_date": start_date_input,
        "end_date": end_date_input,
        "description": job_description,
    }


def render_job_input():
    st.write(st.session_state.job_rows)

    if st.button("Add new job row"):
        st.session_state.job_rows.append(
            {"job": None, "start_date": None, "end_date": None, "description": ""}
        )

    for i, row in enumerate(st.session_state.job_rows):
        new_row = create_job_row(i + 1)

        if len(new_row.keys()) is 0:
            del st.session_state.job_rows[i]
        else:
            st.session_state.job_rows[i] = new_row
