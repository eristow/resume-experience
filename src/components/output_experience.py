import streamlit as st


def render_output_experience(result: str) -> None:
    """Render output experience section"""
    st.header("Output Experience")

    if result != "":
        split_result = result.split("|")
        split_result = [x.strip() for x in split_result]
        for result in split_result:
            st.write(result)
