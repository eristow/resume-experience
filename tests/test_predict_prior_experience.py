from streamlit.testing.v1 import AppTest

def test_increment_and_add():
    """A user increments the number input, then clicks Add"""
    at = AppTest.from_file("Predict_prior_experience.py").run()
    # Ensure the component exists before interacting with it
    assert len(at.file_uploader) > 0, "No file uploader found in the app."
    assert len(at.text_area) > 0, "No text area found in the app."
    assert len(at.button) > 0, "No button found in the app."
    at.button[0].click().run()
    assert "Relevant Experience" in at.markdown[0].value

def test_no_interaction():
    at = AppTest.from_file("Predict_prior_experience.py").run()
    assert len(at.file_uploader) == 2, "File uploaders not found"
    assert len(at.text_area) == 2, "Text areas not found"
    assert "Generating as soon as Resume and Job Description are filled." in at.markdown[0].value
