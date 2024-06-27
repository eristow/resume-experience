from streamlit.testing.v1 import AppTest

def test_increment_and_add():
    """A user increments the number input, then clicks Add"""
    at = AppTest.from_file("Predict_prior_experience.py").run()
    # Ensure the component exists before interacting with it
    assert len(at.number_input) > 0, "No number input found in the app."
    at.number_input[0].increment().run()
    assert len(at.button) > 0, "No button found in the app."
    at.button[0].click().run()
    assert "Relevant Experience" in at.markdown[0].value
