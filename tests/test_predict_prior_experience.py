from streamlit.testing.v1 import AppTest

def test_increment_and_add():
    """A user increments the number input, then clicks Add"""
    at = AppTest.from_file("Predict_prior_experience.py").run()
    at.number_input[0].increment().run()
    assert at.number_input[0].value == 1  # Example assertion
    at.button[0].click().run()
    assert "Expected output" in at.output  # Adjust the expected output based on your app's behavior
