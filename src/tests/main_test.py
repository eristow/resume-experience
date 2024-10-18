from streamlit.testing.v1 import AppTest


def test_app_runs_without_exception():
    at = AppTest.from_file("main.py", default_timeout=10)
    at.run()

    assert not at.exception
