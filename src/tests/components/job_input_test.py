import pytest
from components.job_input import render_job_input
from streamlit.testing.v1 import AppTest
import datetime


class TestRenderJobInput:
    def test_render_job_input_happy(self):
        at = AppTest.from_function(render_job_input)
        at.session_state.job_rows = [
            {
                "job": "test1",
                "start_date": "test2",
                "end_date": "test3",
                "description": "test4",
            }
        ]
        at.session_state.using_dev_data = False
        at.session_state.enable_dev_features = False
        at.run()

        assert not at.exception

        start_date = datetime.date(2025, 1, 13)
        at.date_input[0].set_value(start_date).run()
        assert at.date_input[0].value == start_date

        end_date = datetime.date(2025, 1, 12)
        at.date_input[1].set_value(end_date).run()
        assert at.date_input[1].value == end_date

        at.text_input[0].set_value("Software Engineer").run()
        assert at.text_input[0].value == "Software Engineer"

        at.text_area[0].set_value("Description here").run()
        assert at.text_area[0].value == "Description here"

        at.button[0].click().run()
        assert len(at.session_state.job_rows) == 2

        at.button[1].click().run()
        assert len(at.session_state.job_rows) == 1
