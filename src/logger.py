import logging
import config
import threading
import uuid
import streamlit as st


def setup_logging(session_id: str = None) -> logging.Logger:
    """Configure logging with a custom fomatter that includes a unique request ID."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.getLevelName(config.app_config.LOG_LEVEL))

    thread_local = threading.local()
    if st.session_state.get("session_id"):
        # print(f"session_state.session_id: {st.session_state.session_id}")
        thread_local.request_id = st.session_state.session_id
    elif session_id:
        # print(f"session_id: {session_id}")
        thread_local.request_id = session_id
    else:
        # print("no session_id")
        thread_local.request_id = str(uuid.uuid4())[:8]
    # print(f"request_id: {thread_local.request_id}")

    class RequestIdFilter(logging.Filter):
        def filter(self, record):
            record.request_id = getattr(thread_local, "request_id", "")
            return True

    formatter = logging.Formatter(
        "%(asctime)s [%(request_id)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    request_id_filter = RequestIdFilter()
    stream_handler.addFilter(request_id_filter)

    logger.handlers.clear()

    logger.addHandler(stream_handler)

    return logger
