import logging
import threading
import uuid
from config import app_config


thread_local = threading.local()


def request_id_provider():
    """Provide the current request ID."""
    return getattr(thread_local, "request_id", "")


def setup_logging(name: str):
    """Configure logging with a custom fomatter that includes a unique request ID."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelName(app_config.LOG_LEVEL))

    thread_local.request_id = str(uuid.uuid4())[:8]

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
