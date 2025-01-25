from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path
import os


@dataclass
class AppConfig:
    TEMP_DIR: Path
    LOG_LEVEL: str
    CONTEXT_WINDOW: int
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    OCR_LANG: str
    SUPPORTED_FILE_TYPES: list
    OLLAMA_BASE_URL: str
    ENABLE_DEV_FEATURES: bool

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables"""
        return cls(
            TEMP_DIR=Path(os.getenv("TEMP_DIR", "/tmp/resume-experience")),
            LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
            CONTEXT_WINDOW=int(os.getenv("CONTEXT_WINDOW", 8192)),
            CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", 1024)),
            CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP", 200)),
            OCR_LANG=os.getenv("OCR_LANG", "eng"),
            SUPPORTED_FILE_TYPES=os.getenv(
                "SUPPORTED_FILE_TYPES", "pdf,doc,docx"
            ).split(","),
            OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            ENABLE_DEV_FEATURES=os.getenv("ENABLE_DEV_FEATURES", "False") == "True",
        )


global app_config
app_config = AppConfig.from_env()
