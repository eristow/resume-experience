from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path
import os


@dataclass
class AppConfig:
    TEMP_DIR: Path = Path("/tmp/resume-experience")
    LOG_LEVEL: str = "INFO"
    CONTEXT_WINDOW: int = 8192
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 200
    OCR_LANG: str = "eng"
    SUPPORTED_FILE_TYPES: list = field(default_factory=lambda: ["pdf", "doc", "docx"])
    OLLAMA_BASE_URL: str = "http://ollama:11434"

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
        )


global app_config
app_config = AppConfig.from_env()
