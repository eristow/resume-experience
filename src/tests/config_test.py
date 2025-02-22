from pathlib import Path
import pytest
import os
from config import AppConfig


class TestAppConfig:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("TEMP_DIR", "/custom/path")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        # monkeypatch.setenv("CONTEXT_WINDOW", "2048")
        monkeypatch.setenv("CHUNK_SIZE", "512")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        monkeypatch.setenv("OCR_LANG", "fra")
        monkeypatch.setenv("SUPPORTED_FILE_TYPES", "pdf,txt")
        # monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("ENABLE_DEV_FEATURES", "False")

        config = AppConfig.from_env()

        assert config.TEMP_DIR == Path("/custom/path")
        assert config.LOG_LEVEL == "DEBUG"
        # assert config.CONTEXT_WINDOW == 2048
        assert config.CHUNK_SIZE == 512
        assert config.CHUNK_OVERLAP == 100
        assert config.OCR_LANG == "fra"
        assert config.SUPPORTED_FILE_TYPES == ["pdf", "txt"]
        # assert config.OLLAMA_BASE_URL == "http://localhost:11434"
        assert config.ENABLE_DEV_FEATURES == False

    def test_supported_file_types_parsing(self, monkeypatch):
        monkeypatch.setenv("SUPPORTED_FILE_TYPES", "pdf,txt,docx,odt")
        config = AppConfig.from_env()
        assert config.SUPPORTED_FILE_TYPES == ["pdf", "txt", "docx", "odt"]

    def test_temp_dir_is_path_object(self, monkeypatch):
        monkeypatch.setenv("TEMP_DIR", "/custom/path")
        config = AppConfig.from_env()
        assert isinstance(config.TEMP_DIR, Path)

    def test_numeric_env_vars_validation(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_WINDOW", "invalid")
        with pytest.raises(ValueError):
            AppConfig.from_env()
