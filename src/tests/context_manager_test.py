import pytest
from context_manager import ContextManager
from unittest.mock import Mock, patch
from langchain_chroma import Chroma
import logging

logger = logging.getLogger("root")


class TestContextManager:
    @pytest.fixture
    def mock_request_id(self):
        with patch("context_manager.uuid.uuid4") as mock_request_id:
            mock_request_id.return_value = "1234"
            yield mock_request_id

    def test_init(self):
        manager = ContextManager()

        assert manager._contexts == {}
        assert manager._lock is not None

    def test_create_request_context(self):
        manager = ContextManager()
        request_id = manager.create_request_context()

        assert request_id is not None
        assert request_id in manager._contexts
        assert manager._contexts[request_id] == {
            "job_vectorstore": None,
            "resume_vectorstore": None,
        }

    def test_register_vectorstores(self, mock_request_id):
        manager = ContextManager()
        mock_job_store = Mock(spec=Chroma)
        mock_resume_store = Mock(spec=Chroma)

        request_id = manager.create_request_context()
        manager.register_vectorstores(request_id, mock_job_store, mock_resume_store)

        assert manager._contexts[request_id]["job_vectorstore"] == mock_job_store
        assert manager._contexts[request_id]["resume_vectorstore"] == mock_resume_store

    def test_clear_context_empty_stores(self, mock_request_id):
        manager = ContextManager()

        request_id = manager.create_request_context()
        manager.clear_context(request_id)  # Should not raise

        assert manager._contexts == {}

    def test_clear_context_with_stores(self, mock_request_id):
        manager = ContextManager()
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": ["1", "2"]}

        mock_job_store = Mock(spec=Chroma)
        mock_job_store._collection = mock_collection

        mock_resume_store = Mock(spec=Chroma)
        mock_resume_store._collection = mock_collection

        request_id = manager.create_request_context()
        manager.register_vectorstores(request_id, mock_job_store, mock_resume_store)

        manager.clear_context(request_id)

        mock_collection.delete.assert_called_with(ids=["1", "2"])
        assert manager._contexts == {}
