# 1. Test Setup
# - Create mock Chroma class
# - Create mock Collection class
# - Mock torch.cuda functionality
# - Setup logging capture

# 2. Test Initialization
# - Test vectorstores are None after init

# 3. Test Register Vectorstores
# - Create mock stores
# - Register stores
# - Verify stores are properly set

# 4. Test Clear Context
# a) Test normal operation:
#     - Setup mock stores with collections
#     - Verify delete is called
#     - Verify stores are set to None
#     - Verify gc.collect called

# b) Test error handling:
#     - Test with None collections
#     - Test with missing ids
#     - Test with collection.delete raising exception

# c) Test CUDA cleanup:
#     - Mock cuda available true/false
#     - Verify cuda cleanup called when available

# 5. Test Edge Cases
# - Test clear_context with unregistered stores
# - Test multiple clear_context calls
# - Test register_vectorstores multiple times

import pytest
from context_manager import ContextManager
from unittest.mock import Mock
from langchain_chroma import Chroma


class TestContextManager:
    def test_init(self):
        manager = ContextManager()
        assert manager._current_job_vectorstore is None
        assert manager._current_resume_vectorstore is None

    def test_register_vectorstores(self):
        manager = ContextManager()
        mock_job_store = Mock(spec=Chroma)
        mock_resume_store = Mock(spec=Chroma)

        manager.register_vectorstores(mock_job_store, mock_resume_store)

        assert manager._current_job_vectorstore == mock_job_store
        assert manager._current_resume_vectorstore == mock_resume_store

    def test_clear_context_empty_stores(self):
        manager = ContextManager()
        manager.clear_context()  # Should not raise
        assert manager._current_job_vectorstore is None
        assert manager._current_resume_vectorstore is None

    def test_clear_context_with_stores(self):
        manager = ContextManager()
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": ["1", "2"]}

        mock_job_store = Mock(spec=Chroma)
        mock_job_store._collection = mock_collection

        mock_resume_store = Mock(spec=Chroma)
        mock_resume_store._collection = mock_collection

        manager.register_vectorstores(mock_job_store, mock_resume_store)

        manager.clear_context()

        mock_collection.delete.assert_called_with(ids=["1", "2"])
        assert manager._current_job_vectorstore is None
        assert manager._current_resume_vectorstore is None
