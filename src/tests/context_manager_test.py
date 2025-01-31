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
from context_manager import ContextManager, UsageLock
from unittest.mock import Mock
from langchain_chroma import Chroma
import uuid


class TestUsageLock:
    def test_init(self):
        usage_lock = UsageLock()
        assert usage_lock.lock_engaged is False
        assert usage_lock.user_uuid is None

    def test_take_lock_happy(self):
        usage_lock = UsageLock()
        user_uuid = str(uuid.uuid4())[:8]

        usage_lock.take_lock(user_uuid)

        assert usage_lock.lock_engaged is True
        assert usage_lock.user_uuid is user_uuid

    def test_take_lock_already_taken(self):
        usage_lock = UsageLock()
        usage_lock.lock_engaged = True
        other_user_uuid = str(uuid.uuid4())[:8]
        usage_lock.user_uuid = other_user_uuid
        user_uuid = str(uuid.uuid4())[:8]

        assert usage_lock.lock_engaged is True

        usage_lock.take_lock(user_uuid)

        assert usage_lock.lock_engaged is True
        assert usage_lock.user_uuid is other_user_uuid

    def test_release_lock_happy(self):
        usage_lock = UsageLock()
        usage_lock.lock_engaged = True
        user_uuid = str(uuid.uuid4())[:8]
        usage_lock.user_uuid = user_uuid

        usage_lock.release_lock(user_uuid)

        assert usage_lock.lock_engaged is False
        assert usage_lock.user_uuid is None

    def test_release_lock_already_taken(self):
        usage_lock = UsageLock()
        usage_lock.lock_engaged = True
        other_user_uuid = str(uuid.uuid4())[:8]
        usage_lock.user_uuid = other_user_uuid
        user_uuid = str(uuid.uuid4())[:8]

        usage_lock.release_lock(user_uuid)

        assert usage_lock.lock_engaged is True
        assert usage_lock.user_uuid is other_user_uuid


class TestContextManager:
    def test_init(self):
        manager = ContextManager()
        assert manager.usage_lock.lock_engaged is False
        assert manager.usage_lock.user_uuid is None
        assert len(manager.vectorstores) is 0

    def test_register_vectorstores(self):
        manager = ContextManager()
        mock_job_store = Mock(spec=Chroma)
        mock_resume_store = Mock(spec=Chroma)
        user_uuid = str(uuid.uuid4())[:8]

        manager.register_vectorstores(user_uuid, mock_job_store, mock_resume_store)

        assert manager.vectorstores[user_uuid]["job_vectorstore"] == mock_job_store
        assert (
            manager.vectorstores[user_uuid]["resume_vectorstore"] == mock_resume_store
        )

    def test_clear_context_empty_stores(self):
        manager = ContextManager()
        user_uuid = str(uuid.uuid4())[:8]

        manager.clear_context(user_uuid)  # Should not raise

        assert (user_uuid not in manager.vectorstores) is True

    def test_clear_context_with_stores(self):
        manager = ContextManager()
        user_uuid = str(uuid.uuid4())[:8]
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": ["1", "2"]}

        mock_job_store = Mock(spec=Chroma)
        mock_job_store._collection = mock_collection

        mock_resume_store = Mock(spec=Chroma)
        mock_resume_store._collection = mock_collection

        manager.register_vectorstores(user_uuid, mock_job_store, mock_resume_store)

        manager.clear_context(user_uuid)

        assert (user_uuid not in manager.vectorstores) is True
