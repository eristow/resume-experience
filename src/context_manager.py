from langchain_chroma import Chroma
from typing import Optional
import gc
import torch
import uuid
from logger import setup_logging

logger = setup_logging()


class UsageLock:
    """
    Simple lock class for ContextManager
    """

    lock_engaged = False
    user_uuid = None

    def __init__(self):
        pass

    def take_lock(self, user_uuid: uuid) -> bool:
        logger.info("take_lock:")
        logger.info(f"lock_engaged: {self.lock_engaged}")
        logger.info(f"user_uuid: {self.user_uuid}")

        if self.lock_engaged is True:
            return False

        self.lock_engaged = True
        self.user_uuid = user_uuid

        logger.info("after:")
        logger.info(f"lock_engaged: {self.lock_engaged}")
        logger.info(f"user_uuid: {self.user_uuid}")

        return True

    def release_lock(self, user_uuid: uuid) -> bool:
        logger.info("release_lock:")
        logger.info(f"lock_engaged: {self.lock_engaged}")
        logger.info(f"user_uuid: {self.user_uuid}")

        if self.user_uuid is not user_uuid:
            return False

        self.lock_engaged = False
        self.user_uuid = None

        logger.info("after:")
        logger.info(f"lock_engaged: {self.lock_engaged}")
        logger.info(f"user_uuid: {self.user_uuid}")

        return True


class ContextManager:
    """
    Manages context and state for the analysis and chat processes.
    Note: This class is focused on cleanup rather than storage/reuse of resources.
    """

    usage_lock = UsageLock()
    vectorstores = {}

    def __init__(self):
        pass

    def register_vectorstores(
        self,
        session_id: uuid,
        job_store: Chroma,
        resume_store: Chroma,
    ) -> None:
        """Register current vectorstores for cleanup"""
        if session_id not in self.vectorstores:
            self.vectorstores[session_id] = {}

        self.vectorstores[session_id]["job_vectorstore"] = job_store
        self.vectorstores[session_id]["resume_vectorstore"] = resume_store

    def clear_context(self, session_id: uuid) -> None:
        """Clear all stored context and force garbage collection"""
        try:
            # Check for existing vectorstore for session_id
            if session_id not in self.vectorstores:
                return

            # Clean up job vectorstore
            if self.vectorstores[session_id]["job_vectorstore"] is not None:
                collection = self.vectorstores[session_id]["job_vectorstore"]
                if collection is not None:
                    ids = collection.get().get("ids", [])
                    if ids:
                        collection.delete(ids=ids)
                self.vectorstores[session_id]["job_vectorstore"] = None

            # Clean up resume vectorstore
            if self.vectorstores[session_id]["resume_vectorstore"] is not None:
                collection = self.vectorstores[session_id]["resume_vectorstore"]
                if collection is not None:
                    ids = collection.get().get("ids", [])
                    if ids:
                        collection.delete(ids=ids)
                self.vectorstores[session_id]["resume_vectorstore"] = None

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error during context cleanup: {e}")

        finally:
            # Ensure these are set to None even if cleanup fails
            if session_id not in self.vectorstores:
                return

            if (
                "job_vectorstore" not in self.vectorstores[session_id]
                or "resume_vectorstore" not in self.vectorstores[session_id]
            ):
                return

            self.vectorstores[session_id]["job_vectorstore"] = None
            self.vectorstores[session_id]["resume_vectorstore"] = None
