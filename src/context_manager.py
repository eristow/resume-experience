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

    def _cleanup_embeddings(self, vectorstore: Optional[Chroma]) -> None:
        """Helper method to clean up embeddings from a vectorstore"""
        if vectorstore is not None:
            try:
                if hasattr(vectorstore, "_embeddings"):
                    embeddings = vectorstore._embeddings

                    if hasattr(embeddings, "cleanup"):
                        logger.info("Cleaning up embeddings model...")
                        embeddings.cleanup()
                    elif hasattr(embeddings, "_model"):
                        if embeddings._model is not None:
                            logger.info("Performing fallback embeddings cleanup...")
                            embeddings._model.cpu()
                            del embeddings._model
                            embeddings._model = None

                        if hasattr(embeddings, "_tokenizer"):
                            del embeddings._tokenizer
                            embeddings._tokenizer = None

            except Exception as e:
                logger.error(f"Error during embeddings cleanup: {e}")

    def _cleanup_vectorstore(
        self, vectorstore: Optional[Chroma], store_name: str, session_id: uuid
    ) -> None:
        """Helper method to clean up a vectorstore and its resources"""

        if vectorstore is not None:
            try:
                logger.info(f"Cleaning up {store_name}...")

                self._cleanup_embeddings(vectorstore)

                ids = vectorstore.get().get("ids", [])
                if ids:
                    vectorstore.delete(ids=ids)

                try:
                    vectorstore.delete_collection()
                except Exception as e:
                    logger.error(f"Error deleting collection: {e}")

                self.vectorstores[session_id][store_name] = None

            except Exception as e:
                logger.error(f"Error during {store_name} cleanup: {e}")

    def clear_context(self, session_id: uuid) -> None:
        """Clear all stored context and force garbage collection"""
        logger.info("Clearing context_manager context...")

        try:
            if session_id not in self.vectorstores:
                logger.info(
                    "session_id not in context_manager vectorstores. Returning..."
                )
                return

            self._cleanup_vectorstore(
                self.vectorstores[session_id].get("job_vectorstore"),
                "job_vectorstore",
                session_id,
            )
            self._cleanup_vectorstore(
                self.vectorstores[session_id].get("resume_vectorstore"),
                "resume_vectorstore",
                session_id,
            )

            del self.vectorstores[session_id]

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache...")
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error during context cleanup: {e}")

            # Ensure the session is removed even if cleanup fails
            if session_id in self.vectorstores:
                del self.vectorstores[session_id]
