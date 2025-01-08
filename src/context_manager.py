from langchain_chroma import Chroma
from typing import Optional
import gc
import torch
from logger import setup_logging

logger = setup_logging()


class ContextManager:
    """
    Manages context and state for the analysis and chat processes.
    Note: This class is focused on cleanup rather than storage/reuse of resources.
    """

    def __init__(self):
        self._current_job_vectorstore: Optional[Chroma] = None
        self._current_resume_vectorstore: Optional[Chroma] = None

    def register_vectorstores(
        self,
        job_store: Chroma,
        resume_store: Chroma,
    ) -> None:
        """Register current vectorstores for cleanup"""
        self._current_job_vectorstore = job_store
        self._current_resume_vectorstore = resume_store

    def clear_context(self) -> None:
        """Clear all stored context and force garbage collection"""
        try:
            # Clean up job vectorstore
            if self._current_job_vectorstore is not None:
                collection = self._current_job_vectorstore._collection
                if collection is not None:
                    ids = collection.get().get("ids", [])
                    if ids:
                        collection.delete(ids=ids)
                self._current_job_vectorstore = None

            # Clean up resume vectorstore
            if self._current_resume_vectorstore is not None:
                collection = self._current_resume_vectorstore._collection
                if ids:
                    collection.delete(ids=ids)
                self._current_resume_vectorstore = None

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error during context cleanup: {e}")

        finally:
            # Ensure these are set to None even if cleanup fails
            self._current_job_vectorstore = None
            self._current_resume_vectorstore = None
