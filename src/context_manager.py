from langchain_chroma import Chroma
from typing import Optional, Dict
import gc
import torch
import logging
import uuid
from threading import Lock

logger = logging.getLogger("root")


class ContextManager:
    """
    Thread-safe context manager that maintains separate contexts for each request.
    """

    def __init__(self):
        self._contexts: Dict[str, Dict] = {}
        self._lock = Lock()

    def create_request_context(self) -> str:
        """Create a new isolated context for a request"""
        with self._lock:
            request_id = str(uuid.uuid4())

            self._contexts[request_id] = {
                "job_vectorstore": None,
                "resume_vectorstore": None,
            }

            return request_id

    def register_vectorstores(
        self,
        request_id: str,
        job_store: Chroma,
        resume_store: Chroma,
    ) -> None:
        """Register current vectorstores for a specific request context"""
        with self._lock:
            if request_id in self._contexts:
                self._contexts[request_id]["job_vectorstore"] = job_store
                self._contexts[request_id]["resume_vectorstore"] = resume_store
            else:
                logger.error(f"No context found for request ID: {request_id}")

        logger.info(f"contexts: {self._contexts}")

    def clear_context(self, request_id: str) -> None:
        """Clear all stored context and force garbage collection for a specific request"""
        with self._lock:
            if request_id not in self._contexts:
                return

            try:
                context = self._contexts[request_id]

                # Clean up job vectorstore
                if context["job_vectorstore"] is not None:
                    collection = context["job_vectorstore"]._collection
                    if collection is not None:
                        ids = collection.get().get("ids", [])
                        if ids:
                            collection.delete(ids=ids)

                # Clean up resume vectorstore
                if context["resume_vectorstore"] is not None:
                    collection = context["resume_vectorstore"]._collection
                    if collection is not None:
                        ids = collection.get().get("ids", [])
                        if ids:
                            collection.delete(ids=ids)

                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error during context cleanup: {e}")

            finally:
                # Remove the context
                del self._contexts[request_id]
