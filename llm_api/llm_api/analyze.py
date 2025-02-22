import os
import traceback
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer
import logging
import uuid
from typing import Optional, Tuple, Union
from llm_api.custom_embeddings import CustomEmbeddings
from llm_api.prompts import (
    ANALYSIS_QUESTION,
    ANALYSIS_PROMPT,
    CHAT_QUESTION,
    CHAT_PROMPT,
    passthrough,
)
from llm_api.context_manager import ContextManager
from llm_api.logger import setup_logging

logger = setup_logging()
context_manager = ContextManager()

analyze_inputs_return_type = Union[Tuple[str, Chroma, Chroma], Tuple[str, None, None]]

# TODO: Centralize env var loading
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1024))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
CONTEXT_WINDOW = int(os.environ.get("CONTEXT_WINDOW", 8192))
TEMP_DIR = os.environ.get("TEMP_DIR", "/tmp/resume-experience")


def verify_input_size(
    job_text: str,
    resume_text: str,
) -> int:
    """More accurate token counting using the actual tokenizer"""

    # Get tokenizer instance
    if not hasattr(verify_input_size, "_tokenizer"):
        verify_input_size._tokenizer = AutoTokenizer.from_pretrained("./models/mistral")

    # Count actual tokens
    job_tokens = len(verify_input_size._tokenizer.encode(job_text))
    resume_tokens = len(verify_input_size._tokenizer.encode(resume_text))
    prompt_tokens = len(verify_input_size._tokenizer.encode(ANALYSIS_QUESTION))

    # Calculate total with chunking overhead
    total_text_tokens = job_tokens + resume_tokens
    num_chunks = max(
        1,
        (total_text_tokens + CHUNK_OVERLAP) // (CHUNK_SIZE - CHUNK_OVERLAP),
    )
    total_chunk_tokens = num_chunks * CHUNK_SIZE

    total_tokens = total_chunk_tokens + prompt_tokens

    logger.info(f"Job tokens: {job_tokens}")
    logger.info(f"Resume tokens: {resume_tokens}")
    logger.info(f"Prompt tokens: {prompt_tokens}")
    logger.info(f"Number of chunks: {num_chunks}")
    logger.info(f"Total chunk tokens: {total_chunk_tokens}")
    logger.info(f"Total estimated tokens: {total_tokens}")

    max_safe_tokens = CONTEXT_WINDOW - 200  # Leave some buffer for safety
    if total_tokens > max_safe_tokens:
        raise ValueError(
            f"Input would require approximately {total_tokens} tokens, "
            f"exceeding the {CONTEXT_WINDOW} token context window. "
            f"Please reduce the input size by about "
            f"{((total_tokens - max_safe_tokens) / total_tokens * 100):.1f}%"
        )

    return total_tokens


def create_chain(
    job_retriever,
    resume_retriever,
    ollama: ChatOllama,
    # question: ChatPromptTemplate,
    prompt: str,
    user_input=None,
) -> RunnablePassthrough:
    """Creates the chain for analyzing job and resume text."""
    context = {
        "resume_context": resume_retriever,
        "job_context": job_retriever,
        # "question": lambda x: question,
    }

    if user_input:
        context["user_input"] = lambda x: user_input

    return context | prompt | ollama | passthrough


def create_retrievers(
    job_vectorstore: Chroma,
    resume_vectorstore: Chroma,
) -> Tuple[VectorStoreRetriever, VectorStoreRetriever]:
    """Creates retrievers from vectorstores with consistent settings"""
    job_retriever = job_vectorstore.as_retriever(search_kwargs={"k": 2})
    resume_retriever = resume_vectorstore.as_retriever(search_kwargs={"k": 2})
    return job_retriever, resume_retriever


def analyze_inputs(
    job_text: str,
    resume_text: str,
    session_id: str,
    ollama: ChatOllama,
) -> analyze_inputs_return_type:
    """Analyzes the inputs by processing the job text and resume text using the Mistral model."""
    if not (job_text and resume_text):
        logger.error("Missing job text or resume text")
        output = "Failed: Missing job text or resume text.", None, None

    logger.info("Acquiring lock...")
    is_lock_acquired = context_manager.usage_lock.take_lock(session_id)

    if not is_lock_acquired:
        logger.error("Failed to acquire lock")
        return (
            "Failed to acquire lock. Please try again in a few minutes...",
            None,
            None,
        )

    logger.info("Lock acquired. Clearing context...")
    context_manager.clear_context(session_id)
    # TODO: Removed reset_state_analysis(st) from here. Reset retrievers here
    logger.info("After clear_context:")
    logger.info(f"context_manager.vectorstores: {context_manager.vectorstores}")

    output = None
    embeddings = None

    if output is None:
        try:
            total_tokens = verify_input_size(job_text, resume_text)
            logger.info(f"Total tokens (with overhead): {total_tokens}")
        except ValueError as e:
            logger.error(f"Input size verification failed: {e}")

            output = (
                "Failed: Input too large. Please reduce the size of the job ad or resume.",
                None,
                None,
            )

    if output is None:
        try:
            embeddings = CustomEmbeddings(model_name="./models/mistral")
            logger.info(f"Created embeddings instance")

            job_vectorstore = process_text(
                job_text,
                embeddings,
                collection_name=f"job_{session_id}",
            )
            logger.info(f"Created job vectorstore: {job_vectorstore}")
            logger.info(f"job_vectorstore: {job_vectorstore}")

            resume_vectorstore = process_text(
                resume_text,
                embeddings,
                collection_name=f"resume_{session_id}",
            )
            logger.info(f"Created resume vectorstore: {resume_vectorstore}")
            logger.info(f"resume_vectorstore: {resume_vectorstore}")

            if not (job_vectorstore and resume_vectorstore):
                logger.error("Failed to create vectorstores")
                output = "Failed: Unable to process the files.", None, None

            if output is None:
                logger.info("Both vectorstores exist")
                context_manager.register_vectorstores(
                    session_id, job_vectorstore, resume_vectorstore
                )
                logger.info(
                    f"context_manager.vectorstores: {context_manager.vectorstores}"
                )

                job_retriever, resume_retriever = create_retrievers(
                    job_vectorstore, resume_vectorstore
                )
                logger.info("After creating retrievers")

                chain = create_chain(
                    resume_retriever=resume_retriever,
                    job_retriever=job_retriever,
                    ollama=ollama,
                    prompt=ANALYSIS_PROMPT,
                    # question=ANALYSIS_QUESTION,
                )
                config = RunnableConfig(
                    callbacks=None,
                    configurable={
                        "stop": None,
                        "temperature": 0.3,
                    },
                )
                logger.info("After creating chain")

                response = chain.invoke(
                    "Analyze the resume based on the job description", config=config
                )
                logger.info("After invoking chain to generate response")
                output = response, job_retriever, resume_retriever

        except Exception as e:
            logger.error(f"Analysis failed: {e} | {traceback.format_exc()}")
            output = "Failed: Unable to process the files.", None, None

        finally:
            if "response" not in locals():
                context_manager.clear_context(session_id)

            if embeddings:
                embeddings.cleanup()

    logger.info("Releasing lock")
    context_manager.usage_lock.release_lock(session_id)

    return output


def process_text(
    text: str, embeddings: CustomEmbeddings, collection_name: str = None
) -> Optional[Chroma]:
    """Process text by splitting it into chunks, and creating a vectorstore."""
    if not text or not embeddings:
        logger.error("Missing text or embeddings")
        return None

    if collection_name is None:
        collection_name = str(uuid.uuid4())

    # Create a unique directory for this vectorstore
    persist_directory = os.path.join(TEMP_DIR, collection_name)
    os.makedirs(persist_directory, exist_ok=True)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda x: len(x.split()),
        separators=["\n\n", "\n", " ", ""],
    )
    logger.info(f"text_splitter: {text_splitter}")

    chunks = text_splitter.split_text(text)
    logger.info("chunks created from text_splitter")

    for i, chunk in enumerate(chunks):
        token_count = len(embeddings.get_tokenizer().encode(chunk))
        logger.info(f"Chunk {i} token count: {token_count}")

    if not (chunks and embeddings):
        logger.error("Missing chunks or embeddings")
        return None

    logger.info(f"embeddings: {embeddings}")
    try:
        logger.info(f"chunks: {chunks[0][:50]}")

        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

        logger.info(f"Created vectorstore for collection: {collection_name}")
        logger.info(f"vectorstore: {vectorstore}")
        return vectorstore
    except Exception as e:
        logger.error("An error occurred while creating the vectorstore:")
        traceback.print_exception(type(e), e, e.__traceback__)
        return None


def get_chat_response(
    user_input: str,
    job_retriever: VectorStoreRetriever,
    resume_retriever: VectorStoreRetriever,
    ollama: ChatOllama,
) -> Optional[str]:
    """Retrieves a chat response based on the user input, resume retriever, and job ad retriever."""
    logger.info("in get_chat_response")
    logger.info(f"user_input: {user_input}")
    logger.info(f"resume_retriever: {resume_retriever}")
    logger.info(f"job_retriever: {job_retriever}")
    if not user_input or not job_retriever or not resume_retriever:
        return None

    try:
        chain = create_chain(
            resume_retriever=resume_retriever,
            job_retriever=job_retriever,
            user_input=user_input,
            prompt=CHAT_PROMPT,
            ollama=ollama,
        )

        response = chain.invoke(user_input)
        logger.info("Chat response created")
        return response

    except Exception as e:
        logger.error(f"Error in chat response: {e}")
        raise

    finally:
        # Clean up chain resources
        if "chain" in locals():
            del chain

        # Force cleanup of embeddings after each chat response
        if hasattr(job_retriever, "vectorstore") and hasattr(
            job_retriever.vectorstore, "_embeddings"
        ):
            job_retriever.vectorstore._embeddings.cleanup()
        if hasattr(resume_retriever, "vectorstore") and hasattr(
            resume_retriever.vectorstore, "_embeddings"
        ):
            resume_retriever.vectorstore._embeddings.cleanup()
