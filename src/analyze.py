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
from typing import Optional, Tuple, Union
from custom_embeddings import CustomEmbeddings
import config
from prompts import (
    ANALYSIS_QUESTION,
    ANALYSIS_PROMPT,
    CHAT_QUESTION,
    CHAT_PROMPT,
    passthrough,
)
from context_manager import ContextManager
from logger import setup_logging

logger = setup_logging()
context_manager = ContextManager()

analyze_inputs_return_type = Union[Tuple[str, Chroma, Chroma], Tuple[str, None, None]]


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
        (total_text_tokens + config.app_config.CHUNK_OVERLAP)
        // (config.app_config.CHUNK_SIZE - config.app_config.CHUNK_OVERLAP),
    )
    total_chunk_tokens = num_chunks * config.app_config.CHUNK_SIZE

    total_tokens = total_chunk_tokens + prompt_tokens

    logger.info(f"Job tokens: {job_tokens}")
    logger.info(f"Resume tokens: {resume_tokens}")
    logger.info(f"Prompt tokens: {prompt_tokens}")
    logger.info(f"Number of chunks: {num_chunks}")
    logger.info(f"Total chunk tokens: {total_chunk_tokens}")
    logger.info(f"Total estimated tokens: {total_tokens}")

    max_safe_tokens = (
        config.app_config.CONTEXT_WINDOW - 200
    )  # Leave some buffer for safety
    if total_tokens > max_safe_tokens:
        raise ValueError(
            f"Input would require approximately {total_tokens} tokens, "
            f"exceeding the {config.app_config.CONTEXT_WINDOW} token context window. "
            f"Please reduce the input size by about "
            f"{((total_tokens - max_safe_tokens) / total_tokens * 100):.1f}%"
        )

    return total_tokens


def create_chain(
    job_retriever,
    resume_retriever,
    ollama: ChatOllama,
    question: ChatPromptTemplate,
    prompt: str,
    user_input=None,
) -> RunnablePassthrough:
    """Creates the chain for analyzing job and resume text."""
    context = {
        "resume_context": resume_retriever,
        "job_context": job_retriever,
        "question": lambda x: question,
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
    ollama: ChatOllama,
) -> analyze_inputs_return_type:
    """Analyzes the inputs by processing the job text and resume text using the Mistral model."""
    context_manager.clear_context()

    if not (job_text and resume_text):
        logger.error("Missing job text or resume text")
        return "Failed: Missing job text or resume text.", None, None

    try:
        total_tokens = verify_input_size(job_text, resume_text)
        logger.info(f"Total tokens (with overhead): {total_tokens}")
    except ValueError as e:
        logger.error(f"Input size verification failed: {e}")
        return (
            "Failed: Input too large. Please reduce the size of the job description or resume.",
            None,
            None,
        )

    try:
        embeddings = CustomEmbeddings(model_name="./models/mistral")

        job_vectorstore = process_text(job_text, embeddings)
        resume_vectorstore = process_text(resume_text, embeddings)

        logger.info(f"job_vectorstore: {job_vectorstore}")
        logger.info(f"resume_vectorstore: {resume_vectorstore}")

        if not (job_vectorstore and resume_vectorstore):
            logger.error("Failed to create vectorstores")
            return "Failed: Unable to process the files.", None, None

        logger.info("Both vectorstores exist")
        context_manager.register_vectorstores(job_vectorstore, resume_vectorstore)

        job_retriever, resume_retriever = create_retrievers(
            job_vectorstore, resume_vectorstore
        )
        logger.info("After creating retrievers")

        chain = create_chain(
            resume_retriever=resume_retriever,
            job_retriever=job_retriever,
            ollama=ollama,
            prompt=ANALYSIS_PROMPT,
            question=ANALYSIS_QUESTION,
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
        return response, job_retriever, resume_retriever

    except Exception as e:
        logger.error(f"Analysis failed: {e} | {traceback.format_exc()}")
        return "Failed: Unable to process the files.", None, None
    finally:
        if "response" not in locals():
            context_manager.clear_context()


def process_text(
    text: str,
    embeddings: CustomEmbeddings,
) -> Optional[Chroma]:
    """Process text by splitting it into chunks, and creating a vectorstore."""
    if not text or not embeddings:
        logger.error("Missing text or embeddings")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.app_config.CHUNK_SIZE,
        chunk_overlap=config.app_config.CHUNK_OVERLAP,
        length_function=lambda x: len(x.split()),
        separators=["\n\n", "\n", " ", ""],
    )
    logger.info(f"text_splitter: {text_splitter}")

    chunks = text_splitter.split_text(text)
    logger.info("chunks created from text_splitter")
    for i, chunk in enumerate(chunks):
        token_count = len(embeddings.get_tokenizer(embeddings.model_name).encode(chunk))
        logger.info(f"Chunk {i} token count: {token_count}")

    if not (chunks and embeddings):
        logger.error("Missing chunks or embeddings")
        return None

    embeddings_list = embeddings.embed_documents(chunks)
    logger.info("embeddings_list created")

    if not embeddings_list:
        logger.error("Missing embeddings_list")
        return None

    logger.info(f"embeddings: {embeddings}")
    try:
        vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
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

    chain = create_chain(
        resume_retriever=resume_retriever,
        job_retriever=job_retriever,
        user_input=user_input,
        prompt=CHAT_PROMPT,
        question=CHAT_QUESTION,
        ollama=ollama,
    )

    response = chain.invoke(user_input)
    logger.info("Chat response created")
    return response
