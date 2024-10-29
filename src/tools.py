"""
tools.py

This module provides various utility functions and tools for processing job descriptions and resumes. More specifically, it includes functions for analyzing the inputs, extracting text from files, and generating chat responses based on the analysis.
"""

import os
import traceback
import docx
import pdf2image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pytesseract import Output
from custom_embeddings import CustomEmbeddings
from transformers import AutoTokenizer
from prompts import (
    ANALYSIS_QUESTION,
    ANALYSIS_PROMPT,
    CHAT_QUESTION,
    CHAT_PROMPT,
    passthrough,
)
import logging
import shutil
import torch

OCR_LANG = "eng"
logger = logging.getLogger(__name__)
CONTEXT_WINDOW = 4096
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200


def verify_input_size(job_text, resume_text):
    """
    More accurate token counting using the actual tokenizer
    """

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
        1, (total_text_tokens + CHUNK_OVERLAP) // (CHUNK_SIZE - CHUNK_OVERLAP)
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


def analyze_inputs(job_text, resume_text, ollama):
    """
    Analyzes the inputs by processing the job text and resume text using the Mistral model.

    Args:
        job_text (str): The text of the job description file.
        resume_text (str): The text of the resume file.
        ollama: The Ollama model used for generating the analysis response.

    Returns:
        str: The response generated by the Mistral model based on the analysis of the resume and job description.
            If the analysis fails, it returns "Failed to process the files."
        job_retriever: The job retriever object.
        resume_retriever: The resume retriever object.
    """
    if job_text and resume_text:
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
            if hasattr(analyze_inputs, "_job_vectorstore"):
                del analyze_inputs._job_vectorstore
            if hasattr(analyze_inputs, "_resume_vectorstore"):
                del analyze_inputs._resume_vectorstore

            # Reuse existing embeddings instance if possible
            if not hasattr(analyze_inputs, "_embeddings"):
                analyze_inputs._embeddings = CustomEmbeddings(
                    model_name="./models/mistral"
                )

            analyze_inputs._job_vectorstore = process_text(
                job_text, analyze_inputs._embeddings
            )
            analyze_inputs._resume_vectorstore = process_text(
                resume_text, analyze_inputs._embeddings
            )

            logger.info(f"job_vectorstore: {analyze_inputs._job_vectorstore}")
            logger.info(f"resume_vectorstore: {analyze_inputs._resume_vectorstore}")

            if analyze_inputs._job_vectorstore and analyze_inputs._resume_vectorstore:
                logger.info("Both vectorstores exist")
                job_retriever = analyze_inputs._job_vectorstore.as_retriever()
                resume_retriever = analyze_inputs._resume_vectorstore.as_retriever()
                logger.info("After creating retrievers")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                chain = (
                    {
                        "resume_context": resume_retriever,
                        "job_context": job_retriever,
                        "question": lambda x: ANALYSIS_QUESTION,
                    }
                    | ANALYSIS_PROMPT
                    | ollama
                    | passthrough  # Simple output parsing
                )
                logger.info("After creating chain")
                response = chain.invoke(
                    "Analyze the resume based on the job description"
                )
                logger.info("After invoking chain to generate response")
                return response, job_retriever, resume_retriever

        except Exception as e:
            logger.error(f"Analysis failed: {e} | {traceback.format_exc()}")
            return "Failed: Unable to process the files.", None, None
    else:
        logger.error("Missing job text or resume text")
        return "Failed: Missing job text or resume text.", None, None


def extract_text(file_type, file, temp_dir):
    """
    Extracts text from a given file.

    Args:
        file_type (str): The type of file (either "job ad" or "resume").
        file (File): The uploaded file object.
        temp_dir (str): The temporary directory to save the uploaded file.

    Returns:
        str: The extracted text from the file, or None if no text is found.
    """

    if file_type not in ["job", "resume"]:
        logger.error(f"Invalid file type: {file_type}")
        return

    logger.info(f"Extracting {file_type} text")
    file_path = os.path.join(temp_dir, os.path.basename(file.name))
    # This is for testing purposes...
    file_test = open("tests/test.pdf", "r")
    if type(file) is type(file_test):
        shutil.copyfile(file.name, file_path)
    else:
        with open(file_path, "wb") as temp_file:
            temp_file.write(file.getbuffer())

    logger.info(f"Before extracting {file_type} text")
    text = extract_text_from_file(file, file_path)
    logger.info(f"After extracting {file_type} text")
    return text


def extract_text_from_file(uploaded_file, file_path):
    """
    Extracts text from a given file.

    Args:
        uploaded_file (File): The uploaded file object.
        file_path (str): The path to the uploaded file.

    Returns:
        str: The extracted text from the file, or None if no text is found.
    """
    file_extension = uploaded_file.name.split(".")[-1].lower()
    text = ""

    if file_extension == "pdf":
        text = extract_text_from_image(file_path)
    elif file_extension in ["doc", "docx"]:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

    return None if not text else text


# Doesn't work if using uploaded_file. Only with file_path
def extract_text_from_image(file_path):
    """
    Extracts text from a PDF image using OCR (Optical Character Recognition).

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF image.
    """

    text = ""
    images = pdf2image.convert_from_path(file_path)

    for image in images:
        ocr_dict = pytesseract.image_to_data(
            image, lang=OCR_LANG, output_type=Output.DICT
        )
        text += " ".join([word for word in ocr_dict["text"] if word])

    return text


def process_text(text, embeddings):
    """
    Process text by splitting it into chunks, and creating a vectorstore.

    Args:
        text (str): The text to be processed.
        embeddings: The embeddings object used for creating the vectorstore.

    Returns:
        vectorstore: The created vectorstore object.

    Raises:
        Exception: If an error occurs while creating the vectorstore.
    """
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
        token_count = len(embeddings.get_tokenizer(embeddings.model_name).encode(chunk))
        logger.info(f"Chunk {i} token count: {token_count}")

    if chunks and embeddings:
        embeddings_list = embeddings.embed_documents(chunks)
        logger.info("embeddings_list created")

        if embeddings_list:
            logger.info(f"embeddings: {embeddings}")
            try:
                vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
                logger.info(f"vectorstore: {vectorstore}")
                return vectorstore
            except Exception as e:
                logger.error("An error occurred while creating the vectorstore:")
                traceback.print_exception(type(e), e, e.__traceback__)
                return None
    return None


def get_chat_response(user_input, job_retriever, resume_retriever, ollama):
    """
    Retrieves a chat response based on the user input, resume retriever, and job ad retriever.

    Args:
        user_input (str): The user's input.
        resume_retriever: The resume retriever object.
        job_retriever: The job retriever object.
        ollama: The Ollama model used for generating the chat response.

    Returns:
        str: The chat response.
    """
    logger.info(f"user_input: {user_input}")
    logger.info(f"resume_retriever: {resume_retriever}")
    logger.info(f"job_retriever: {job_retriever}")
    if not user_input or not job_retriever or not resume_retriever:
        return None

    chain = (
        {
            "resume_context": resume_retriever,
            "job_context": job_retriever,
            "user_input": lambda x: user_input,
            "question": lambda x: CHAT_QUESTION,
        }
        | CHAT_PROMPT
        | ollama
        | passthrough  # Simple output parsing
    )

    response = chain.invoke(user_input)
    logger.info("Chat response created")
    return response
