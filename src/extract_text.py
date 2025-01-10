from datetime import datetime
import logging
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import Optional
import os
import docx
import pdf2image
import pytesseract
from pytesseract import Output
import shutil
import config
from logger import setup_logging

logger = setup_logging()


def extract_text_from_uploaded_files(
    job_file: UploadedFile,
    temp_dir: str,
) -> Optional[str]:
    """Extracts text from uploaded job files."""
    if not job_file:
        logger.error("No job file provided")
        return None, None

    start_time = datetime.now()

    job_text = extract_text("job", job_file, temp_dir)

    logger.info(f"Time spent extracting text: {datetime.now() - start_time}")

    return job_text


def get_file_extension(file: UploadedFile) -> Optional[str]:
    file_extension = file.name.split(".")[-1].lower()

    if file_extension not in config.app_config.SUPPORTED_FILE_TYPES:
        logger.error(f"Unsupported file type: {file_extension}")
        return None

    return file_extension


def extract_text(
    file_type: str,
    file: UploadedFile,
    temp_dir: str,
) -> Optional[str]:
    """Extracts text from a given file."""
    if not file:
        logger.error(f"File is missing")
        return

    if not get_file_extension(file):
        logger.error(f"Invalid file extension: {file.name}")
        return

    if file_type not in ["job", "resume"]:
        logger.error(f"Invalid file type: {file_type}")
        return

    logger.info(f"Extracting {file_type} text")
    file_path = os.path.join(temp_dir, os.path.basename(file.name))

    # This is so both test and actual runs work...
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


def extract_text_from_file(
    uploaded_file: UploadedFile,
    file_path: str,
) -> Optional[str]:
    """Extracts text from a given file."""
    file_extension = get_file_extension(uploaded_file)
    if not file_extension:
        return None

    text = ""

    try:
        if file_extension == "pdf":
            text = extract_text_from_image(file_path)
        elif file_extension in ["doc", "docx"]:
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
    except Exception as e:
        logger.error(f"An error occurred while extracting text: {e}")
        return None

    return text if text else None


# Doesn't work if using uploaded_file. Only with file_path
def extract_text_from_image(file_path: str) -> str:
    """Extracts text from a PDF image using OCR (Optical Character Recognition)."""

    text = ""
    try:
        images = pdf2image.convert_from_path(file_path)

        for image in images:
            ocr_dict = pytesseract.image_to_data(
                image, lang=config.app_config.OCR_LANG, output_type=Output.DICT
            )
            text += " ".join([word for word in ocr_dict["text"] if word])
    except Exception as e:
        logger.error(f"An error occurred while extracting text from image: {e}")
        return ""

    return text
