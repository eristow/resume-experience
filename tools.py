import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.embeddings.base import Embeddings
from PyPDF2 import PdfReader
import docx
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chromadb import Client
from chromadb.config import Settings
import pdf2image
import pytesseract
from pytesseract import Output, TesseractError
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.instructor import InstructorEmbedding

OCR_LANG = "eng"

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, quantization_config=quant_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def embed_documents(self, documents):
        embeddings = []
        for doc in documents:
            tokens = self.tokenizer(doc, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                outputs = self.model(**tokens)

            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states
            elif hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                raise AttributeError("The model's output does not have 'hidden_states' or 'last_hidden_state'.")

            embedding = hidden_states[-1].mean(dim=1).cpu().numpy().flatten().tolist()
            embeddings.append(embedding)

        return embeddings

    def embed_query(self, query):
        tokens = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**tokens)

        if hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            raise AttributeError("The model's output does not have 'hidden_states' or 'last_hidden_state'.")

        embedding = hidden_states[-1].mean(dim=1).cpu().numpy().flatten().tolist()
        return embedding


def extract_text_from_file(uploaded_file, file_path):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    text = ""

    if file_extension == "pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif file_extension in ["doc", "docx"]:
        doc = docx.Document(uploaded_file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    
    # If text is empty, try extracting using image and OCR
    if text == "" or text.isspace():
        text = extract_text_from_image(file_path)

    return None if not text else text

# ENSURE POPPLER AND TESSERACT ARE INSTALLED ON LOCAL MACHINE
#   - poppler: https://poppler.freedesktop.org/
#   - tesseract: https://tesseract-ocr.github.io/tessdoc/Installation.html 
# Doesn't work if using uploaded_file. Only with file_path
def extract_text_from_image(file_path):
    text = ""
    images = pdf2image.convert_from_path(file_path)
    
    for image in images:
        ocr_dict = pytesseract.image_to_data(image, lang=OCR_LANG, output_type=Output.DICT)
        text += " ".join([word for word in ocr_dict["text"] if word])

    return text

def process_file(file_path, embeddings):
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            with open("temp_text.txt", "w") as text_file:
                text_file.write(text)
            loader = TextLoader("temp_text.txt")
            data = loader.load()
            # TODO: this isn't working for job_ad_vectorstore
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            print(f"chunks: {chunks}")
            if chunks:
                embeddings_list = embeddings.embed_documents([chunk.page_content for chunk in chunks])
                # print(f"embeddings_list: {embeddings_list}")

                if embeddings_list:
                    print(f"embeddings: {embeddings}")
                    try:
                        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
                        print(f"vectorstore: {vectorstore}")
                        return vectorstore
                    except Exception as e:
                        print("An error occurred while creating the vectorstore:")
                        print(traceback.format_exc())
                        return None
    return None