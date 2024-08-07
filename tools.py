import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.embeddings.base import Embeddings
from PyPDF2 import PdfReader
import docx
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def embed_documents(self, documents):
        embeddings = []
        for doc in documents:
            tokens = self.tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**tokens)
            embeddings.append(outputs.logits.mean(dim=1).cpu().numpy())
        return embeddings

    def embed_query(self, query):
        tokens = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.logits.mean(dim=1).cpu().numpy()
    
def pdfimage_text_extract_from_file(uploaded_file):
    ### my install tesseract
    # pip install pytesseract
    # sudo apt-get install tesseract-ocr
    import pytesseract
    from pdf2image import convert_from_path
    
    pages = convert_from_path(uploaded_file, 500)

    for pageNum,imgBlob in enumerate(pages):
        text = pytesseract.image_to_string(imgBlob,lang='eng')
            
    return text

# def extract_text_from_file(uploaded_file):
#     file_extension = uploaded_file.name.split(".")[-1].lower()
#     text = ""

#     if file_extension == "pdf":
#         reader = PdfReader(uploaded_file)
#         for page in reader.pages:
#             text += page.extract_text() + "\n"
#     elif file_extension in ["doc", "docx"]:
#         doc = docx.Document(uploaded_file)
#         for paragraph in doc.paragraphs:
#             text += paragraph.text + "\n"

#     return text

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
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            if chunks:
                embeddings_list = embeddings.embed_documents([chunk.page_content for chunk in chunks])
                if embeddings_list:
                    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
                    return vectorstore
    return None