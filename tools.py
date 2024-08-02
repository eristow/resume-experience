import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

class CustomEmbeddings:
    def __init__(self, model_name):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Add this line to set padding token

    def embed_documents(self, documents):
        embeddings = []
        for doc in documents:
            tokens = self.tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**tokens)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist())  # Convert to list
        return embeddings

    def embed_query(self, query):
        tokens = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()  # Convert to list

def process_file(file_path, embeddings):
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            with open("temp_text.txt", "w") as text_file:
                text_file.write(text)
            
            data = TextLoader("temp_text.txt").load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            if chunks:  # Check if chunks is not empty
                vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
                return vectorstore
    return None

def extract_text_from_file(uploaded_file):
    if uploaded_file:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return uploaded_file.name
    return None