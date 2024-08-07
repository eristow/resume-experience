import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.embeddings.base import Embeddings
from PyPDF2 import PdfReader
import docx
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.retrievers import BM25Retriever
# from tools import CustomEmbeddings

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.embeddings.base import Embeddings
from PyPDF2 import PdfReader
import docx
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate

import pytesseract
from pdf2image import convert_from_path


# '/home/james/models/BAAI/bge-base-en-v1.5'
BASE_MODEL = '/home/james/models/mistralai/Mistral-7B-Instruct-v0.3'

# Define the query prompt
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Based on the [Job Description] and the provided [Resume], extract the number of years of relevant experience from the resume. Provide your answer in the following format: 'The candidate has X years of relevant experience for this role.' Replace X with the actual number of years. The output should be in years to the closest 0.5 year."""
)

# Define the RAG prompt
PROMPT = ChatPromptTemplate.from_template(
    template="""Answer the question based ONLY on the following context:
    {context}
    Question: {question}"""
)

# Simple passthrough function
def passthrough(input_data):
    return input_data

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
    # import pytesseract
    # from pdf2image import convert_from_path
    
    pages = convert_from_path(uploaded_file, 500)

    for pageNum,imgBlob in enumerate(pages):
        text = pytesseract.image_to_string(imgBlob,lang='eng')
            
    return text

def extract_text_from_file(uploaded_file):
    file_extension = uploaded_file.split(".")[-1].lower()
    text = ""

    if "pdf" in file_extension:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        if len(text) <= 1:
            text = pdfimage_text_extract_from_file(uploaded_file)
            
    elif file_extension in ["doc", "docx"]:
        doc = docx.Document(uploaded_file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

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
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            if chunks:
                embeddings_list = embeddings.embed_documents([chunk.page_content for chunk in chunks])
                if embeddings_list:
                    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
                    return vectorstore
    return None


    
    
# Function to analyze inputs and return relevant information
def analyze_inputs(uploaded_files, job_ad_text, resume_text):
    try:
        if uploaded_files:
            embeddings = MistralAIEmbeddings()  # Use MistralAIEmbeddings without API key
            job_ad_vectorstore = None
            resume_vectorstore = None

            for file in uploaded_files:
                if "Job" in file.name:
                    job_ad_vectorstore = process_file(file, embeddings)
                else:
                    resume_vectorstore = process_file(file, embeddings)

            if job_ad_vectorstore and resume_vectorstore:
                retriever = MultiQueryRetriever.from_llm(
                    vector_db=job_ad_vectorstore.as_retriever(),
                    llm=ChatOllama(),
                    prompt=QUERY_PROMPT,
                )
                chain = (
                    {"context": retriever, "question": passthrough}
                    | PROMPT
                    | ChatOllama()
                    | passthrough  # Simple output parsing
                )
                response = chain.invoke("Analyze the resume based on the job description")
                return response

        return "Failed to process the files."
    except Exception as e:

        return "Failed to analyze inputs."