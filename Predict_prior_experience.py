import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import MistralAIEmbeddings
import traceback
import os

# Simple passthrough function
def passthrough(input_data):
    return input_data

# Function to process each file and return a vectorstore
def process_file(file_path, embeddings):
    try:
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error processing file {file_path}: {e}")
        st.error(traceback.format_exc())
        return None

# Function to analyze inputs and return relevant information
def analyze_inputs(uploaded_files, job_ad_text, resume_text):
    try:
        if uploaded_files:
            embeddings = MistralAIEmbeddings()  # Use MistralAIEmbeddings without API key
            job_ad_vectorstore = None
            resume_vectorstore = None

            for uploaded_file in uploaded_files:
                # Save the uploaded file to a temporary location
                temp_file_path = os.path.join("temp_dir", uploaded_file.name)
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.getbuffer())

                if "Job" in uploaded_file.name:
                    job_ad_vectorstore = process_file(temp_file_path, embeddings)
                else:
                    resume_vectorstore = process_file(temp_file_path, embeddings)

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
        st.error(f"Error analyzing inputs: {e}")
        st.error(traceback.format_exc())
        return "Failed to analyze inputs."

# Streamlit UI components
def main():
    st.title("Intelligent Resume Matcher: Analyze Relevant Experience")

    st.write("This app will compare a job description to a resume and extract the number of years of relevant work experience from the resume.")

    uploaded_files = st.file_uploader("Upload Job Descriptions and Resumes", type=["pdf"], accept_multiple_files=True)
    job_ad_text = st.text_area("Job Description")
    resume_text = st.text_area("Resume Text")

    if st.button("Analyze"):
        with st.spinner("Processing..."):
            result = analyze_inputs(uploaded_files, job_ad_text, resume_text)
            st.write(result)

# Define the query prompt
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate three different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Define the RAG prompt
PROMPT = ChatPromptTemplate.from_template(
    template="""Answer the question based ONLY on the following context:
    {context}
    Question: {question}"""
)

if __name__ == "__main__":
    # Create a temporary directory to save uploaded files
    os.makedirs("temp_dir", exist_ok=True)
    main()
    # Clean up the temporary directory after the app closes
    import shutil
    shutil.rmtree("temp_dir")
