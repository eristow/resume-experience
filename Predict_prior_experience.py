import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import SimpleOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.runnables import RunnablePassthrough
from langchain_mistralai import MistralAIEmbeddings

# Function to process each file and return a vectorstore
def process_file(file, embeddings):
    loader = PyPDFLoader(file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore

# Function to analyze inputs and return relevant information
def analyze_inputs(uploaded_files, job_ad_text, resume_text):
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
                {"context": retriever, "question": RunnablePassthrough()}
                | PROMPT
                | ChatOllama()
                | SimpleOutputParser()
            )
            response = chain.invoke("Analyze the resume based on the job description")
            return response

    return "Failed to process the files."

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
    main()
