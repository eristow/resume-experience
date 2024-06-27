import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit.components.v1 as components

# Create necessary directories if they don't exist
if not os.path.exists('files'):
    os.mkdir('files')
if not os.path.exists('jj'):
    os.mkdir('jj')

# Initialize session state
if 'template' not in st.session_state:
    st.session_state.template = """Based on the job description [Job Description] and the provided resume [Resume], extract the total years of experience and the number of years of relevant experience from the resume. Provide your answer in the following format: 
    'The candidate has X years of total experience and Y years of relevant experience for this role.' 
    Replace X with the total number of years of experience and Y with the number of years of relevant experience."""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(persist_directory='jj',
                                          embedding_function=OllamaEmbeddings(base_url='http://localhost:11434',
                                                                              model="mistral")
                                          )

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(base_url="http://localhost:11434",
                                  model="mistral",
                                  verbose=True,
                                  callback_manager=CallbackManager(
                                      [StreamingStdOutCallbackHandler()]),
                                  )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Define file processing function
def process_file(uploaded_file, file_path):
    if not os.path.isfile(file_path):
        with st.spinner("Analyzing your document..."):
            bytes_data = uploaded_file.read()
            with open(file_path, "wb") as f:
                f.write(bytes_data)
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                return None

            data = loader.load()

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )
            all_splits = text_splitter.split_documents(data)

            # Create and persist the vector store
            vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=OllamaEmbeddings(model="mistral")
            )
            vectorstore.persist()
            return vectorstore
    return None

# Define text processing function
def process_text(input_text):
    data = [{"text": input_text}]

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    all_splits = text_splitter.split_documents(data)

    # Create the vector store
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=OllamaEmbeddings(model="mistral")
    )
    return vectorstore

# Define analysis function
def analyze_inputs(uploaded_files, job_ad_text, resume_text):
    job_ad_vectorstore = None
    resume_vectorstore = None

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = f"files/{uploaded_file.name}"
            if "job" in uploaded_file.name.lower():
                job_ad_vectorstore = process_file(uploaded_file, file_path)
            elif "resume" in uploaded_file.name.lower():
                resume_vectorstore = process_file(uploaded_file, file_path)

    if job_ad_text:
        job_ad_vectorstore = process_text(job_ad_text)
    
    if resume_text:
        resume_vectorstore = process_text(resume_text)
    
    if job_ad_vectorstore and resume_vectorstore:
        st.session_state.job_ad_retriever = job_ad_vectorstore.as_retriever()
        st.session_state.resume_retriever = resume_vectorstore.as_retriever()
        
        # Initialize QA chains for both job ads and resumes
        if 'job_ad_qa_chain' not in st.session_state:
            st.session_state.job_ad_qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type='stuff',
                retriever=st.session_state.job_ad_retriever,
                verbose=True,
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": st.session_state.prompt,
                    "memory": st.session_state.memory,
                }
            )
        
        if 'resume_qa_chain' not in st.session_state:
            st.session_state.resume_qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type='stuff',
                retriever=st.session_state.resume_retriever,
                verbose=True,
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": st.session_state.prompt,
                    "memory": st.session_state.memory,
                }
            )

# Define the Streamlit UI and logic
st.title("Intelligent Resume Matcher: Analyze Relevant Experience")
st.write("### Introduction")
st.write("This app will compare a job description to a resume and extract the number of years of relevant work experience from the resume.")

tab1, tab2 = st.tabs(["Upload Files", "Text Input"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        job_ad_files = st.file_uploader("Upload Job Descriptions", type=['pdf', 'docx'], accept_multiple_files=True, key="job_ad_files")

    with col2:
        resume_files = st.file_uploader("Upload Resumes", type=['pdf', 'docx'], accept_multiple_files=True, key="resume_files")

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        job_ad_text = st.text_area("Job Description Text", height=200)

    with col2:
        resume_text = st.text_area("Resume Text", height=200)

# Collect uploaded files from both uploaders
uploaded_files = (job_ad_files if job_ad_files else []) + (resume_files if resume_files else [])

# Process files or text inputs if provided
if uploaded_files or job_ad_text or resume_text:
    analyze_inputs(uploaded_files, job_ad_text, resume_text)

st.write("### Relevant Experience")
if 'job_ad_qa_chain' in st.session_state and 'resume_qa_chain' in st.session_state:
    job_ad_response = st.session_state.job_ad_qa_chain("Extract the job description")
    resume_response = st.session_state.resume_qa_chain("Extract the resume details")
    
    job_ad_message = job_ad_response['result']
    resume_message = resume_response['result']
    
    combined_input = f"Based on the job description: {job_ad_message} and the provided resume: {resume_message}, extract the total years of experience and the number of years of relevant experience from the resume. Provide your answer in the following format: 'The candidate has X years of total experience and Y years of relevant experience for this role.' Replace X with the total number of years of experience and Y with the number of years of relevant experience."
    combined_response = st.session_state.llm(combined_input)
    
    st.write(f"### Relevant Experience\n{combined_response}")
else:
    st.write("Generating as soon as Resume and Job Description are filled.")

# Chat interface for additional questions
st.write("### Questions")
components.html("""
<div style="background-color:#DAFFDA; padding:10px; border-radius:5px;">
    <p>Sub, I’d be happy to answer any further questions about this resume's prior experience!</p>
    <textarea id="chat-input" style="width:100%; height:100px;"></textarea>
    <button style="margin-top:10px;">Submit</button>
</div>
""", height=200)
