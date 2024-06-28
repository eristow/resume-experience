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
import os
import time

# Create necessary directories if they don't exist
if not os.path.exists('files'):
    os.mkdir('files')
if not os.path.exists('jj'):
    os.mkdir('jj')

# Initialize session state variables
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

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Set up the layout
st.title("Intelligent Resume Matcher: Analyze Relevant Experience")
st.write("This app will compare a job description to a resume and extract the number of years of relevant work experience from the resume.")

tab1, tab2 = st.tabs(["Upload Files", "Text Input"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        job_ad_files = st.file_uploader("Upload Job Descriptions", type=['pdf', 'docx'], accept_multiple_files=True, key="job_ad_file")
    with col2:
        resume_files = st.file_uploader("Upload Resumes", type=['pdf', 'docx'], accept_multiple_files=True, key="resume_file")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        job_ad_text = st.text_area("Job Description Text", height=300)
    with col2:
        resume_text = st.text_area("Resume Text", height=300)

st.header("Relevant Experience")
st.write("Generating as soon as Resume and Job Description are filled.")

st.header("Questions")
question = st.text_area("Ask any questions you have about the app or the results", key="user_input")
submit_button = st.button("Submit", key="submit_button")

def process_file(uploaded_file, file_path):
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(file_path)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = Docx2txtLoader(file_path)
    else:
        st.error("Unsupported file type.")
        return None
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    vectorstore = Chroma(persist_directory="jj", embedding_function=OllamaEmbeddings(base_url="http://localhost:11434", model="mistral"))
    vectorstore.add_texts([doc.page_content for doc in texts])
    return vectorstore

def analyze_inputs(uploaded_files, job_ad_text, resume_text):
    job_ad_vectorstore = None
    resume_vectorstore = None
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("files", uploaded_file.name)
            if "job" in uploaded_file.name.lower():
                job_ad_vectorstore = process_file(uploaded_file, file_path)
            elif "resume" in uploaded_file.name.lower():
                resume_vectorstore = process_file(uploaded_file, file_path)
    if job_ad_text:
        job_ad_vectorstore = process_file(job_ad_text, "job_ad_text.txt")
    if resume_text:
        resume_vectorstore = process_file(resume_text, "resume_text.txt")
    return job_ad_vectorstore, resume_vectorstore

if job_ad_files or resume_files or job_ad_text or resume_text:
    st.write("Processing...")
    job_ad_vectorstore, resume_vectorstore = analyze_inputs(job_ad_files + resume_files, job_ad_text, resume_text)
    if job_ad_vectorstore and resume_vectorstore:
        qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type="stuff",
            retriever=resume_vectorstore.as_retriever()
        )
        question = "Based on the job description and resume, how many years of relevant experience does the candidate have?"
        result = qa_chain.run(input_documents=[job_ad_vectorstore, resume_vectorstore], question=question)
        st.write(result)

# Handle the question input and generate a response
if submit_button and question:
    st.session_state.chat_history.append({"role": "user", "message": question})
    with st.spinner("Assistant is typing..."):
        response = st.session_state.qa_chain(question)
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response['result'].split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    chatbot_message = {"role": "assistant", "message": response['result']}
    st.session_state.chat_history.append(chatbot_message)

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f'**User:** {message["message"]}')
    else:
        st.markdown(f'**Assistant:** {message["message"]}')
