import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import VectorDBQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings

def process_file(uploaded_file, file_path):
    try:
        # Load the document based on its file type
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(file_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(file_path)
        else:
            st.error("Unsupported file type.")
            return None

        texts = loader.load()

        if not texts:
            st.error("No content extracted from the document.")
            return None

        # Use MistralAIEmbeddings to generate embeddings
        embedding = MistralAIEmbeddings(api_key="your-api-key")
        vectorstore = Chroma.from_texts([doc.page_content for doc in texts], embedding)
        return vectorstore
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        return None

def analyze_inputs(job_ad_files, resume_files, job_ad_text, resume_text):
    job_ad_vectorstore = None
    resume_vectorstore = None

    for uploaded_file in job_ad_files:
        file_path = os.path.join(st.session_state.upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        job_ad_vectorstore = process_file(uploaded_file, file_path)

    for uploaded_file in resume_files:
        file_path = os.path.join(st.session_state.upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        resume_vectorstore = process_file(uploaded_file, file_path)

    return job_ad_vectorstore, resume_vectorstore

def main():
    st.title("Intelligent Resume Matcher: Analyze Relevant Experience")
    st.write("This app will compare a job description to a resume and extract the number of years of relevant work experience from the resume.")

    if "upload_dir" not in st.session_state:
        st.session_state.upload_dir = "uploads"
        if not os.path.exists(st.session_state.upload_dir):
            os.makedirs(st.session_state.upload_dir)

    # Upload Job Descriptions
    st.header("Job Description")
    job_ad_files = st.file_uploader("Upload Job Descriptions", accept_multiple_files=True, type=["pdf", "docx"], key="job_ads")

    # Upload Resumes
    st.header("Resume")
    resume_files = st.file_uploader("Upload Resumes", accept_multiple_files=True, type=["pdf", "docx"], key="resumes")

    st.header("Relevant Experience")
    st.write("Generating as soon as Resume and Job Description are filled.")

    # Job Description Text Area
    st.header("Job Description")
    job_ad_text = st.text_area("Enter the job description here...")

    # Resume Text Area
    st.header("Resume Text")
    resume_text = st.text_area("Enter the resume text here...")

    # Analyze Button
    if st.button("Analyze"):
        if (job_ad_files or job_ad_text) and (resume_files or resume_text):
            job_ad_vectorstore, resume_vectorstore = analyze_inputs(job_ad_files if job_ad_files else [], resume_files if resume_files else [], job_ad_text, resume_text)

            if job_ad_vectorstore and resume_vectorstore:
                st.success("Files processed successfully. Generating analysis...")

                template = "Based on the job description [Job Description] and the provided resume [Resume], extract the total years of experience and the number of years of relevant experience from the resume. Provide your answer in the following format: 'The candidate has X years of total experience and Y years of relevant experience for this role.' Replace X with the total number of years of experience and Y with the number of years of relevant experience."
                prompt = PromptTemplate(input_variables=[], template=template)
                memory = ConversationBufferMemory(input_key='question', return_messages=True)

                llm = Ollama(model="mistral", verbose=True)
                qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=resume_vectorstore, memory=memory, input_key='question', prompt=prompt)

                response = qa({"question": "Based on the job description and the resume, how many years of total and relevant experience does the candidate have?"})
                st.write(response["output_text"])
            else:
                st.error("Failed to process the files.")
        else:
            st.warning("Please upload the files or enter the texts.")

if __name__ == "__main__":
    main()
