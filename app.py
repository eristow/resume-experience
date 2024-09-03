import streamlit as st
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from tools import CustomEmbeddings, process_file, extract_text_from_file
import os
from datetime import datetime

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Simple passthrough function
def passthrough(input_data):
    return input_data


# Function to analyze inputs and return relevant information
def analyze_inputs(job_file_path, resume_file_path, job_ad_text, resume_text):
    if job_file_path and resume_file_path:
        # Use the local Mistral model
        embeddings = CustomEmbeddings(model_name="./mistral")

        job_ad_vectorstore = process_file(job_file_path, embeddings)
        print("after job_ad_vectorstore")
        resume_vectorstore = process_file(resume_file_path, embeddings)
        print("after resume_vectorstore")
        print(f"job_ad_vectorstore: {job_ad_vectorstore}")
        print(f"resume_vectorstore: {resume_vectorstore}")

        if job_ad_vectorstore and resume_vectorstore:
            print("both vectorstores exist")
            resume_retriever = resume_vectorstore.as_retriever()
            job_ad_retriever = job_ad_vectorstore.as_retriever()
            print("after retriever")
            chain = (
                {
                    "resume_context": resume_retriever,
                    "job_ad_context": job_ad_retriever,
                    "question": lambda x: QUERY_PROMPT,
                }
                | PROMPT
                | ChatOllama(model="mistral:v0.3", temperature=0.3)
                | passthrough  # Simple output parsing
            )
            print("after chain")
            response = chain.invoke("Analyze the resume based on the job description")
            print("after response")
            return response

    return "Failed to process the files."


# Streamlit UI components
def main():
    st.title("Resume Experience Analyzer")

    st.write(
        "This app will compare a job description to a resume and extract the number of years of relevant work experience from the resume."
    )

    col1, col2 = st.columns(2)

    job_ad_text = ""
    resume_text = ""

    with col1:
        job_file = st.file_uploader(
            "Upload Job Description", type=["pdf", "doc", "docx"], key="job_file"
        )
        if job_file:
            job_file_path = os.path.join("temp_dir", job_file.name)
            with open(job_file_path, "wb") as temp_file:
                temp_file.write(job_file.getbuffer())
            job_ad_text = extract_text_from_file(job_file, job_file_path)

    with col2:
        resume_file = st.file_uploader(
            "Upload Resume", type=["pdf", "doc", "docx"], key="resume_file"
        )
        if resume_file:
            resume_file_path = os.path.join("temp_dir", resume_file.name)
            with open(resume_file_path, "wb") as temp_file:
                temp_file.write(resume_file.getbuffer())
            resume_text = extract_text_from_file(resume_file, resume_file_path)

    col1.text_area("Job Description", value=job_ad_text, height=300)
    col2.text_area("Resume Text", value=resume_text, height=300)

    if st.button("Analyze"):
        with st.spinner("Processing (this can take a few minutes)..."):
            result = analyze_inputs(
                job_file_path, resume_file_path, job_ad_text, resume_text
            )
            st.write(result.content)
            # TODO: Format output and display in Output Experience section

    # Display output experience
    st.subheader("Output Experience")
    st.write("Overall Experience: X years")
    st.write("Relevant Experience: X years")
    st.write("Notes: ...")

    # Chatbot feature
    st.subheader("Chatbot Feature")
    user_input = st.text_input("Ask a question:")

    if st.button("Submit Query"):
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]

        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        # TODO: provide job ad and resume context
        response = ChatOllama().invoke(user_input)
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": response}
        )

        for chat in st.session_state["chat_history"]:
            st.write(f"{chat['role']}: {chat['content']}")


QUERY_PROMPT = f"Based on the Job Description and the provided Resume, extract the number of years of relevant experience from the resume. Provide your answer in the following format: 'The candidate has X years of relevant experience for this role.' Replace X with the actual number of years. The output should be in years to the closest 0.5 year. Only consider relevant years of experience obtained from work and not from skills. For context and accurately calculating years of experience, the current year is {datetime.now().year}."

# Define the query prompt
# QUERY_PROMPT_TEMPLATE = PromptTemplate(
#     input_variables=["question"],
#     template="""Based on the [Job Description] and the provided [Resume], extract the number of years of relevant experience from the resume. Provide your answer in the following format: 'The candidate has X years of relevant experience for this role.' Replace X with the actual number of years. The output should be in years to the closest 0.5 year.""",
# )

# Define the RAG prompt
PROMPT = ChatPromptTemplate.from_template(
    template="""Answer the question based ONLY on the following context:
    Resume: {resume_context}
    Job Description: {job_ad_context}
    Question: {question}"""
)

if __name__ == "__main__":
    # Create a temporary directory to save uploaded files
    os.makedirs("temp_dir", exist_ok=True)
    main()
    # Clean up the temporary directory after the app closes
    import shutil

    shutil.rmtree("temp_dir")
