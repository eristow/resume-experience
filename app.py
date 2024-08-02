import streamlit as st
from tools import CustomEmbeddings, process_file, extract_text_from_file

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Simple passthrough function
def passthrough(input_data):
    return input_data

# Function to analyze inputs and return relevant information
def analyze_inputs(job_file_path, resume_file_path, job_ad_text, resume_text):
    embeddings = CustomEmbeddings(model_name="./mistral")
    job_ad_vectorstore = process_file(job_file_path, embeddings)
    resume_vectorstore = process_file(resume_file_path, embeddings)

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

# Streamlit UI components
def main():
    st.title("Resume Experience Analyzer")

    col1, col2 = st.columns(2)
    
    with col1:
        job_file = st.file_uploader("Upload Job Description", type=["pdf", "doc", "docx"])
        job_file_path = extract_text_from_file(job_file)
        if job_file_path:
            job_ad_text = open(job_file_path, "r").read()
            st.text_area("Job Description Text", value=job_ad_text, height=300)

    with col2:
        resume_file = st.file_uploader("Upload Resume", type=["pdf", "doc", "docx"])
        resume_file_path = extract_text_from_file(resume_file)
        if resume_file_path:
            resume_text = open(resume_file_path, "r").read()
            st.text_area("Resume Text", value=resume_text, height=300)

    if st.button("Analyze"):
        with st.spinner("Processing..."):
            result = analyze_inputs(job_file_path, resume_file_path, job_ad_text, resume_text)
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