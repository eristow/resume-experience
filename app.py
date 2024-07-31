"""
Resume Experience App

Description: This app compares a job description to a resume and extract
the number of years of relevant work experience from
the resume.

"""
# Define CustomEmbeddings class
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
    
    def embed_documents(self, documents):
        embeddings = []
        for doc in documents:
            tokens = self.tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**tokens)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
        return embeddings

    def embed_query(self, query):
        tokens = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Simple passthrough function
def passthrough(input_data):
    return input_data

# Define file processing function
def process_file(uploaded_file, file_path, embeddings):
    if os.path.isfile(file_path):
        print ("after if")
        with st.spinner("Analyzing your document..."):
            reader = PdfReader(file_path)
            print ("after reader")
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            print(f'text: {text}')
            with open("temp_text.txt", "w") as text_file:
                text_file.write(text)
            print("created temp txt file")
            
            loader = TextLoader("temp_text.txt")
            data = loader.load()
            print ("after data")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
            print ("after text_splitter")
            chunks = text_splitter.split_documents(data)
            print ("after chunks")
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
            print ("after vectorstore")
            return vectorstore

# Function to analyze inputs and return relevant information
def analyze_inputs(uploaded_files, job_ad_text, resume_text):
    print (uploaded_files, job_ad_text, resume_text)

    if uploaded_files:
        # Use the local Mistral model
        embeddings = CustomEmbeddings(model_name="/home/tristow/resume-experience/mistral")
        job_ad_vectorstore = None
        resume_vectorstore = None
        print ("after embeddings")

        for file in uploaded_files:
            # Save the uploaded file to a temporary location
            temp_file_path = os.path.join("temp_dir", file.name)
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file.getbuffer())

            if "Job" in file.name:
                job_ad_vectorstore = process_file(file, temp_file_path, embeddings)
                print ("job")
            else:
                resume_vectorstore = process_file(file, temp_file_path, embeddings)
                print ("resume")
        print (job_ad_vectorstore)
        print (resume_vectorstore)

        if job_ad_vectorstore and resume_vectorstore:
            retriever = MultiQueryRetriever.from_llm(
                vector_db=job_ad_vectorstore.as_retriever(),
                llm=ChatOllama(),
                prompt=QUERY_PROMPT,
            )
            print ("retriever")
            chain = (
                {"context": retriever, "question": passthrough}
                | PROMPT
                | ChatOllama()
                | passthrough  # Simple output parsing
            )
            print ("chain")
            response = chain.invoke("Analyze the resume based on the job description")
            print ("response")
            return response

    return "Failed to process the files."

# Streamlit UI components
def main():
    st.title("Resume Experience Analyzer")

    st.write("This app will compare a job description to a resume and extract the number of years of relevant work experience from the resume.")

    uploaded_files = st.file_uploader("Upload Job Descriptions and Resumes", type=["pdf", "doc", "docx"], accept_multiple_files=True)
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
