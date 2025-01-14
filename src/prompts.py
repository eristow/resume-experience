from datetime import datetime
from langchain.prompts import ChatPromptTemplate

ANALYSIS_QUESTION = f"""
**Instructions for Date Calculation:**
- **Overall Experience:** Sum the durations of all jobs.
- **Relevant Experience:** Include only roles that directly relate to the Job Description. Sum their durations similarly.
- **Output Format:** Output in years/months/days.

Based on the Job Description, the provided Resume, and the Instructions for Date Calculation, calculate the number of years/months/days of relevant experience from the resume. Provide your answer in the following format:

'Notes: Explanation: ...'

In the 'Explanation' section, provide a brief explanation of how the years/months/days were calculated and include any additional considerations such as gaps in employment or part-time work. Ensure each section is separated by |.

**Important:** Ensure the **Explanation** section clearly explains how the years/months/days were calculated for both overall and relevant experience, including the start and end dates of each job, and any additional considerations.
"""

CHAT_QUESTION = f"""
Based on the Job Description and the provided Resume, answer the user-provided question. If the output contains years of experience, they should be in years/months/days.
"""

# Define the RAG prompt
ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    template="""Answer the question based ONLY on the following context:
    Resume: {resume_context}
    Job Description: {job_context}
    Question: {question}"""
)

CHAT_PROMPT = ChatPromptTemplate.from_template(
    template="""Answer the question based ONLY on the following context:
    Resume: {resume_context}
    Job Description: {job_context}
    User Input: {user_input}
    Question: {question}"""
)


# Simple passthrough function
def passthrough(input_data: any) -> any:
    """A function that returns the input data as is."""
    return input_data
