from datetime import datetime
from langchain.prompts import ChatPromptTemplate

# TODO: add the idea of half-relevancy into the prompt?
ANALYSIS_QUESTION = f"""
**Instructions for Formatting Output:**
- **Overall Experience:** Take length from "Total length of jobs:" section.
- **Relevant Experience:** Include only jobs from the job info that directly relate to the Job Ad. List each relevant job with its corresponding length.
- **Output Format:** Output in years/months/days.

Based on the Job Ad, the provided job info of the candidate, and the Instructions for Formatting Output, calculate the number of years/months/days of relevant experience from the given job info.

Provide a brief explanation of how the years/months/days were calculated and include any additional considerations such as gaps in employment or part-time work. Ensure each section is separated by |.

**Important:** Ensure the **Explanation** section clearly explains how the years/months/days were calculated for both overall and relevant experience, including any additional considerations.
"""

CHAT_QUESTION = f"""
Based on the Job Ad, and the provided job info of the candidate, answer the user-provided question. If the output contains years of experience, they should be in years/months/days.
"""

# Define the RAG prompt
ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    # template="""Answer the question based ONLY on the following context:
    # Job Info of the Candidate: {resume_context}
    # Job Ad: {job_context}
    # Question: {question}"""
    template="""
**Job Ad**
{job_context}

**Job Info of the Candidate**
{resume_context}

**Instructions for Formatting Output:**
- **Overall Experience:** Take length from "Total length of jobs:" section.
- **Relevant Experience:** Include only jobs from the job info that directly relate to the Job Ad. List each relevant job with its corresponding length.
- **Output Format:** Output in years/months/days.

Based on the Job Ad, the provided Job Info of the Candidate, and the Instructions for Formatting Output, calculate the number of years/months/days of relevant experience from the given job info. The Job Ad does not count as relevant experience for the candidate.

Provide a brief explanation of how the years/months/days were calculated and include any additional considerations such as gaps in employment or part-time work. Ensure each section is separated by |.

**Important:** Ensure the **Explanation** section clearly explains how the years/months/days were calculated for both overall and relevant experience, including any additional considerations.
    """
)

CHAT_PROMPT = ChatPromptTemplate.from_template(
    # template="""Answer the question based ONLY on the following context:
    # Job Info of the Candidate: {resume_context}
    # Job Ad: {job_context}
    # User Input: {user_input}
    # Question: {question}"""
    template="""
Based on the Job Ad {job_context}, and the provided job info of the candidate {resume_context}, answer the user-provided question. If the output contains years of experience, they should be in years/months/days.
    """
)


# Simple passthrough function
def passthrough(input_data: any) -> any:
    """A function that returns the input data as is."""
    return input_data
