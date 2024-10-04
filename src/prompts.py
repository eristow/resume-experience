"""
prompts.py

This module provides the prompts and templates used in the LLM/chatbot for extracting the number of years of relevant experience from a resume.
"""

from datetime import datetime
from langchain.prompts import ChatPromptTemplate

ANALYSIS_QUESTION = f"Based on the Job Description and the provided Resume, extract the number of years of relevant experience from the resume. Provide your answer in the following format: 'Overall Experience: X years. Relevant Experience: X years. Notes: ...' Replace X with the actual number of years,... with any extra info. The output should be in years to the closest 0.5 year. Return an exact number for years of experience instead of \"Over X years\". For context and accurately calculating years of experience, the current date and time is {datetime.now()}."

CHAT_QUESTION = f'Based on the Job Description and the provided Resume, answer the user provided question. If the output contains years of experience, they should be in years to the closest 0.5 year. Return an exact number for years of experience instead of "Over X years". For context and accurately calculating years of experience, the current date and time is {datetime.now()}.'

# Define the query prompt
# QUERY_PROMPT_TEMPLATE = PromptTemplate(
#     input_variables=["question"],
#     template="""Based on the [Job Description] and the provided [Resume], extract the number of years of relevant experience from the resume. Provide your answer in the following format: 'The candidate has X years of relevant experience for this role.' Replace X with the actual number of years. The output should be in years to the closest 0.5 year.""",
# )

# Define the RAG prompt
ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    template="""Answer the question based ONLY on the following context:
    Resume: {resume_context}
    Job Description: {job_ad_context}
    Question: {question}"""
)

CHAT_PROMPT = ChatPromptTemplate.from_template(
    template="""Answer the question based ONLY on the following context:
    Resume: {resume_context}
    Job Description: {job_ad_context}
    User Input: {user_input}
    Question: {question}"""
)


# Simple passthrough function
def passthrough(input_data):
    """
    A function that returns the input data as is.

    Parameters:
    input_data (any): The input data to be returned.

    Returns:
    any: The input data.

    """
    return input_data
