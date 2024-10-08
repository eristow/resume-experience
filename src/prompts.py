"""
prompts.py

This module provides the prompts and templates used in the LLM/chatbot for extracting the number of years of relevant experience from a resume.
"""

from datetime import datetime
from langchain.prompts import ChatPromptTemplate

# ANALYSIS_QUESTION = f"Based on the Job Description and the provided Resume, extract the number of years of relevant experience from the resume. Provide your answer in the following format: 'Overall Experience: X years | Relevant Experience: X years | Notes: ...' Replace X with the actual number of years,... with a breakdown of the calculations for both overall and relevant experience, and ensure each section is separated by |. The output should be in years to the closest 0.5 year. Return an exact number for years of experience instead of \"Over X years\". For context and accurately calculating years of experience, the current date and time is {datetime.now()}."

# ANALYSIS_QUESTION = f"""
# Based on the Job Description and the provided Resume, extract the number of years of relevant experience from the resume. Provide your answer in the following format:

# 'Overall Experience: X years | Relevant Experience: Y years | Notes: ...'

# Replace X with the actual number of years of overall experience and Y with the actual number of years of relevant experience. In the 'Notes' section, provide a breakdown of the calculations for both overall and relevant experience, and include any additional considerations such as gaps in employment, part-time work, or other relevant factors. Ensure each section is separated by |. The output should be in years to the closest 0.5 year. Return an exact number for years of experience instead of "Over X years".

# For context and accurately calculating years of experience, the current date and time is {datetime.now()}.

# ### Instructions for Calculation:
# - **Overall Experience**: Sum the total years of work experience listed on the resume.
# - **Relevant Experience**: Sum the years of experience that directly relate to the skills and responsibilities outlined in the Job Description.

# ### Example Output:
# 'Overall Experience: 7.5 years | Relevant Experience: 5 years | Notes: Overall Experience: 3 years at Company A + 4.5 years at Company B. Relevant Experience: 2 years at Company A (related role) + 3 years at Company B (related role). Additional Considerations: 6-month gap in employment between Company A and Company B.'

# Ensure the 'Notes' section clearly explains how the years were calculated for both overall and relevant experience, and includes any additional considerations.
# """

ANALYSIS_QUESTION = f"""
Based on the Job Description and the provided Resume, extract the number of years of relevant experience from the resume. Provide your answer in the following format:

'Overall Experience: X years | Relevant Experience: Y years | Notes: ...'

Replace X with the actual number of years of overall experience and Y with the actual number of years of relevant experience. In the 'Notes' section, provide a brief explanation of how the years were calculated and include any additional considerations such as gaps in employment or part-time work. Ensure each section is separated by |. The output should be in years to the closest 0.5 year. Return an exact number for years of experience instead of "Over X years".

For context and accurately calculating years of experience, the current date and time is {datetime.now()}.
### Instructions for Calculation:
- **Overall Experience**: Sum the total years of work experience listed on the resume.
- **Relevant Experience**: Sum the years of experience that directly relate to the skills and responsibilities outlined in the Job Description.

### Example Output:
'Overall Experience: 7.5 years | Relevant Experience: 5 years | Notes: Overall Experience calculated from the start of the first job to the end of the most recent job. Relevant Experience includes roles that match the job description. Additional Considerations: 6-month gap in employment between jobs.'

Ensure the 'Notes' section clearly explains how the years were calculated for both overall and relevant experience, and includes any additional considerations.
"""

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
