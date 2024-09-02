# flake8: noqa
# Import the necessary module from LangChain
from langchain.prompts import PromptTemplate

# A welcome message displayed to users when they start the application
WELCOME_MESSAGE = """\
Welcome to Introduction to LLM App Development Sample PDF QA Application!
To get started:
1. Upload a PDF or text file
2. Ask any question about the file!
"""

# Template for generating answers as an expert financial analyst.
# Instructions to focus on financial statements, especially operating margin calculations.
# The prompt will guide the model to generate a final answer with references ("SOURCES").
template = """Please act as an expert financial analyst when you answer the questions and pay special attention to the financial statements.  Operating margin is also known as op margin and is calculated by dividing operating income by revenue.
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" field in your answer, with the format "SOURCES: <source1>, <source2>, <source3>, ...".

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

# Creating a PromptTemplate object using the template defined above,
# with placeholders for 'summaries' and 'question' that will be filled in dynamically.
PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

# Example prompt template for formatting extracted content from a document.
# Each section of content is paired with its source.
EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)