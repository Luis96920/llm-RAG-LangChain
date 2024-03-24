# Description: This file contains the code to create a chatbot that can answer questions about patient reviews of hospitals.
# The chatbot uses the langchain_openai module to interact with the OpenAI API and the langchain module to create a chat prompt template.

# Import the required modules
import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
# Import the output parser. This is used to parse the output from the chatbot.
from langchain_core.output_parsers import StrOutputParser

# Load the environment variables
dotenv.load_dotenv()

# Define the template for the chatbot
review_template_str = """Your job is to use patient
reviews to answer questions about their experience at
a hospital. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.

{context}
"""

# Review system prompt is a template for the chatbot to ask the user for a review. 
review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

# Review human prompt is a template for the chatbot to ask the user for a question.
review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
# Combine the system and human prompts into a list
messages = [review_system_prompt, review_human_prompt]

# Create a chat prompt template using the messages
review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

# Create a chatbot using the OpenAI API
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Chain the review prompt template and the chatbot to create a chatbot that can answer questions about patient reviews of hospitals.
output_parser = StrOutputParser()

# Chain the review prompt template, chatbot, and output parser. This creates a chatbot that can answer questions about patient reviews of hospitals.
review_chain = review_prompt_template | chat_model | output_parser