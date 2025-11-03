from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)

# Define the prompt templates

# 1. Learning prompt, it indicates the activity to learn, and is the first step in the chain
learning_prompt = PromptTemplate(
    input_variables=["activity"],
    template="I want to learn how to {activity}. Can you suggest how I can learn this step-by-step?"
)

# 2. Once we have the activity to learn, we want to create a concise plan for one week
time_prompt = PromptTemplate(
    input_variables=["learning_plan"],
    template="I only have one week. Can you create a concise plan to help me hit this goal: {learning_plan}."
)

# Complete the sequential chain with LCEL (LangChain Expression Language)
# The chain will receive the input realted with the first element in the chain
# but, the key specified in the chain should be the one expected by the second element
seq_chain = ({"learning_plan": learning_prompt | llm | StrOutputParser()}
    | time_prompt
    | llm
    | StrOutputParser())

# Call the chain, with the key of the first element in the chain
print(seq_chain.invoke({"activity": "code in Rust"}))