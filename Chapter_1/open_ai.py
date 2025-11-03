from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

# Define the LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)

# Predict the words following the text in question
prompt = 'Three reasons for using LangChain for LLM application development.'
response = llm.invoke(prompt)

print(response.content)