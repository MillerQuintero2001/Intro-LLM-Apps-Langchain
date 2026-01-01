from langchain_core.tools import tool
# from langgraph.prebuilt import create_react_agent # DEPRECATED
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)

customers = pd.DataFrame({
    "name": ["Peak Performance Co.", "Tech Innovations LLC"],
    "location": ["New York", "San Francisco"],
    "industry": ["Health & Fitness", "Technology"],
    "sales": [500000, 1200000],
    "costs": [300000, 800000]
})

@tool
def retrieve_customer_info(name: str) -> str:
    """Retrieve customer information based on their name."""
    customer_info = customers[customers['name'] == name]
    return customer_info.to_string()

# Create a ReAct agent
# agent = create_react_agent(llm, [retrieve_customer_info]) # DEPRECATED
agent = create_agent(llm, [retrieve_customer_info])
prompt_template = PromptTemplate.from_template("Create a summary of our customer: {customer_name}")

# Invoke the agent on the input
messages = agent.invoke({"messages": [("human", prompt_template.format(customer_name="Tech Innovations LLC"))]})
print(messages["messages"][-1].content)

print("=="*30)

messages = agent.invoke({"messages": [("human", prompt_template.format(customer_name="Peak Performance Co."))]})
print(messages["messages"][-1].content)
