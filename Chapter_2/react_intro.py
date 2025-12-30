# from langgraph.prebuilt import create_react_agent # DEPRECATED
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.messages import HumanMessage
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY,
                 temperature=0.1, seed=42)

# Define the tools
tools = load_tools(["wikipedia"], llm=llm)

# Define the agent
# agent = create_react_agent(llm, tools) # DEPRECATED
agent = create_agent(llm, tools)

# Invoke the agent
# response = agent.invoke({"messages": [("human", "How many people live in New York City?")]})
# NOTE: More LangChain idiomatic way
response = agent.invoke({"messages": [HumanMessage(content="How many people live in New York City?")]})
print(response["messages"][-1].content)
