from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from models import model_gpt
from tools import search_web
from dotenv import load_dotenv
import logging 

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# creating subagents
logging.info("initializing SubAgent1...")
sub_agent1 = create_agent(model=model_gpt, tools=[search_web], name="SubAgent1")
logging.info("initializing SubAgent2...")
sub_agent2 = create_agent(model=model_gpt, tools=[search_web], name="SubAgent2")

@tool 
def delegate_to_subagent1(query: str)->str:
    """Search the web for given a topic by delegating the task to SubAgent1 to do so"""
    logging.info(f'Delegating query to subagent1: {query}')
    response = sub_agent1.invoke({'messages':[HumanMessage(content=query)]})
    return response['messages'][-1].content

@tool 
def delegate_to_subagent2(query: str)->str:
    """Search the web for given a topic by delegating the task to SubAgent2 to do so"""
    logging.info(f'Delegating query to subagent2: {query}')
    response = sub_agent2.invoke({'messages':[HumanMessage(content=query)]})
    return response['messages'][-1].content