from prompts import USER_PROMPT_FOR_MAIN_AGENT, WEDDING_PLANNER_AGENT_PROMPT
from langchain.agents import create_agent 
from dotenv import load_dotenv
from models import model_gpt
from agents import delegate_to_subagent1, delegate_to_subagent2
import logging 
from langchain.messages import HumanMessage
load_dotenv()

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s")


# Asking user for their requirements for their wedding 
logging.info('Asking user for their requirements....')
user_requirements = input('Please enter your requirements and preferences for your wedding\n')

# Updating the system_prompt 
updated_system_prompt = WEDDING_PLANNER_AGENT_PROMPT.format(requirements=user_requirements)

# Creating main agent
logging.info('initializing the main wedding planner agent...')
main_planning_wedding_agent = create_agent(model=model_gpt,
                                        tools=[delegate_to_subagent1, delegate_to_subagent2],
                                        name = "MainWeddingPlannerAgent",
                                        system_prompt = updated_system_prompt)

# Invoke agent 
logging.info('Invoking main wedding planner agent...')
response_main_agent = main_planning_wedding_agent.invoke({'messages': [HumanMessage(content=USER_PROMPT_FOR_MAIN_AGENT)]})

# Printing the response of the main agent
print("\nMain Wedding planner Agent's Response:")
print(response_main_agent['messages'][-1].content)



