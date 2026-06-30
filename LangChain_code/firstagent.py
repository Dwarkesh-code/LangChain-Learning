from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

model = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")

search_tool = DuckDuckGoSearchRun()

@tool
def multiplication(x : float, y: float) -> float:
    """For multiply
    
    Args :- 
    x = first number
    y = second number"""
    return x*y

agent = create_agent(
    model=model,
    tools=[search_tool, multiplication]
)

result = agent.invoke({"messages": [{'role': 'user', 'content':"What's the current 1 bitcoin price. And multiply it with 5"}]})
print("reasoning content:",result["messages"][-1].content)