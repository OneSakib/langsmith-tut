from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
import os


load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'AI Agent'

search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
    '''
    This function fetch the current weather data for a given city
    '''
    url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'
    response = requests.get(url)
    return response.json()


llm = ChatOpenAI()

prompt = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)
agent_executor = AgentExecutor(agent=agent,
                               tools=[search_tool, get_weather_data],
                               verbose=True,
                               max_iterations=5
                               )


response = agent_executor.invoke(
    {"input": "What is release data of coolie 2025?"})
print(response)
