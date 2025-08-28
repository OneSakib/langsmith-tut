from typing import cast
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.runnables import RunnableConfig
os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'
load_dotenv()


prompt_1 = PromptTemplate(
    template="Generate a detail report on {topic}", input_variables=['topic'])
prompt_2 = PromptTemplate(
    template="Generate a five point summary on the following text \n {text}", input_variables=['text'])
model = ChatOpenAI(model="gpt-5-nano", temperature=0.5)
parser = StrOutputParser()

chain = prompt_1 | model | parser | prompt_2 | model | parser

config: RunnableConfig = cast(
    RunnableConfig,
    {
        'run_name': 'Sequential Chain',
        'tags': ['LLM App', 'Report Generator', 'Summarization'],
        'metadata': {'model1': 'gpt-4o-mini', 'model2': 'gpt-4o'}
    }
)
result = chain.invoke({"topic": "Unemployment in India"}, config=config)

print(result)
