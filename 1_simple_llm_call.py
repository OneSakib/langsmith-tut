from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


prompt = PromptTemplate(template="{question}", input_variables=['question'])
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({"question": "What is the capital of india?"})
print(result)
