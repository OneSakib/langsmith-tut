from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableConfig
import os

os.environ['LANGCHAIN_PROJECT'] = 'RAG App v1'

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
if not os.path.exists('faiss_db'):
    loader = PyPDFLoader('islr.pdf')
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150)
    documents = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(
        embedding=embeddings, documents=documents)
    vectorstore.save_local('faiss_db')
else:
    vectorstore = FAISS.load_local(
        'faiss_db', embeddings=embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={'k': 4})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from the provided context. if answer not found in the context then say no"),
    ("human", "Question: {question} \n\n  {context}")
])

llm = ChatOpenAI(model="gpt-5-nano")


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()
chain = parallel | prompt | llm | parser
config: RunnableConfig = {
    "run_name": "Rag Based App",
    "tags": ['Rag', 'Pdf', 'vector store'],
}

print("PDF Rag is Ready, Now you ask a question related to this document\n")
q = input("Q: ")
ans = chain.invoke(q.strip(), config=config)
print(f"Ans: {ans}")
