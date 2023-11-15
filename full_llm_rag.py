import os
import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

os.environ['OPENAI_API_KEY']=''

#url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
#res = requests.get(url)
#with open("state_of_the_union.txt", "w") as f:
#    f.write(res.text)

loader = TextLoader('./my_information.txt')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=80, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

vectorstore =  FAISS.from_documents(chunks,OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
)

query = "What did the author's parents do ?"
print(rag_chain.invoke(query))