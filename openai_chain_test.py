from langchain.chat_models import ChatOpenAI
import os

llm = ChatOpenAI(openai_api_key=os.environ['OPENAI_API'])

