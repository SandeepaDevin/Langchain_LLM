from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts.chat import ChatPromptTemplate
import os

os.environ['OPENAI_API_KEY']=''

########################################## PART 01 ################################################################################################
# Testing Basic OpenAI model invocation
#
# llm = OpenAI(model="text-davinci-003", temperature=0.9)
# text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
# print(llm(text))


########################################## PART 02 #############################################################################################
#    Now we compare difference between a llm model and chat model
#    llm = OpenAI()
#    chat_model = ChatOpenAI()

#    text = "What would be a good company name for a company that makes colorful socks?"
#    messages = [HumanMessage(content=text)]

#    print(llm.invoke(text))
#    print(chat_model.invoke(messages))


########################################## PART 03 ############################################################################################3

template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
