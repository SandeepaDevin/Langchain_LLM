from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import os

os.environ['OPENAI_API_KEY']=''

prompt = PromptTemplate(template="Question: {question}\nAnswer:", input_variables=["question"])

llm = OpenAI(model_name="text-davinci-003", temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("what is the meaning of life?"))
print(chain.run("How can we control time-space variation ?"))