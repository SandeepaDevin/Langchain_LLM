import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import (
    InMemoryStore,
    LocalFileStore,
    RedisStore,
    UpstashRedisStore,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

os.environ['OPENAI_API_KEY']=''
######### To import data from csv files ######################################################
#loader = CSVLoader(file_path='./example_data/mlb_teams_2012.csv',csv_args={
#    'delimiter': ',',
#    'quotechar': '"',
#    'fieldnames': ['MLB Team', 'Payroll in millions', 'Wins']
#})

#data = loader.load()

################### to import data from pdfs ####################################################
#   loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
#   pages = loader.load_and_split()
#faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
#docs = faiss_index.similarity_search("How will the community be engaged?", k=2)
#for doc in docs:
#    print(str(doc.metadata["page"]) + ":", doc.page_content[:300])


################### Testing loader and splitter ##################################################
#    loader = TextLoader("./my_information.txt")
#    data = loader.load()

#    text_splitter = RecursiveCharacterTextSplitter(
#        chunk_size = 100,
#        chunk_overlap  = 20,
#        length_function = len,
#        add_start_index = True,
#    )

#    with open('./my_information.txt') as f:
#        content = f.read()
        
#    texts = list(text_splitter.create_documents([content]))
#    page_content_list = [element.page_content for element in texts]
# print(page_content_list)


################### Testing embedding of openai ####################################################

#embeddings_model = OpenAIEmbeddings()
#embeddings = embeddings_model.embed_documents(page_content_list)
#print(len(embeddings), len(embeddings[0]))

#embedded_query = embeddings_model.embed_query("What was the name of the author?")
#print(embedded_query[:5])


################### Using FAISS vector database to store and query ####################################
# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader("./my_information.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=5)
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())

query = "When was author born ?"
#docs = db.similarity_search(query)
#print(docs[0].page_content)

embedding_vector = OpenAIEmbeddings().embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)