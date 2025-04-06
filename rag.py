import os
import bs4
from langchain_openai import OpenAI
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from dotenv import load_dotenv

load_dotenv()

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

gpt = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo-1106",
    openai_api_key="api key",
    openai_api_type="azure",
    openai_api_version="2023-12-01-preview",
    azure_endpoint="https://chatbotopenaikeyswe.openai.azure.com/",
    verbose=True,
)


path = "C:/Users/Admin/Desktop/hsag chatbot/CSVDATASET"
csv_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader(path, glob="**/*.csv", loader_cls=CSVLoader, loader_kwargs=csv_loader_kwargs)
db = loader.load()
#path = 'C:/Users/Admin/Desktop/RAG pipeline/Dataset/CSVDATASET'
#csv_loader = DirectoryLoader(path, glob="**/*.csv", loader_cls=CSVLoader)
#db = csv_loader.load()

urls = ["https://www.thuega-energie-gmbh.de/privatkunden.html"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()

html2text = Html2TextTransformer()
db += html2text.transform_documents(docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(db)

embed = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_key="api key",
    azure_endpoint="https://chatbotopenaikeyswe.openai.azure.com/"
)

#vectorstore = Chroma.from_documents(texts, embed, persist_directory="C:/chroma_db")
#vectorstore.persist()

vectorstore2 = Chroma(persist_directory="C:/chroma_db", embedding_function=embed)
retriever = vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k":3})

prompt = hub.pull("rlm/rag-prompt")

template = """You are an assistant named "hsag-chatbot" for question-answering tasks 
related to the Energy industry and general conversation. 

If the question is related to the energy domain try to answer the question from the knowledge you have in your memory.

Use the following pieces of retrieved context to answer the question 
or engage in small talk with the user in a friendly and informative way. 
If you don't know the answer to a factual question, 
just say that you don't know. 
Use three sentences maximum and keep the answer concise. 
If the user asks funny questions or jokes, try to answer them 
using your knowledge or generate a humorous response. 
If they ask general knowledge about the world, try to answer 
those questions using your knowledge and also domain-specific knowledge 
about the energy industry. 
Indicate the dataset from which you retrieved the answer (if applicable) 
in the response.

Context:
{context}

Question:
{question}
"""




prompt = ChatPromptTemplate.from_template(template)
def format_docs(pages):
    return "\n\n".join(doc.page_content for doc in pages)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | gpt
    | StrOutputParser()
)

def get_response(user_input):
    response = rag_chain.invoke(user_input)
    return response
