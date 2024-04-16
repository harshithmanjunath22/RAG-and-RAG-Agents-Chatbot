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



gpt = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo-1106",
    openai_api_key="7c3f9550b69c419aa0f1830e338ff562",
    openai_api_type="azure",
    openai_api_version="2023-12-01-preview",
    azure_endpoint="https://chatbotopenaikeyswe.openai.azure.com/",
    verbose=True,
)

path = 'C:/Users/Admin/Desktop/RAG pipeline/Dataset/CSVDATASET'
csv_loader = DirectoryLoader(path, glob="**/*.csv", loader_cls=CSVLoader)
db = csv_loader.load()

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
    openai_api_key="7c3f9550b69c419aa0f1830e338ff562",
    azure_endpoint="https://chatbotopenaikeyswe.openai.azure.com/"
)

#vectorstore = Chroma.from_documents(texts, embed, persist_directory="C:/chroma_db")
#vectorstore.persist()

vectorstore2 = Chroma(persist_directory="C:/chroma_db", embedding_function=embed)
retriever = vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k":3})

prompt = hub.pull("rlm/rag-prompt")
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