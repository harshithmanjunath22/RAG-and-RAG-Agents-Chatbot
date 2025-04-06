import os
import bs4
from langchain_openai import OpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter
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
    temperature=0.1,
)


embed = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_key="api key",
    azure_endpoint="https://chatbotopenaikeyswe.openai.azure.com/"
)

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import CSVLoader


import requests

# Base URL of your mock server
base_url = "https://b07dcf0f-cb3c-49d3-9990-47521f1415d5.mock.pstmn.io"

# Customer data endpoint
meter_reading_url = f"{base_url}/meter/getMeterForCustomer"

# Meter reading data endpoint
customer_url = f"{base_url}/contract/getCustomerData"

# Send GET requests to each endpoint
customer_response = requests.get(customer_url)
meter_reading_response = requests.get(meter_reading_url)

# Handle responses
if customer_response.status_code == 200:
  customer_data = customer_response.json()
  #print("Customer Data:", customer_data)
#else:
#  print("Error:", customer_response.status_code)

if meter_reading_response.status_code == 200:
  meter_reading_data = meter_reading_response.json()
  #print("Meter Reading Data:", meter_reading_data)
#else:
 # print("Error:", meter_reading_response.status_code)

erpdata = {**customer_data, **meter_reading_data}



from langchain_community.document_loaders import AsyncHtmlLoader

urls = ["https://www.thuega-energie-gmbh.de/privatkunden.html"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()

from langchain_community.document_transformers import Html2TextTransformer

html2text = Html2TextTransformer()
thuegaenergie_company_data = html2text.transform_documents(docs)




thuegaenergie_company_data_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)


erpdata_text_splitter = RecursiveJsonSplitter(max_chunk_size=50)
chuncked_erp_data = erpdata_text_splitter.create_documents(texts=[erpdata])


chuncked_thuegaenergie_company_data = thuegaenergie_company_data_text_splitter.split_documents(thuegaenergie_company_data)

#erp_data_vectorstore = Chroma.from_documents(chuncked_erp_data, embed, persist_directory="C:/erp_db")
#erp_data_vectorstore.persist()

#thuegaenergie_company_data_vectorstore = Chroma.from_documents(chuncked_thuegaenergie_company_data, embed, persist_directory="C:/thuegaenergie_company_db")
#thuegaenergie_company_data_vectorstore.persist()


erp_data_vectorstore2 = Chroma(persist_directory="C:/erp_db", embedding_function=embed)
thuegaenergie_company_data_vectorstore2 = Chroma(persist_directory="C:/thuegaenergie_company_db", embedding_function=embed)


retriever1 = erp_data_vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k":3})

retriever2 = thuegaenergie_company_data_vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k":3})

from langchain.tools.retriever import create_retriever_tool


erp_data_retriever_tool=create_retriever_tool(retriever1,"erpdatabase_search",
                      "Search for information about Customerdata and meterreading. For any questions about Customerdata and meterreading, you must use this tool!")

thuegaenergie_company_data_retriever_tool=create_retriever_tool(retriever2,"thuegaenergiecompany_search",
                      "Search for information about thuegaenergie company. For any questions about thuegaenergie company, you must use this tool!")


from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)


tools=[erp_data_retriever_tool,  
       thuegaenergie_company_data_retriever_tool,
       wiki ]

prompt = hub.pull("hwchase17/openai-functions-agent")

from langchain.agents import create_openai_tools_agent
agent=create_openai_tools_agent(gpt,tools,prompt)

from langchain.agents import AgentExecutor
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=False)


def get_response(user_input):
    response = agent_executor.invoke({"input": user_input})
    return response
