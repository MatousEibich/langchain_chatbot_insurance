from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]

loader = PyPDFLoader("european-insurance-in-figures-2020-data.pdf")
pages = loader.load()

print(pages[4])
print(len(pages[4].page_content))
print(len(pages))

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = r_splitter.split_documents(pages)

print(chunks[4])
print(len(chunks[4].page_content))
print(len(chunks))

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents = chunks,
    embedding = embedding,
    # persist_directory= zatim staci in memory vectordb?
)

question = "What were total gross premiums written in the European countries in the year of the report?"
docs = vectordb.similarity_search(question, k = 3)
print(docs[1])
print(docs[2])

llm = OpenAI(temperature = 0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor = compressor,
    base_retriever = vectordb.as_retriever()
)

question = "What were total gross premiums written in the European countries in the year of the report?"
compressed_docs = compression_retriever.get_relevant_documents(question)
print(compressed_docs[0])
print(compressed_docs[1])

chat_llm = ChatOpenAI(model_name = "gpt-3.5-turbo",
                     temperature = 0)

qa_chain = RetrievalQA.from_chain_type(
    chat_llm,
    retriever = compression_retriever
)

result = qa_chain({"query": question})
print(result["result"])
question = "What were the total claims and benefits paid in Europe?"
result = qa_chain({"query": question})
print(result["result"])
question = "What types of retirement products are there?"
result = qa_chain({"query": question})
print(result["result"])
question = "How is Norway life insurance looking?"
result = qa_chain({"query": question})
print(result["result"])
question = "How are looking claims in UK?"
result = qa_chain({"query": question})
print(result["result"])