from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import os
from langchain import HuggingFaceHub
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
# openai.api_key = os.environ["OPENAI_API_KEY"]

loader = PyPDFDirectoryLoader("IFRS17")
pages = loader.load()

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=600
)

chunks = r_splitter.split_documents(pages)

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents = chunks,
    embedding = embedding,
    # persist_directory= zatim staci in memory vectordb?
)

repo_id = "google/flan-t5-base"

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0}
)

compressor = LLMChainExtractor.from_llm(llm)

retriever = vectordb.as_retriever(search_type = "mmr")
retriever.search_kwargs = {'k':10}

compression_retriever = ContextualCompressionRetriever(
    base_compressor = compressor,
    base_retriever = retriever
)

question = "What are the submodules of natural catastrophe risk sub module?"
compressed_docs = compression_retriever.get_relevant_documents(question)
print(compressed_docs[0])
print(compressed_docs[1])

#
from langchain import HuggingFacePipeline

chat_llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    task="text-generation",
    model_kwargs={"temperature": 0},
)

prompt_template = """/
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Both the question and context are about IFRS17 Standard, which is a very complicated legal topic. 
You may often need to consider multiple pieces of context together to come up with the final answer.

{context}

Question: {question}
Answer:?
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

qa_chain = RetrievalQA.from_chain_type(
    chat_llm,
    retriever = compression_retriever,
    chain_type_kwargs = chain_type_kwargs
)
question = "What are IFRS17 guidelines on constructing CF discount curve?"
result = qa_chain({"query": question})
print(result["result"])

question = "How should one construct CF discount curve for products with profit sharing?"
result = qa_chain({"query": question})
print(result["result"])

question = "What are IFRS17 guidelines on constructing CF discount curve for products without profit sharing?"
result = qa_chain({"query": question})
print(result["result"])

question = "What are IFRS17 guidelines on constructing CF discount curve for reinsurance contracts?"
result = qa_chain({"query": question})
print(result["result"])

question = "Please explain in detail bottom up approach to constructing a CF discount curve."
result = qa_chain({"query": question})
print(result["result"])

question = "Please explain in detail top down approach to constructing a CF discount curve."
result = qa_chain({"query": question})
print(result["result"])

question = "In the context of creating a CF discount curve, what does risk-free mean?"
result = qa_chain({"query": question})
print(result["result"])

question = "In the context of creating a CF discount curve, what does credit spread mean?"
result = qa_chain({"query": question})
print(result["result"])

question = "In the context of creating a CF discount curve, what does illiquidity premium mean?"
result = qa_chain({"query": question})
print(result["result"])