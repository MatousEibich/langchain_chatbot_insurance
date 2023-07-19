from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]

loader = PyPDFLoader("CDR 2015-35 Solvency II EN conso 08072019.pdf")
pages = loader.load()

print(pages[4])
print(len(pages[4].page_content))
print(len(pages))

pages = pages[1:300]

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=600
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

question = "What are the submodules of natural catastrophe risk sub module?"
docs = vectordb.similarity_search(question, k = 6)
print(docs[1])
print(docs[2])

llm = OpenAI(temperature = 0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor = compressor,
    base_retriever = vectordb.as_retriever()
)

question = "What are the submodules of natural catastrophe risk sub module?"
compressed_docs = compression_retriever.get_relevant_documents(question)
print(compressed_docs[0])
print(compressed_docs[1])

chat_llm = ChatOpenAI(model_name = "gpt-3.5-turbo",
                     temperature = 0)

prompt_template = """/
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Both the question and context are about Solvency II, which is a very complicated legal topic. 
You may often need to consider multiple pieces of context together to come up with the final answer.
Also, the context consists of equations, that are not read correctly. Never use these incorrectly formatted 
equations to formulate the final answer. Only use the plain legal english text. 

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

question = "What are the submodules of natural catastrophe risk sub module?"
result = qa_chain({"query": question})
print(result["result"])

question = "How is the windstorm risk submodule calculated?"
result = qa_chain({"query": question})
print(result["result"])

question = "How is the capital requirement for health expense risk calculated?"
result = qa_chain({"query": question})
print(result["result"])

question = "How is the capital requirement for SLT health mass lapse risk calculated?"
result = qa_chain({"query": question})
print(result["result"])

question = "How is the capital requirement for pandemic risk sub-module calculated?"
result = qa_chain({"query": question})
print(result["result"])

question = "What criteria need to be met for an investment to be considered qualifying infrastructure investment?"
result = qa_chain({"query": question})
print(result["result"])

question = "How is the capital requirement for spread risk calculated?"
result = qa_chain({"query": question})
print(result["result"])

question = "What criteria need to be met for data used to calculate undertaking-specific parameters?"
result = qa_chain({"query": question})
print(result["result"])

question = "What is Minimum Capital Requirement and how is it calculated?"
result = qa_chain({"query": question})
print(result["result"])

question = "What are some functions described in the Solvency II legislature and what do they do?"
result = qa_chain({"query": question})
print(result["result"])