import random
import time
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms.ollama import Ollama
from sentence_transformers import CrossEncoder

# Load LLM using Ollama
llm = Ollama(model="gemma:2b")

#Load Embedding and Re-rannker models
reranker = CrossEncoder('mixedbread-ai/mxbai-rerank-large-v1')
embedding = embedding_functions.SentenceTransformerEmbeddingFunction("WhereIsAI/UAE-Large-V1")

# Load Chroma DB
chroma_client = chromadb.PersistentClient(path="chroma_db/")
db=chroma_client.get_collection(name="finance_fusion",embedding_function=embedding)
#print(db.peek())

def cross_encoder_rerankar(documents,query):
    results = reranker.rank(query, documents, return_documents=True, top_k=2)
    print("[Info]: Results has been re-ranked.")
    return results

def croma_db_retriever(query,n_results):
    print("[Info]: Documents has been retrived.")
    return db.query(query_texts=query,n_results=n_results, include=["metadatas", "documents"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def combine_chunks(chunks):
    return "\n\n".join(chunk["text"] for chunk in chunks)


def get_sources(docs):
    return list(set([doc.metadata["source"] for doc in docs]))

def augmentation(query,chunks):
    print("[Info]: Generation...")
    context=combine_chunks(chunks)
    template= """<start_of_turn> user
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer: <start_of_turn>model
        """
    prompt = PromptTemplate.from_template(template)
    rag_chain = prompt | llm | StrOutputParser() 
    return rag_chain.invoke({"question":query,"context":context})

def rag_with_sources(question):
    response=croma_db_retriever(question,5)
    docs = response["documents"][0]
    reranked_results=cross_encoder_rerankar(docs, question)
    response=augmentation(question,reranked_results)
    print("[Info]: Generated")
    return response, reranked_results