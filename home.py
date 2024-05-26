import streamlit as st
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
from rag_pipeline import rag_with_sources

# APIs Configs
from config import set_environment
set_environment()
#Logging
import logging
logging.basicConfig()

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.set_page_config(page_title="FinanceFusionRAG", page_icon="")
st.title("FinanceFusionRAG")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response, reranked_results = rag_with_sources(prompt)
        response = st.write_stream(response_generator(response))
        st.write(reranked_results)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})