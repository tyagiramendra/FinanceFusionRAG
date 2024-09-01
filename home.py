import streamlit as st
import time
from rag_pipeline import rag_with_sources

# APIs Configs
from config import setup_langsmith
setup_langsmith()
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
        with st.expander("Sources"):
            st.dataframe(reranked_results)

        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})