import streamlit as st

# open source llama model using ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# using configurations from .env file
llm_model = os.getenv("OLLAMA_MODEL_LLM")
embedding_model = os.getenv("OLLAMA_MODEL_EMBEDDING")
base_url = os.getenv("OLLAMA_BASE_URL")
docs_path = os.getenv("DOCS_FILE")

# Initialize models with the loaded configuration
llm = ChatOllama(
    model=llm_model,
    base_url=base_url
)

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Ollama Chat", layout="centered")

# Load Ollama config
llm_model = os.getenv("OLLAMA_MODEL_LLM")
base_url = os.getenv("OLLAMA_BASE_URL")

# Initialize LLM
llm = ChatOllama(
    model=llm_model,
    base_url=base_url,
    streaming=True
)

import streamlit as st
import random
import time

st.header("Chat App")

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
        with st.spinner("Thinking..."):
            response = llm.invoke(prompt).content
        st.markdown(response)  #
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
