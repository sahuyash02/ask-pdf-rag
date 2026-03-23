from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI  

# open source llama model using ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings

# import vector store and document loaders
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import UnstructuredURLLoader
from bs4 import SoupStrainer

import os
from dotenv import load_dotenv
load_dotenv()

# using configurations from .env file
llm_model = os.getenv("OLLAMA_MODEL_LLM") ## using llama3.2:1b model
embedding_model_name = os.getenv("OLLAMA_MODEL_EMBEDDING")
base_url = os.getenv("OLLAMA_BASE_URL")

# Initialize models with the loaded configuration
llm = ChatOllama(
    model=llm_model,
    base_url=base_url
)

# define embedding model to generate embeddings of chunks
embedding_model = OllamaEmbeddings(
    model=embedding_model_name,
    base_url=base_url
)

# define recusive character text splitter
recursive_char_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]    
)

# function to take pdf and return vector store
def pdf_to_vector_store(file_name):
    pdf_loader = PyPDFLoader(file_name)
    pdf_docs = pdf_loader.load()
    text_splitter = recursive_char_text_splitter
    documents = text_splitter.split_documents(pdf_docs)
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)
    return vector_store

# function to take url and return vector store
def url_to_vector_store(url):
    bs_kwargs = {
        "parse_only": SoupStrainer(
            ["article", "main", "div", "section"]
        )
    }

    url_loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs=bs_kwargs,
        requests_kwargs={
            "headers": {"User-Agent": "Mozilla/5.0"}
        }
    )
    
    url_docs = url_loader.load()
    text_splitter = recursive_char_text_splitter
    documents = text_splitter.split_documents(url_docs)
    # embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)
    return vector_store


# function to take text file and return vector store
def text_file_to_vector_store(file_name):
    text_loader = TextLoader(file_name)
    text_docs = text_loader.load()
    text_splitter = recursive_char_text_splitter
    documents = text_splitter.split_documents(text_docs)
    # embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)
    return vector_store 


# function to take user query and return response
def answer_query(vector_store, query):
    # define the llm model
    # model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    model = llm
    retrieved_docs = vector_store.similarity_search(query, k=3)
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    # prompt = f"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know, don't try to make up an answer.\n{context}\nQuestion: {question}"
    prompt = f"""
    You are a helpful assistant answering questions strictly using the provided context.

    INSTRUCTIONS:
    - Use ONLY the information from the context below.
    - If the question asks for a summary, provide a clear and concise summary based only on the context.
    - If the question asks for specific details, extract and explain those details from the context.
    - If the answer is not present in the context, respond exactly with:
    'Unable to respond to your query based on provided context...'

    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """
    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error: {e}"
    


# --- for testing in terminal ---
# # run the code
# vector_store = pdf_to_vector_store(docs_path)
# query = "What is langchain?"
# response = answer_query(vector_store, query)
# print(response)