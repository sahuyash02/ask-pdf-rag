# import required libraries and modules
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# open source llama model using ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import UnstructuredURLLoader

from dotenv import load_dotenv
load_dotenv()


# function to take pdf and return vector store
def pdf_to_vector_store(file_name):
    pdf_loader = PyPDFLoader(file_name)
    pdf_docs = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    documents = text_splitter.split_documents(pdf_docs)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)
    return vector_store


# function to take url and return vector store
def url_to_vector_store(url):
    url_loader = UnstructuredURLLoader(urls=[url])
    url_docs = url_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    documents = text_splitter.split_documents(url_docs)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)
    return vector_store


# function to take text file and return vector store
def text_file_to_vector_store(file_name):
    text_loader = TextLoader(file_name)
    text_docs = text_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    documents = text_splitter.split_documents(text_docs)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)
    return vector_store


# function to take user query and return response
def answer_query(vector_store, query):
    # define the llm model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    retrieved_docs = vector_store.similarity_search(query, top_k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"Use the following pieces of context to answer the question at the end. If question is about summary or detail of document then give the summary of document but if question is not about summary and you don't know the answer, just say you don't know, don't try to make up an answer.\n{context}\nQuestion: {query}"

    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error: {e}"
