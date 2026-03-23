from pydoc import doc
import streamlit as st
# from main import pdf_to_vector_store, answer_query, url_to_vector_store, text_file_to_vector_store
from main_open_source import answer_query, pdf_to_vector_store, url_to_vector_store, text_file_to_vector_store
from main import answer_query, pdf_to_vector_store, url_to_vector_store, text_file_to_vector_store

st.title("Ask Question from Provided Source")

# session state variable to store vector store 
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None

# session state variable to store all the messages user and rag response
if 'messages' not in st.session_state:
    st.session_state.messages = []


# a function to return RAG response to user query
def response():
    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Getting answer..."):
            response = answer_query(vector_store, query)
        st.write(response)


selected_option = st.selectbox("Select the source of your document", ["PDF File", "Text File", "URL"])

# if the selected document is PDF
if selected_option == "PDF File":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        vector_store = pdf_to_vector_store("temp.pdf")
        st.session_state['vector_store'] = vector_store
        st.success("PDF processed successfully! You can now ask questions about the content.")
        
# if the selected document is text file
elif selected_option == "Text File":
    uploaded_file = st.file_uploader("Upload a Text file", type="txt")
    if uploaded_file is not None:
        with open("temp.txt", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        vector_store = text_file_to_vector_store("temp.txt")
        st.session_state['vector_store'] = vector_store
        st.success("Text file processed successfully! You can now ask questions about the content.")
        
# if the selected type is url
elif selected_option == "URL":
    url = st.text_input("Enter the URL of the document:")
    if not url:
        st.warning("Please provide a URL...")
    else:
        with st.spinner("Processing URL and creating vector store..."):
            vector_store = url_to_vector_store(url)
        st.session_state['vector_store'] = vector_store
        st.success("URL processed successfully! You can now ask questions about the content.")


# if vector store in session state start chat 
if st.session_state['vector_store'] is not None:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Accept user input
    if query := st.chat_input("Enter Question: "):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Getting Answer..."):
                response = answer_query(st.session_state['vector_store'], query)
            st.markdown(response)  #
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    
        
# # Query and its response
# if vector_store:
#     query = st.text_input("Enter your question:")
#     if query:
#         with st.spinner("Getting answer..."):
#             response = answer_query(vector_store, query)
#         st.write(response)

        
        
