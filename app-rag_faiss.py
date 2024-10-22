import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts.chat import SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.prompts import  ChatPromptTemplate
def get_pdf_content(documents):
    raw_text = ""
    for document in documents:
        pdf_reader = PdfReader(document)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def get_embeddings(chunks):
    model_name = "models/embedding-001"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings =  GoogleGenerativeAIEmbeddings(model=model_name,google_api_key="AIzaSyCPxAO8gzbSYmQwk3bIbTpy_CKzqkSec00")
    vector_storage = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_storage

def start_conversation(vector_embeddings):
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyCPxAO8gzbSYmQwk3bIbTpy_CKzqkSec00")
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    promptTemplate = """Please answer the user question. You can use the chat history: {chat_history} and {context} to answer users' question: {question}.
    If you don't know, please answer considering your knowledge base. Please be polite and answer in english."""

    messages = [
        SystemMessagePromptTemplate.from_template(promptTemplate),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_embeddings.as_retriever(),
        memory=memory,
        get_chat_history=lambda h: h,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    return conversation


def process_query(query_text):
    if st.session_state.conversation is None:
        st.error("Conversation is not initialized.")
        return

    response = st.session_state.conversation({'question': query_text})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"**User:** {message.content}")
        else:
            st.write(f"**Bot:** {message.content}")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:", layout="wide")
    st.header("Hi, I am your PDF ChatBot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.subheader("PDF documents")
        documents = st.file_uploader(
            "Upload your PDF files", type=["pdf"], accept_multiple_files=True
        )
        if st.button("Run"):
            with st.spinner("Processing..."):
                extracted_text = get_pdf_content(documents)
                text_chunks = get_chunks(extracted_text)
                vector_embeddings = get_embeddings(text_chunks)
                st.session_state.conversation = start_conversation(vector_embeddings)
                if st.session_state.conversation is not None:
                    st.success("Conversation initialized successfully.")

    query = st.text_input("How can I help you today?")
    if query:
        process_query(query)

if __name__ == "__main__":
    main()
