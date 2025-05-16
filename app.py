import streamlit as st
import openai
import tempfile
import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from dotenv import load_dotenv
load_dotenv(".env")

OPENAI_API_KEY = os.getenv('OPENAI_KEY')

def load_pdf(uploaded_file):
    """Save uploaded file to a temporary location and load PDF"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    return pages
    
def process_text(pages):
    """Split text into smaller chunks and store in FAISS"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(docs, embeddings)
    
    return vector_store

def get_response(vector_store, query):
    """Retrieve and generate response from vector store"""
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
    retriever = vector_store.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    response = qa_chain.run(query)
    return response

st.title("📄 PDF Chatbot")
st.write("Upload File PDF dan Tanyakan Tentang Isinya !")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Memproses dokumen..."):
        pages = load_pdf(uploaded_file)
        vector_store = process_text(pages)
        st.success("Dokumen telah dipelajari, silahkan bertanya")

    query = st.text_input("Pertanyaan:")

    if query:
        with st.spinner("Memproses jawaban..."):
            response = get_response(vector_store, query)
            st.write("**Jawaban:**", response)