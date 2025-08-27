import os
import tempfile
from typing import List
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Load environment variables
load_dotenv()

def read_uploaded_files(uploaded_files) -> List[Document]:
    docs: List[Document] = []
    for uf in uploaded_files:
        suffix = os.path.splitext(uf.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uf.read())   # ✅ fixed
            temp_file_path = temp_file.name
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif suffix in [".txt", ".md"]:
                loader = TextLoader(temp_file_path, encoding="utf-8")
            else:
                st.warning(f"Unsupported file type: {suffix}")
                continue
            docs.extend(loader.load())
        finally:
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
    return docs

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)
