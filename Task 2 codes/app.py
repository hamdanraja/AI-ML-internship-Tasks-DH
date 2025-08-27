# app.py
import os
import streamlit as st
from dotenv import load_dotenv


from ingest import read_uploaded_files,chunk_documents
from vectorstore_utils import build_vectorstore, make_chain

load_dotenv()


st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")
st.title("💬 Context-Aware RAG Chatbot")
st.caption("Upload documents and chat with memory + retrieval.")


with st.sidebar:
    st.header("Settings")
    embedding_choice = st.radio("Embeddings backend", ["HuggingFace (local)", "OpenAI"], index=0)
    llm_choice = st.radio("LLM backend", ["OpenAI", "HuggingFace (local demo)"], index=0)
    k = st.slider("Top-K documents", 1, 10, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    uploaded = st.file_uploader("Upload PDF/TXT/MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    build_idx = st.button("Build / Rebuild index")


if "chain" not in st.session_state:
      st.session_state.chain = None
if "messages" not in st.session_state:
     st.session_state.messages = []
if "indexed_count" not in st.session_state:
   st.session_state.indexed_count = 0


if build_idx:
       if not uploaded:
            st.warning("Please upload at least one document before building the index.")
       else:
            with st.spinner("Processing files…"):
                raw_docs = read_uploaded_files(uploaded)
                if not raw_docs:
                    st.error("No readable documents found.")
                else:
                    chunks = chunk_documents(raw_docs, chunk_size=800, chunk_overlap=100)
                    device = "cuda" if os.environ.get("USE_CUDA") == "1" else "cpu"
                    emb_backend = "OpenAI" if embedding_choice.startswith("OpenAI") else "HuggingFace"
                    vectorstore = build_vectorstore(chunks, embedding_backend=emb_backend, device=device)
                    llm_backend = "OpenAI" if llm_choice.startswith("OpenAI") else "HF"
                    st.session_state.chain = make_chain(vectorstore, llm_backend=llm_backend, k=k, temperature=temperature)
                    st.session_state.indexed_count = len(chunks)
                    st.success(f"Index built with {len(chunks)} chunks.")


for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)


user_q = st.chat_input("Ask something about the uploaded documents...")


if user_q:
    st.session_state.messages.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)


    if st.session_state.chain is None:
        with st.chat_message("assistant"):
            st.warning("No index yet. Upload docs and click Build / Rebuild index.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    res = st.session_state.chain({"question": user_q})
                    answer = res.get("answer") or res.get("result") or "(no answer)"
                    sources = res.get("source_documents", [])


                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for i, d in enumerate(sources, 1):
                                meta = d.metadata or {}
                                src = meta.get("source", "uploaded doc")
                                page = meta.get("page")
                                loc = f" — page {page+1}" if page is not None else ""
                                preview = (d.page_content[:500] + "…") if len(d.page_content) > 500 else d.page_content
                                st.markdown(f"**{i}.** `{src}`{loc}\n\n> {preview}")
                    else:
                        st.caption("No retrieved sources.")
                    st.session_state.messages.append(("assistant", answer))
                except Exception as e:
                        st.error(f"Error generating answer: {e}")


st.sidebar.markdown("---")
st.sidebar.metric("Indexed chunks", st.session_state.indexed_count)