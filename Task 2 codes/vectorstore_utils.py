from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline



def build_vectorstore(docs, embedding_backend: str, device: str = "cpu") -> FAISS:
    if embedding_backend == "OpenAI":
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device}
        )
    return FAISS.from_documents(docs, embeddings)


def get_llm(llm_backend: str, temperature: float = 0.1):
    if llm_backend == "OpenAI":
        return ChatOpenAI(temperature=temperature)

    # Local fallback: FLAN-T5 base
    model_id = "google/flan-t5-base"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    gen = pipeline("text2text-generation", model=mdl, tokenizer=tok, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=gen)


def make_chain(vectorstore: FAISS, llm_backend: str, k: int = 4, temperature: float = 0.1):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    llm = get_llm(llm_backend, temperature=temperature)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain
