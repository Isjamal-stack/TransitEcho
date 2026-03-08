import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Modern 2026 Modular Imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# UI Setup
st.set_page_config(page_title="TransitEcho", layout="wide")
st.title("🚌 TransitEcho: Pioneer Valley AI Assistant")

# RAG Brain
@st.cache_resource
def load_rag_system():
    # Uses the data processed in ingest.py
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)
    llm = Ollama(model="llama3")
    
    system_prompt = (
        "Use the provided context to answer the user question. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    #This builds the chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vector_db.as_retriever(), question_answer_chain)

try:
    rag_chain = load_rag_system()
    
    query = st.text_input("Ask about regional transit plans (2026-2030):")
    if query:
        with st.spinner("Analyzing..."):
            response = rag_chain.invoke({"input": query})
            st.subheader("Analysis")
            st.write(response["answer"])
            
            with st.expander("View Citations"):
                for doc in response["context"]:
                    st.write(f"Source: {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")
except Exception as e:
    st.error(f"System Load Error: {e}")