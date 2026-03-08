import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import Ollama

VECTOR_DB_DIR = "data/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)


# The following is to ensure the AI stays focused on transit data and doesn't hallucinate.
template = """
You are a Transit Policy Assistant for the Pioneer Valley. 
Use the following pieces of retrieved context from the 2026-2030 Transportation Improvement Program (TIP) to answer the user's question. 

Context: {context}

Question: {question}

Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Initialize the LLM and the QA Chain
llm = Ollama(model="llama3") 

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}), # Get top 3 most relevant chunks
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

def ask_transit_bot(query):
    print(f"\n--- Querying: {query} ---")
    result = qa_chain.invoke({"query": query})
    
    print("\nAnswer:", result["result"])
    
    # To show citations:
    print("\nSources Used:")
    for doc in result["source_documents"]:
        print(f"- Page {doc.metadata.get('page', 'Unknown')} from {doc.metadata.get('source')}")

if __name__ == "__main__":
    user_query = input("Ask a question about the 2026 Transit Plan: ")
    ask_transit_bot(user_query)