import os
import sys


print(" Script Started ")

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    print("--- Libraries Loaded Successfully ---")
except ImportError as e:
    print(f"--- Import Error: {e} ---")
    sys.exit(1)

DOCS_DIR = "data/transit_docs"
VECTOR_DB_DIR = "data/chroma_db"

def run_ingestion():
    # Check if directory exists and has files
    if not os.path.exists(DOCS_DIR):
        print(f"ERROR: The directory {DOCS_DIR} does not exist!")
        return
    
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    print(f"Found {len(files)} PDF files: {files}")
    
    if not files:
        print("No PDF files found to process. Exiting.")
        return

    documents = []
    for file in files:
        file_path = os.path.join(DOCS_DIR, file)
        print(f"Loading {file}...")
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    
    print(f"Successfully loaded {len(documents)} pages.")

    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    
    print("Initializing Embedding Model (this may take a minute on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Generating Vector DB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    print(f"DONE! Vector DB saved to {VECTOR_DB_DIR}")

if __name__ == "__main__":
    run_ingestion()