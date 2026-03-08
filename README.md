# TransitEcho
A RAG-powered tool for analyzing the Pioneer Valley 2026-2030 Transportation Improvement Program (TIP).


**AI-Driven Policy Analysis for the Pioneer Valley (2026-2030)**

TransitEcho is a Retrieval-Augmented Generation (RAG) assistant designed to make the **265-page FFY 2026-2030 Transportation Improvement Program (TIP)** accessible to the public. 
---

# Project Impact
The regional transit plan is a dense, technical document that dictates millions of dollars in infrastructure spending. TransitEcho democratizes this data, allowing community members to ask natural language questions such as:
* *"What are the electrification goals for the PVTA fleet by 2030?"*
* *"How much funding is allocated for bicycle and pedestrian safety in Amherst?"*

---

# Technical Architecture
This project implements a local, privacy-first AI pipeline designed for high-fidelity technical retrieval:

- **Data Engineering:** Ingested and tokenized 265 pages of PDF documentation into 756 semantic chunks using a `RecursiveCharacterTextSplitter` with context-preserving overlap.
- **Vector Database:** Utilizes **ChromaDB** with `all-MiniLM-L6-v2` embeddings (Sentence-Transformers) for quick semantic retrieval.
- **Inference Engine:** Powered by **Llama 3 (8B)** orchestrated via Ollama, ensuring all data processing remains local and secure.
- **Frontend:** A **Streamlit** dashboard designed for ease of use by non-technical stakeholders.

---

# Getting Started

### Prerequisites
- **Python 3.12:** This project was specifically developed to resolve inter-interpreter version conflicts (e.g., Python 3.9 vs 3.12).
- **Ollama:** Must be running locally with the `llama3` model installed.

### Installation & Setup
1. **Clone the repository:**
   git clone [https://github.com/Isjamal-stack/TransitEcho.git](https://github.com/Isjamal-stack/TransitEcho.git)
   cd TransitEcho

2. **Install Depenedncies:**
   pip install -r requirements.txt

3. **Ingest the data:**
   python ingest.py

4. **Launch the application:**
   streamlit run app.py





This project is licensed under the MIT License—encouraging open-source collaboration for public transit transparency.
