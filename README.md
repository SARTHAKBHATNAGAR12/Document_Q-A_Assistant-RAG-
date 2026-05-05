# Document_Q-A_Assistant-RAG-Book_Sapeins(a book of historer)

# 📚 Sapiens Scholar: Hybrid RAG Document Q&A Bot

An intelligent, context-aware AI Chatbot built using LangChain, HuggingFace, ChromaDB, and Google Gemini. This system allows users to interactively ask questions about the book "Sapiens" (or any large PDF) and receive accurate, context-grounded answers without hallucinations.

![Terminal Demo]() 
<img width="1273" height="597" alt="RAG_Output" src="https://github.com/user-attachments/assets/8f03acc9-0b1b-41f4-89b9-8162bcb3ee26" />


## 🛠️ Tech Stack
* **Framework:** LangChain
* **LLM:** Google Gemini 1.5 Flash (via `langchain-google-genai`)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) for fast, local, and cost-free embedding generation.
* **Vector Database:** ChromaDB (Persistent local storage)
* **Initial Data Source:** AWS S3 (Used for initial PDF retrieval before local vectorization)
* **Language:** Python

## 🧠 Architecture & How We Achieved It

This project uses a **Retrieval-Augmented Generation (RAG)** pipeline. To overcome API rate limits and network timeouts during the processing of a massive 400+ page document, a **Hybrid Approach** was implemented: local models for heavy data processing and cloud models for reasoning.

![Architecture Diagram](<img width="1408" height="768" alt="Architecture" src="https://github.com/user-attachments/assets/edda7552-0619-45ab-aa6f-1915dbda34dc" />

### 1. Data Ingestion & Chunking
* The original PDF was securely fetched from an **AWS S3 Bucket**.
* The document was split into smaller, manageable pieces using LangChain's `RecursiveCharacterTextSplitter` (Chunk size: 1000, Overlap: 200) to maintain context between paragraphs.

### 2. Local Embeddings (Cost Optimization)
* Instead of sending thousands of chunks to a paid cloud API, the text was embedded locally using **HuggingFaceEmbeddings**. 
* This bypassed Google API's free-tier rate limits (`RESOURCE_EXHAUSTED` errors) and processed the entire book in minutes.

### 3. Vector Storage
* The embeddings were stored in a local **ChromaDB** database. 
* A persistence layer (`os.path.exists` check) was implemented. If the database already exists, the system loads it directly in 0.1 seconds, eliminating repetitive AWS S3 data transfer costs and embedding time.

### 4. Retrieval & Generation
* When a user asks a query, the system performs a semantic similarity search in ChromaDB to fetch the top 3 most relevant chunks.
* These chunks are passed as context to the **Google Gemini 1.5 Flash** LLM via a strictly instructed prompt template, ensuring the AI only answers based on the provided text.

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/SARTHAKBHATNAGAR12/Document_Q-A_Assistant-RAG-.git](https://github.com/SARTHAKBHATNAGAR12/Document_Q-A_Assistant-RAG-.git)
   cd Document_Q-A_Assistant-RAG-
   cd langchain-rag-chatbot
