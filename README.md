<<<<<<< HEAD
🧬 Overview

Medi Assistant is an AI-powered medical question-answering assistant designed to provide trustworthy, document-based health information using Retrieval-Augmented Generation (RAG).

The system extracts text from medical PDFs, converts them into embeddings, stores them inside a FAISS vector database, and answers user medical queries using Groq LLaMA 3.1.

=======
# Medi_assistant-
🧬 Overview
Medi Assistant is an AI-powered medical question-answering assistant designed to provide trustworthy, document-based health information using Retrieval-Augmented Generation (RAG).
The system extracts text from medical PDFs, converts them into embeddings, stores them inside a FAISS vector database, and answers user medical queries using Groq LLaMA 3.1.
>>>>>>> 16fb175fa59e30d6106d4f59055dbbea55201d12
It is designed for accuracy, fast inference, multilingual support, and offline document search.


🚀 Features
🔍 RAG-based Medical Q&A
<<<<<<< HEAD

Embeds medical PDFs using HuggingFace MiniLM

Stores vectors with FAISS

Retrieves relevant chunks (top-k search)

Uses Groq LLaMA 3.1 8B for generating accurate answers

💬 Streamlit Chat Interface

Modern, dark-themed UI

Adjustable creativity (temperature), max tokens

Adjustable number of retrieved documents

Lives completely on your machine

📚 Medical Knowledge Base

Load large medical encyclopedias (PDF)

Automated text splitting (500 chars/chunk)

Designed for performance and precision


🙌 Credits

Developed by Bipin Dahat

Powered by:

🧠 LangChain

⚡ Groq LLaMA-3.1

🔎 FAISS

🤗 HuggingFace Embeddings

🖥️ Streamlit
=======
  Embeds medical PDFs using HuggingFace MiniLM
  Stores vectors with FAISS
  Retrieves relevant chunks (top-k search)
  Uses Groq LLaMA 3.1 8B for generating accurate answers

💬 Streamlit Chat Interface
  Modern, dark-themed UI
  Adjustable creativity (temperature), max tokens
  Adjustable number of retrieved documents
  Lives completely on your machine

📚 Medical Knowledge Base
  Load large medical encyclopedias (PDF)
  Automated text splitting (500 chars/chunk)
  Designed for performance and precision

🔬 Architecture Diagram (RAG Workflow)
            ┌────────────────────────┐
            │   Medical PDF Files    │
            └────────────┬───────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │ Text Splitting (500 char chunks)   │
        └────────────┬───────────────────────┘
                     │
                     ▼
            ┌─────────────────────────────┐
            │ MiniLM Embeddings (HF)      │
            └────────────┬────────────────┘
                         │
                         ▼
            ┌─────────────────────────────┐
            │ FAISS Vector Database        │
            └────────────┬────────────────┘
                         │
                         ▼
            ┌───────────────────────────────┐
            │ Groq LLaMA-3.1 8B (Chat API)   │
            └────────────┬──────────────────┘
                         │
                         ▼
            ┌─────────────────────────────┐
            │ Medi Assistant UI (Streamlit)│
            └─────────────────────────────┘



🙌 Credits
  Developed by Bipin Dahat
  Powered by:
  🧠 LangChain
  ⚡ Groq LLaMA-3.1🔎 FAISS
  🤗 HuggingFace Embeddings
  🖥️ Streamlit
>>>>>>> 16fb175fa59e30d6106d4f59055dbbea55201d12
