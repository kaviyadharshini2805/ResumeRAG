# 📄 Resume RAG System (From Scratch)

A simple **Retrieval-Augmented Generation (RAG)** system built from scratch using Python.  
It allows you to ask questions about a resume and get accurate, context-aware answers.

---

## 🚀 Features

- Extracts text from PDF resumes
- Splits text into overlapping chunks
- Generates embeddings using Sentence Transformers
- Stores embeddings in FAISS vector database
- Retrieves most relevant chunks using semantic search
- Uses an LLM (Gemini/OpenAI) to generate final answers

---

## 🧠 How It Works


PDF Resume
↓
Text Extraction (PyPDF)
↓
Chunking (with overlap)
↓
Embeddings (Sentence Transformers)
↓
Vector Storage (FAISS)
↓
Query Embedding
↓
Similarity Search
↓
Top-K Context Retrieval
↓
LLM Response Generation


---

## 🛠️ Tech Stack

- Python
- PyPDF
- Sentence Transformers
- FAISS
- NumPy
- Gemini API / OpenAI API

---

## 📦 Installation

```bash
git clone https://github.com/your-username/resume-rag.git
cd resume-rag

pip install -r requirements.txt

```
🔑 Setup

If using Gemini:

Get API key from Google AI Studio
Add it in your code:
genai.configure(api_key="YOUR_API_KEY")
▶️ Run the Project
python rag.py

### Then ask questions like:

What skills does the candidate have?
What projects are mentioned in the resume?
What is the candidate’s experience?
```
📌 Example Output
Question: What projects has the candidate worked on?
```
### Answer:
- RAG-based PDF Chatbot
- EV Battery Management System
- MSG Detection Kit
