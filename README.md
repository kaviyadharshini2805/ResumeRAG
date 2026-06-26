# Resume RAG System

A Retrieval-Augmented Generation (RAG) system built from scratch using Python that answers questions about resume documents using semantic search and Large Language Models (LLMs). The application retrieves the most relevant information from a resume before generating accurate, context-aware responses.

---

## Overview

This project demonstrates the complete RAG pipeline, including document processing, text embedding, vector similarity search, and LLM-based answer generation. Instead of sending the entire document to the language model, the system retrieves only the most relevant sections, improving both response quality and efficiency.

---

## Features

* Extracts text from PDF resumes
* Splits text into overlapping chunks
* Generates sentence embeddings using Sentence Transformers
* Stores embeddings in a FAISS vector database
* Performs semantic similarity search
* Retrieves the Top-K relevant document chunks
* Generates context-aware answers using Gemini or OpenAI
* Modular and easy-to-understand implementation

---

## Architecture

```text
                PDF Resume
                     │
                     ▼
            Text Extraction (PyPDF)
                     │
                     ▼
        Recursive Text Chunking
                     │
                     ▼
 Sentence Embeddings (Sentence Transformers)
                     │
                     ▼
        FAISS Vector Database
                     │
                     ▼
             User Question
                     │
                     ▼
           Query Embedding
                     │
                     ▼
      Top-K Similarity Search
                     │
                     ▼
      Relevant Context Retrieval
                     │
                     ▼
     Gemini / OpenAI Response
```

---

## Tech Stack

* Python
* PyPDF
* Sentence Transformers
* FAISS
* NumPy
* Google Gemini API / OpenAI API

---

## Project Structure

```text
ResumeRAG/
│
├── rag.py
├── resume.pdf
├── requirements.txt
├── README.md
└── vector_store/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kaviyadharshini2805/ResumeRAG.git
```

Navigate to the project directory:

```bash
cd ResumeRAG
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## API Configuration

### Using Google Gemini

Obtain an API key from Google AI Studio.

Configure the API key in your code:

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
```

### Using OpenAI

Alternatively, configure your OpenAI API key:

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")
```

---

## Running the Project

Execute the application:

```bash
python rag.py
```

---

## Sample Questions

You can ask questions such as:

* What skills does the candidate have?
* What projects are mentioned in the resume?
* What is the candidate's work experience?
* What programming languages does the candidate know?
* Summarize the candidate's profile.
* What machine learning technologies has the candidate worked with?

---

## Example

**Question**

```text
What projects has the candidate worked on?
```

**Answer**

```text
The candidate has worked on:

• Resume RAG Assistant
• Agri Chatbot
• Pothole Detection System
```

---

## Future Improvements

* Support multiple PDF documents
* Hybrid search (semantic + keyword)
* Web-based interface using Streamlit
* Chat history and conversational memory
* Metadata filtering
* Support for additional vector databases such as ChromaDB, Pinecone, and Weaviate
* Docker deployment
* Cloud deployment

---

## Learning Outcomes

This project helped build practical knowledge of:

* Retrieval-Augmented Generation (RAG)
* Sentence Embeddings
* Vector Databases
* Semantic Search
* Information Retrieval
* Prompt Engineering
* Large Language Models (LLMs)
* FAISS Indexing
* Document Processing Pipelines

---

## License

This project is intended for educational and learning purposes.
