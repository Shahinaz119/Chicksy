# 🐥 Chicksy

## AI Broiler Farm Assistant

Chicksy is a Retrieval-Augmented Generation (RAG) application that answers questions about broiler farm management using the ROSS Broiler Management Handbook.

---

## Features

- PDF Processing
- Text Preprocessing
- Document Chunking
- TF-IDF Retrieval
- Sentence Transformer Embeddings
- Hybrid Retrieval
- OpenRouter LLM
- Streamlit Web Application

---

## Project Structure

```
Chicksy/
│
├── data/
├── documents.py
├── preprocessing.py
├── chunking.py
├── vector_representation.py
├── retrieve_context.py
├── prompting.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .env.example
```

---

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```text
OPENROUTER_API_KEY=YOUR_API_KEY
```

---

## Run

```bash
streamlit run streamlit_app.py
```

---

## Technologies

- Python
- Streamlit
- Sentence Transformers
- Scikit-learn
- OpenRouter
- ChromaDB
- PyMuPDF

---

## Retrieval Pipeline

```
PDF
↓

Preprocessing
↓

Chunking
↓

TF-IDF + Embeddings
↓

Hybrid Retrieval
↓

OpenRouter

↓

Answer
```

---

## Author

Developed as an Information Retrieval (IR) course project.