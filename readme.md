# 01-RAG

A simple **Retrieval-Augmented Generation (RAG)** application built with Python, Ollama, ChromaDB, and a local embedding model.

The project demonstrates how a PDF can be converted into searchable chunks and used to provide context-aware answers to user questions.

## Architecture

### Document Ingestion

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
nomic-embed-text
 ↓
ChromaDB
```

### Question Answering

```text
User Question
 ↓
nomic-embed-text
 ↓
Similarity Search
 ↓
Retrieved Chunks
 ↓
Gemma 3
 ↓
Answer
```

## Project Structure

```text
01-rag/
│
├── ingest.py              # PDF → chunks → embeddings → ChromaDB
├── app.py                 # Question → retrieval → Ollama → answer
├── documents/             # Source PDF files
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Technologies Used

* **Python**
* **Ollama** – runs the local LLM and embedding model
* **Gemma 3** – generates answers
* **nomic-embed-text** – creates text embeddings
* **ChromaDB** – stores and searches embeddings
* **PyMuPDF** – extracts text from PDF files

## How It Works

### 1. Ingest the PDF

Place your PDF inside the `documents/` folder.

Then run:

```bash
python ingest.py
```

This will:

1. Read the PDF
2. Extract the text
3. Split the text into chunks
4. Generate embeddings using `nomic-embed-text`
5. Store the embeddings and chunks in ChromaDB

### 2. Ask Questions

After ingestion is complete, run:

```bash
python app.py
```

You can then enter questions about the information contained in the PDF.

The application retrieves the most relevant chunks from ChromaDB and provides them to the Gemma 3 model as context.

## Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

On Git Bash:

```bash
source .venv/Scripts/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Make sure Ollama is installed and running, and pull the required models:

```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

## Running the Project

Run these commands in order:

```bash
python ingest.py
python app.py
```

`ingest.py` only needs to be run again when the source documents are changed or new documents are added.

## Learning Goal

This project was built to understand the fundamentals of Retrieval-Augmented Generation (RAG), including:

Document ingestion
Text chunking
Embeddings
Vector databases
Similarity search
Context retrieval
LLM-based answer generation

The project intentionally uses a simple implementation to understand the underlying RAG pipeline before moving to higher-level frameworks and more advanced architectures.
