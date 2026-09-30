01-rag
│
├── ingest.py          → PDF → chunks → embeddings → ChromaDB
├── app.py             → question → retrieval → Ollama → answer
├── documents/         → source PDF
├── requirements.txt
└── README.md

Run ingest.py to create the chroma DB
Run app.py to ask qns 