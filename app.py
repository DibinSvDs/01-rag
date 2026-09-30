        

            #RAG ARCHITECTURE
        #             USER
        #              │
        #              │
        #              ▼
        #       "How many leave
        #        days do I get?"
        #              │
        #              ▼
        #      ┌───────────────┐
        #      │   Embedding   │
        #      │     Model     │
        #      └───────┬───────┘
        #              │
        #              ▼
        #         ChromaDB
        #              │
        #       Similarity Search
        #              │
        #              ▼
        #      Relevant Chunks
        #              │
        #              ▼
        #   ┌───────────────────┐
        #   │      Prompt       │
        #   │                   │
        #   │ Context + Question│
        #   └─────────┬─────────┘
        #             │
        #             ▼
        #        ┌─────────┐
        #        │  Gemma  │
        #        │  3:1b   │
        #        └────┬────┘
        #             │
        #             ▼
        #           ANSWER


#We use 2 models here, nomic for retrieval and gemma for generation.
# nomic-embed-text
#        │
#        └── Retrieval

# gemma3:1b
#        │
#        └── Generation


# ---------------------------------------------------------
# app.py
# ---------------------------------------------------------
# This file handles USER QUESTIONS.
#
# IMPORTANT:
# This file does NOT read the original document.
# It does NOT create embeddings for the documents.
# It does NOT ingest anything.
#
# The knowledge base has already been created by ingest.py.
#
# Our job here is only:
#
# Question
#    ↓
# Question embedding
#    ↓
# Search ChromaDB
#    ↓
# Retrieve relevant chunks
#    ↓
# Send context + question to Gemma
#    ↓
# Answer
# ---------------------------------------------------------


# Import ChromaDB so we can connect to our existing
# persistent vector database.
import chromadb

# Import Ollama so we can use:
# 1. nomic-embed-text for embeddings
# 2. gemma3:1b for generating answers
import ollama


# ---------------------------------------------------------
# 1. CONNECT TO THE EXISTING CHROMADB
# ---------------------------------------------------------

# Connect to the ChromaDB database that was created
# earlier by ingest.py.
#
# Notice that we are NOT creating a new database.
# We are opening the existing one stored on disk.
client = chromadb.PersistentClient(path="./chroma_db")


# ---------------------------------------------------------
# 2. OPEN THE EXISTING COLLECTION
# ---------------------------------------------------------

# Open the collection that ingest.py created.
#
# We don't need get_or_create_collection() here because
# the collection should already exist.
collection = client.get_collection(
    name="company_policies"
)


# ---------------------------------------------------------
# 3. ASK THE USER FOR A QUESTION
# ---------------------------------------------------------

# Ask the user to type a question.
question = input("\nAsk a question: ")


# ---------------------------------------------------------
# 4. CREATE AN EMBEDDING FOR THE QUESTION
# ---------------------------------------------------------

# Convert the user's question into a vector.
#
# We MUST use the same embedding model that we used
# during ingestion.
question_response = ollama.embed(
    model="nomic-embed-text",
    input=question
)


# Extract the actual vector from Ollama's response.
question_embedding = question_response["embeddings"][0]


# ---------------------------------------------------------
# 5. SEARCH THE VECTOR DATABASE
# ---------------------------------------------------------

# Search ChromaDB for the 2 chunks that are most
# semantically similar to the user's question.
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)


# ---------------------------------------------------------
# 6. EXTRACT THE RETRIEVED DOCUMENTS
# ---------------------------------------------------------

# ChromaDB returns documents inside a nested list.
#
# results["documents"] looks approximately like:
#
# [
#     [
#         "Employees can carry forward...",
#         "Full-time employees receive..."
#     ]
# ]
#
# We take [0] because we only performed one query.
retrieved_documents = results["documents"][0]

# ---------------------------------------------------------
# SHOW THE RETRIEVED CHUNKS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RETRIEVED CHUNKS")
print("=" * 70)

# Loop through every chunk returned by ChromaDB.
for i, document in enumerate(retrieved_documents):

    print(f"\n--- Retrieved Chunk {i + 1} ---")
    print(document)

print("=" * 70)
# ---------------------------------------------------------
# 7. COMBINE THE RETRIEVED CHUNKS
# ---------------------------------------------------------

# Join the retrieved chunks together into one piece
# of context that we can give to the language model.
context = "\n\n".join(retrieved_documents)


# ---------------------------------------------------------
# 8. CREATE THE PROMPT FOR GEMMA
# ---------------------------------------------------------

# Tell Gemma to answer using the retrieved information.
#
# This is the "generation" part of RAG.
prompt = f"""
You are an employee assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not present in the context,
say that you don't know based on the available information.

Context:
{context}

Question:
{question}
"""


# ---------------------------------------------------------
# 9. ASK GEMMA TO GENERATE THE ANSWER
# ---------------------------------------------------------

# Send our prompt to the local Gemma model.
response = ollama.chat(
    model="gemma3:1b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# ---------------------------------------------------------
# 10. EXTRACT THE ANSWER
# ---------------------------------------------------------

# Ollama returns the generated message inside the response.
answer = response["message"]["content"]


# ---------------------------------------------------------
# 11. DISPLAY THE ANSWER
# ---------------------------------------------------------

print("\nAnswer:")
print(answer)



