# ---------------------------------------------------------
# ingest.py
# ---------------------------------------------------------
# This program builds our RAG knowledge base.
#
# It performs:
#
# PDF
#  ↓
# Extract text
#  ↓
# Split into chunks
#  ↓
# Create embeddings
#  ↓
# Store everything in ChromaDB
#
# This program is NOT responsible for answering questions.
# That job belongs to app.py.
# ---------------------------------------------------------
# ID: hand_hygiene_0
# Text: "...."
# Vector: [0.021, -0.184, 0.093, ...]


# ID: hand_hygiene_1
# Text: "...."
# Vector: [0.114, -0.052, 0.231, ...]


# ID: hand_hygiene_2
# Text: "...."
# Vector: [0.087, -0.194, 0.102, ...]

# Path lets us work with files and folders.
from pathlib import Path

# PyMuPDF is imported as "fitz".
# It allows us to open PDF files and extract their text.
import fitz

# Ollama gives us access to our local embedding model.
import ollama

# ChromaDB is our vector database.
import chromadb


# ---------------------------------------------------------
# 1. LOCATE THE PDF
# ---------------------------------------------------------

# Create a Path pointing to our PDF.
pdf_path = Path("documents/HR-Policy.pdf")


# ---------------------------------------------------------
# 2. OPEN THE PDF
# ---------------------------------------------------------

# Open the PDF using PyMuPDF.
pdf = fitz.open(pdf_path)


# ---------------------------------------------------------
# 3. EXTRACT TEXT FROM EVERY PAGE
# ---------------------------------------------------------

# Create an empty list.
# We will store the text from each page here.
pages = []


# Loop through every page in the PDF.
for page_number, page in enumerate(pdf):

    # Extract all text from the current page.
    text = page.get_text()

    # Remove unnecessary whitespace at the beginning
    # and end of the extracted text.
    text = text.strip()

    # Only keep the page if it actually contains text.
    if text:

        # Store both the page number and the text.
        pages.append(
            {
                "page_number": page_number + 1,
                "text": text
            }
        )


# Close the PDF because we have finished reading it.
pdf.close()


# ---------------------------------------------------------
# 4. COMBINE THE PAGE TEXT
# ---------------------------------------------------------

# Create one large string containing the complete document.
#
# We put two newline characters between pages so that
# there is a clear separation between them.
full_text = "\n\n".join(
    page["text"]
    for page in pages
)


# ---------------------------------------------------------
# 5. CREATE CHUNKS
# ---------------------------------------------------------

# For now, we will use a simple character-based chunker.
#
# Each chunk will contain approximately 1,000 characters.
chunk_size = 1000

# We will overlap neighboring chunks by 200 characters.
#
# This means the end of one chunk is repeated at the
# beginning of the next chunk.
#
# Why?
#
# Imagine an important sentence starts near the end of
# Chunk 1. Without overlap, the surrounding information
# could be separated from it.
chunk_overlap = 200


# Create an empty list to store our chunks.
chunks = []


# Start at the beginning of the document.
start = 0


# Continue creating chunks until we reach the end.
while start < len(full_text):

    # Calculate where this chunk should end.
    end = start + chunk_size

    # Extract the current chunk.
    chunk = full_text[start:end]

    # Remove unnecessary whitespace.
    chunk = chunk.strip()

    # Only add non-empty chunks.
    if chunk:

        # Store the chunk.
        chunks.append(chunk)

    # Move forward by chunk_size minus overlap.
    #
    # Example:
    #
    # chunk_size = 1000
    # overlap = 200
    #
    # First chunk:
    # characters 0 → 1000
    #
    # Second chunk:
    # characters 800 → 1800
    #
    # So 200 characters overlap.
    start += chunk_size - chunk_overlap

# ---------------------------------------------------------
# 6. INSPECT A SPECIFIC CHUNK
# ---------------------------------------------------------

# Tell the user how many chunks were created.
print(f"\nTotal chunks created: {len(chunks)}")

# Ask the user which chunk they want to inspect.
#
# If they simply press Enter, we skip inspection.
chunk_input = input(
    "\nEnter a chunk number to inspect (or press Enter to skip): "
)


# Check whether the user entered something.
if chunk_input.strip():

    # Convert the user's input from text into an integer.
    chunk_number = int(chunk_input)

    # Make sure the requested chunk actually exists.
    if 0 <= chunk_number < len(chunks):

        # Print a visual separator.
        print("\n" + "=" * 70)

        # Show which chunk we're displaying.
        print(f"CHUNK {chunk_number}")

        # Print another separator.
        print("=" * 70)

        # Print the actual text inside the chunk.
        print(chunks[chunk_number])

        # Print the size of the chunk.
        print("\n" + "-" * 70)
        print(
            f"Characters in this chunk: "
            f"{len(chunks[chunk_number])}"
        )

        # Print the separator again.
        print("=" * 70)

    else:

        # Tell the user if they entered an invalid chunk number.
        print(
            f"Invalid chunk number. "
            f"Please choose between 0 and {len(chunks) - 1}."
        )
# ---------------------------------------------------------
# 7. CONNECT TO CHROMADB
# ---------------------------------------------------------

# Open our persistent ChromaDB database.
#
# The database will be stored inside:
#
# 01-rag/chroma_db/
#
client = chromadb.PersistentClient(
    path="./chroma_db"
)


# ---------------------------------------------------------
# 8. CREATE OR OPEN OUR COLLECTION
# ---------------------------------------------------------

# Open the collection used for our documents.
collection = client.get_or_create_collection(
    name="company_policies"
)


# ---------------------------------------------------------
# 9. EMBED AND STORE EACH CHUNK
# ---------------------------------------------------------

# Loop through every chunk.
for i, chunk in enumerate(chunks):

    # Create a unique ID for this chunk.
    chunk_id = f"hand_hygiene_{i}"

    # Convert the chunk into a numerical vector.
    embedding_response = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )

    # Extract the actual embedding vector.
    embedding = embedding_response["embeddings"][0]

    # Insert or update the chunk in ChromaDB.
    #
    # upsert means:
    #
    # If the ID doesn't exist → INSERT
    #
    # If the ID already exists → UPDATE
    collection.upsert(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[chunk]
    )


# ---------------------------------------------------------
# 10. SHOW INGESTION RESULTS
# ---------------------------------------------------------

# Print how many pages were extracted.
print(f"Pages extracted: {len(pages)}")

# Print how many chunks were created.
print(f"Chunks created: {len(chunks)}")

# Print how many chunks currently exist in ChromaDB.
print(f"Chunks in database: {collection.count()}")

# Tell us that ingestion has finished.
print("Ingestion complete.")



# for i, chunk in enumerate(chunks):

# takes one chunk at a time.


# response = ollama.embed(
#     model="nomic-embed-text",
#     input=chunk
# )

# does:

# "Employees receive 24 days..."
#               ↓
#       nomic-embed-text
#               ↓
# [0.02, -0.13, 0.74, ...]

# Then:

# collection.add(
#     ids=[f"chunk_{i}"],
#     embeddings=[embedding],
#     documents=[chunk]
# )

# stores them together:

# ┌─────────────────────────────────────────────┐
# │                 ChromaDB                    │
# │                                             │
# │ ID:        chunk_1                          │
# │ Embedding: [0.02, -0.13, 0.74, ...]         │
# │ Document:  "Employees receive 24 days..."   │
# │                                             │
# └─────────────────────────────────────────────┘

# This relationship is the heart of vector search.

# Later, we'll give ChromaDB a question such as:

# "How many vacation days do I get?"

# ChromaDB will compare the question's embedding against these stored embeddings and return the most semantically similar chunks.

# That's when we'll finally perform the R — Retrieval in RAG.

# One important distinction

# Chunking ≠ embeddings.

# Chunking:

# "Break this document into manageable pieces."

# Embedding:

# "Convert each piece into numbers that represent its semantic meaning."


# Text → numerical vectors
#         ↓
# ChromaDB
#         ↓
# Find relevant information
#         ↓
# gemma3:1b
#         ↓
# Generate human-readable answer