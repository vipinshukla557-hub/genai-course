# ─────────────────────────────────────────────────────────────────────────────
# WEEK 8 · SECTION 3: EMBED + STORE IN CHROMADB
# ─────────────────────────────────────────────────────────────────────────────
# Applied GenAI Engineering Program
#
# WHAT THIS FILE COVERS:
#   - Quick recap: what embeddings are (from Week 7)
#   - Embedding all our chunks using HuggingFace sentence-transformers
#   - Storing embedded chunks in ChromaDB with metadata
#   - Verifying the store has our data
#
# PREREQUISITE: Run Section 1 and Section 2 first
#   (to create namaste_foods_handbook.txt)
#
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: RECAP — WHAT IS AN EMBEDDING?
# ─────────────────────────────────────────────────────────────────────────────
# Quick reminder from Week 7:
#
# An embedding converts text into a LIST OF NUMBERS (a vector).
# These numbers capture the MEANING of the text, not just the words.
#
# Example:
#   "I love biryani" → [0.12, -0.45, 0.78, 0.33, ...]  (384 numbers)
#   "Biryani is my favourite food" → [0.11, -0.43, 0.80, 0.31, ...]
#   "The stock market crashed today" → [-0.55, 0.22, -0.10, 0.67, ...]
#
# Notice: the first two vectors are SIMILAR (close numbers) because
# the sentences have similar meaning. The third is DIFFERENT because
# the meaning is completely unrelated.
#
# In RAG, we embed BOTH the document chunks AND the user's question.
# Then we find which chunk embeddings are closest to the question embedding.
# That's the retrieval step!

print("=" * 60)
print("SECTION 3: Embed Chunks + Store in ChromaDB")
print("=" * 60)
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: LOAD AND CHUNK THE DOCUMENT (same as Sections 1-2)
# ─────────────────────────────────────────────────────────────────────────────
# We need to repeat the loading and chunking steps here because each
# section file is self-contained — you should be able to run it on its own.

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the handbook
loader = TextLoader(
    file_path=BASE_DIR / "namaste_foods_handbook.txt",
    encoding="utf-8"
)
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)

print(f"📄 Loaded and split: {len(chunks)} chunks ready to embed")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: SET UP THE EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────────────
# We use HuggingFace's "all-MiniLM-L6-v2" model for embeddings.
#
# WHY THIS MODEL?
#   - FREE — runs locally on your laptop, no API key needed
#   - FAST — small model (80MB), generates embeddings in milliseconds
#   - GOOD ENOUGH — performs well for semantic search tasks
#   - Each text gets converted into 384 numbers (a 384-dimensional vector)
#
# The first time you run this, it downloads the model (~80MB).
# After that, it's cached locally and loads instantly.
#
# IMPORTANT: sentence-transformers requires PyTorch (~2GB download).
# If you couldn't install it, use the OpenAI fallback below.

from langchain_huggingface import HuggingFaceEmbeddings

# Create the embedding model instance
# model_name tells it which pre-trained model to use
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Free, local, fast
)

# ─── FALLBACK: If sentence-transformers didn't install ───────────────
# Uncomment these 3 lines and comment out the 3 lines above:
#
# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# # Requires OPENAI_API_KEY in your .env file. Costs ~$0.02 per 1M tokens.
# ─────────────────────────────────────────────────────────────────────

print(f"✅ Embedding model loaded: all-MiniLM-L6-v2")
print()

# Quick test: embed a single sentence to see what an embedding looks like
test_embedding = embeddings.embed_query("What is the leave policy?")
print(f"Test embedding for 'What is the leave policy?':")
print(f"  Vector length: {len(test_embedding)} dimensions")
print(f"  First 5 values: {test_embedding[:5]}")
print(f"  (each value is a floating-point number between roughly -1 and 1)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: STORE CHUNKS IN CHROMADB
# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB is a vector database — a database designed specifically to
# store and search embedding vectors.
#
# Think of it like this:
#   Regular database (MySQL, PostgreSQL):
#     "Find all employees where department = 'Engineering'"
#     → Exact match search
#
#   Vector database (ChromaDB):
#     "Find the chunks most SIMILAR to 'What is the leave policy?'"
#     → Similarity search based on meaning
#
# LangChain's Chroma wrapper makes this incredibly easy.
# In ONE line, it:
#   1. Takes all your chunks
#   2. Embeds each one (converts text → vector)
#   3. Stores the vector + original text + metadata in ChromaDB
#
# We'll store the database in a local folder so it persists between runs.
# This means you don't have to re-embed every time you restart your script.

import shutil  # shutil lets us delete folders

# shutil is a built-in Python module for file operations like
# copying, moving, and deleting files/folders.
# We use shutil.rmtree() to delete a folder and everything inside it.

from langchain_chroma import Chroma

# Clean up any existing database from previous runs.
# This ensures we start fresh every time for learning purposes.
# In a real project, you'd KEEP the database and only re-index
# when documents change.
db_directory = BASE_DIR / "week8_chroma_db"
if __import__('os').path.exists(db_directory):
    shutil.rmtree(db_directory)
    print(f"🗑️  Cleaned up previous database at '{db_directory}'")

# Chroma.from_documents() does everything in one call:
#   1. Embeds every chunk using our embedding model
#   2. Stores the embedding vector alongside the original text
#   3. Stores the metadata (source file, page number, etc.)
#   4. Saves everything to disk in the specified directory

print(f"⏳ Embedding {len(chunks)} chunks and storing in ChromaDB...")
print(f"   (this may take 10-30 seconds the first time)")
print()

vectorstore = Chroma.from_documents(
    documents=chunks,              # The list of Document objects to store
    embedding=embeddings,          # The embedding model to use
    persist_directory=db_directory  # Where to save the database on disk
)

print(f"✅ ChromaDB created at '{db_directory}'")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: VERIFY WHAT'S IN THE DATABASE
# ─────────────────────────────────────────────────────────────────────────────
# Let's make sure all our chunks are stored properly.

# _collection is the underlying ChromaDB collection object.
# .count() returns the number of items stored in the database.
collection = vectorstore._collection
print(f"📊 Database stats:")
print(f"   Total chunks stored: {collection.count()}")
print()

# Let's peek at the first stored chunk to see what ChromaDB saved
# .peek() returns a sample of the stored data
peek = collection.peek(limit=1)

# peek is a dictionary with these keys:
#   'ids' — unique ID for each chunk
#   'documents' — the original text
#   'metadatas' — the metadata dictionaries
#   'embeddings' — the embedding vectors (lists of numbers)

print("--- Sample stored item ---")
print(f"  ID: {peek['ids'][0]}")
print(f"  Text preview: {peek['documents'][0][:100]}...")
print(f"  Metadata: {peek['metadatas'][0]}")
if peek.get('embeddings') is not None and len(peek['embeddings']) > 0:
    print(f"  Embedding: [{peek['embeddings'][0][0]:.4f}, {peek['embeddings'][0][1]:.4f}, ...] "
          f"({len(peek['embeddings'][0])} dimensions)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: LOADING AN EXISTING DATABASE
# ─────────────────────────────────────────────────────────────────────────────
# Because we used persist_directory, the database is saved to disk.
# In a real app, you'd create the database ONCE when documents are uploaded,
# and then LOAD it every time a user asks a question.
#
# Loading is much faster than creating — no embedding needed.

# This is how you load an EXISTING ChromaDB from disk:
loaded_vectorstore = Chroma(
    persist_directory=db_directory,  # Same directory we saved to
    embedding_function=embeddings    # Same embedding model (critical!)
)

# Note: the parameter name is "embedding_function" when loading,
# but "embedding" when creating with from_documents(). This is a
# LangChain naming inconsistency — just memorize it.

loaded_count = loaded_vectorstore._collection.count()
print(f"✅ Loaded existing database: {loaded_count} chunks")
print(f"   (No re-embedding needed — just loaded from disk!)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# KEY TAKEAWAYS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("KEY TAKEAWAYS — Section 3")
print("=" * 60)
print()
print("1. Embeddings convert text → numbers that capture meaning")
print("2. all-MiniLM-L6-v2 is free, local, and produces 384-dim vectors")
print("3. Chroma.from_documents() embeds + stores in ONE call")
print("4. persist_directory saves the DB to disk — no re-embedding needed")
print("5. ALWAYS use the SAME embedding model for storing and querying")
print()
print("Next: Section 4 — Retrieval (searching the database with a question)")
