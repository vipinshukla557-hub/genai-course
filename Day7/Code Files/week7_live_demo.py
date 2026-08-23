"""
Week 7 Live Demo: Embeddings + Vector Databases (Compact)
Applied GenAI Engineering Program | GenAIEducate

Setup: pip install sentence-transformers faiss-cpu chromadb scikit-learn numpy
Data:  week7_data.json (same folder)
"""

# json — Loads week7_data.json (the 20 sentences, categories, and 50 movie descriptions). Keeps data separate from code so the demo file stays clean.

# os — Used for os.walk() and os.path when we explore what ChromaDB actually stores on disk (the folder structure with .sqlite3 and .bin files).

# shutil — shutil.rmtree() deletes the ChromaDB folders (chroma_demo, chroma_movies) at the start of each run so you get a clean slate. Without this, re-running would try to add duplicate IDs.

# time — time.sleep(0.3) gives Windows a moment to release file handles after we del col, client. ChromaDB on Windows holds locks on its .bin files — without the sleep, the next PersistentClient() call can fail.

# numpy as np — FAISS requires embeddings as float32 numpy arrays. We use embeddings.astype("float32") before adding to the FAISS index. sentence-transformers returns numpy arrays by default, so numpy is already in play behind the scenes.

# faiss — Facebook AI Similarity Search. Creates the in-memory vector index (IndexFlatL2), adds embeddings, and runs nearest-neighbour search. The "calculator" — fast but no persistence or metadata.

# chromadb — The persistent vector database. Stores embeddings to disk, handles metadata filtering, auto-embeds text so you don't need to call model.encode() manually. The "real database" counterpart to FAISS.

# SentenceTransformer (from sentence_transformers) — Loads the all-MiniLM-L6-v2 model that converts text → 384-dim vectors. This is the actual embedding engine. One call to model.encode() takes a list of strings and returns a numpy array of vectors.

# cosine_similarity (from sklearn.metrics.pairwise) — Measures the angle between two vectors, returning a score from -1 to 1. We use it in Part 2 to show that "movie vs movie" scores higher than "movie vs food" — proving embeddings capture meaning.

import json, os, shutil, time, sys
from pathlib import Path

import numpy as np, faiss, chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DEMO_DIR = BASE_DIR / "chroma_demo"
CHROMA_MOVIES_DIR = BASE_DIR / "chroma_movies"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# We load all our data from a separate JSON file to keep this code file clean.
# The JSON has 20 sentences (5 each for movies, food, sports, tech),
# a category label for each sentence, and 50 movie descriptions with metadata.
with open(BASE_DIR / "week7_data.json", encoding="utf-8") as f:
    data = json.load(f)
sentences, categories, movies = data["sentences"], data["categories"], data["movies"]

# Load the embedding model. "all-MiniLM-L6-v2" is a free, open-source model
# from sentence-transformers. It runs 100% locally — no API key, no internet,
# no cost. First run downloads ~90 MB, then it's cached on your machine forever.
# This model converts any text into a vector of 384 numbers.
model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================================================
# PART 1: Generate Embeddings
# ==========================================================================
# This is the core concept. We take plain text and convert it into numbers.
# model.encode() takes a list of sentences and returns a numpy array.
# Each sentence — no matter how long or short — becomes exactly 384 numbers.
# These numbers capture the MEANING of the text, not the words themselves.
# "cat" and "kitten" will have similar numbers. "cat" and "stock market" won't.
# ==========================================================================

embeddings = model.encode(sentences)

# Shape will be (20, 384) — 20 sentences, each represented by 384 numbers.
# This is what makes embeddings powerful: meaning becomes math.
print(f"Embedded {len(sentences)} sentences → {embeddings.shape}")

# Let's look at actual numbers. These don't have human-readable meaning
# individually, but the PATTERN of numbers captures semantic meaning.
print(f"First 5 values: {embeddings[0][:5].round(4)}\n")


# ==========================================================================
# PART 2: Cosine Similarity
# ==========================================================================
# Now that text is numbers, we can MEASURE how similar two texts are.
# Cosine similarity measures the angle between two vectors:
#   1.0  = identical meaning (same direction)
#   0.0  = no relationship (perpendicular)
#  -1.0  = opposite meaning (rare with text)
# In practice, text pairs fall between 0.3 (unrelated) and 0.95 (paraphrase).
# We use cosine instead of Euclidean distance because cosine ignores magnitude
# — it only cares about direction, which is where the meaning lives.
# ==========================================================================

# Sentence 0 = "The action movie had incredible fight scenes and explosions"
# Sentence 1 = "A heartwarming story about a family reuniting after years apart"
# Both are about movies → should score higher than cross-topic pairs.
print(f"Movie vs Movie:  {cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]:.4f}")

# Sentence 0 (movie) vs Sentence 5 (Italian pasta) — completely different topics.
# This should score much lower, proving embeddings capture topic similarity.
print(f"Movie vs Food:   {cosine_similarity([embeddings[0]], [embeddings[5]])[0][0]:.4f}")

# Any sentence compared to itself is always exactly 1.0 — good sanity check.
print(f"Self vs Self:    {cosine_similarity([embeddings[0]], [embeddings[0]])[0][0]:.4f}\n")


# ==========================================================================
# PART 3: FAISS — Fast Local Vector Search
# ==========================================================================
# FAISS = Facebook AI Similarity Search, built by Meta.
# Instead of comparing a query against every embedding one by one,
# FAISS builds an INDEX — an organized structure that makes search instant.
# Think of it like a library's catalog vs reading every book to find one.
#
# IndexFlatL2 = simplest index type, compares using L2 (Euclidean) distance.
# 384 = dimension must match our embedding model's output size.
#
# Key limitations of FAISS:
#   - Returns indices (positions), NOT the actual text — you map back yourself
#   - No metadata filtering (can't say "only search movies from 2024")
#   - No built-in persistence — data is gone when script ends unless you save
# Think of FAISS as a calculator: fast, simple, no storage.
# ==========================================================================

index = faiss.IndexFlatL2(384)

# .add() loads all our embeddings into the index. FAISS requires float32.
index.add(embeddings.astype("float32"))

# To search: encode the query with the SAME model (critical rule!),
# then call .search() with k = number of results we want.
# Returns two arrays: D = distances (lower = more similar), I = indices.
D, I = index.search(model.encode(["best action movies"]).astype("float32"), k=3)

# Notice: FAISS gives us indices [0, 2, 1] not the text itself.
# We have to look up sentences[index] to get the actual content.
print("FAISS — 'best action movies':")
print(f"  1. {sentences[I[0][0]]}")
print(f"  2. {sentences[I[0][1]]}")
print(f"  3. {sentences[I[0][2]]}\n")


# ==========================================================================
# PART 4: ChromaDB — Persistent Vector Database
# ==========================================================================
# ChromaDB solves FAISS's three problems: no persistence, no metadata, no documents.
#
# Client types (why we pick PersistentClient):
#   Client()            = in-memory only, data gone on exit (like FAISS)
#   PersistentClient()  = saves to disk folder, survives restarts ← we use this
#   HttpClient()        = connects to a server (production setup)
#
# SQL mental model:
#   Collection ≈ Table  |  .add() ≈ INSERT  |  .get() ≈ SELECT
#   .query() ≈ SELECT ... ORDER BY similarity  |  where={} ≈ WHERE clause
# ==========================================================================

# Clean up leftover ChromaDB folders from previous runs.
# We only clean at the START of the script — not at the end.
# Why? On Windows, ChromaDB keeps file handles open while Python runs,
# so shutil.rmtree fails at the end. By cleaning at the start (before
# any client is created), the files are unlocked and cleanup works fine.
for p in [CHROMA_DEMO_DIR, CHROMA_MOVIES_DIR]:
    if os.path.exists(p):
        shutil.rmtree(p, ignore_errors=True)

# Creates a "chroma_demo" folder on disk with SQLite + binary vector files.
client = chromadb.PersistentClient(path=str(CHROMA_DEMO_DIR))
col = client.get_or_create_collection("my_sentences")  # like CREATE TABLE IF NOT EXISTS

# .add() takes raw text — ChromaDB calls the embedding model internally.
# No need for model.encode() like we did with FAISS. Less code, fewer bugs.
col.add(documents=sentences, ids=[f"id_{i}" for i in range(len(sentences))])

# ---- SQL-style operations ----
# SELECT COUNT(*)
print(f"Documents in collection: {col.count()}")

# SELECT * WHERE id IN (...)  — primary key lookup
specific = col.get(ids=["id_0", "id_5"])
print(f"Get by ID: {specific['documents'][0][:60]}...")

# SELECT ... ORDER BY similarity — the superpower no SQL database has natively
results = col.query(query_texts=["best action movies"], n_results=3)
print(f"\nSemantic search — 'best action movies':")
for i, doc in enumerate(results["documents"][0]):
    dist = results["distances"][0][i]  # lower distance = more similar
    print(f"  {i+1}. (dist={dist:.4f}) {doc[:70]}...")


# ---- What does ChromaDB store on disk? ----
# Let's look inside the folder so you know where your data actually lives.
print("\n--- What's inside the ChromaDB folder? ---")
for root, dirs, files in os.walk(CHROMA_DEMO_DIR):
    level = str(root).replace(str(CHROMA_DEMO_DIR), "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root) or 'chroma_demo/'}/")
    for file in files:
        size_kb = os.path.getsize(os.path.join(root, file)) / 1024
        print(f"{indent}  {file} ({size_kb:.1f} KB)")
# Key files:
#   chroma.sqlite3  — metadata (IDs, documents, key-values). Regular SQLite.
#   data_level0.bin — the actual 384-dim vectors in binary (grows with data).
#   header.bin      — index config (dimension, distance metric).

# ---- Proving persistence ----
# Close and reopen — data survives. FAISS would lose everything here.
del col, client
time.sleep(0.3)
client = chromadb.PersistentClient(path=str(CHROMA_DEMO_DIR))
col = client.get_or_create_collection("my_sentences")
print(f"\nReopened after closing — still has {col.count()} docs!")
print("(FAISS would have lost everything here)\n")
del col, client
time.sleep(0.3)


# ==========================================================================
# PART 5: Metadata Filtering
# ==========================================================================
# Metadata = extra tags on each document (category, date, source).
# Think of it as: semantic search finds MEANING, metadata filters by CATEGORY.
#
# Real-world example: 10,000 support tickets in ChromaDB.
#   Without filter → "reset password" searches everything
#   With where={"dept": "IT"} → only searches IT tickets
#
# Filter operators (like SQL WHERE):
#   {"category": "movies"}                   → = 'movies'
#   {"year": {"$gt": 2020}}                  → > 2020
#   {"year": {"$gte": 2020}}                 → >= 2020
#   {"category": {"$in": ["movies","food"]}} → IN ('movies','food')
#   {"$and": [{"genre": "Action"}, {"year": {"$gt": 2015}}]}  → AND
# ==========================================================================

client = chromadb.PersistentClient(path=str(CHROMA_DEMO_DIR))
col = client.get_or_create_collection("with_metadata")

# Each sentence gets a category tag — like adding a column to your table.
col.add(documents=sentences, ids=[f"id_{i}" for i in range(len(sentences))],
        metadatas=[{"category": c} for c in categories])

# Without filter vs with filter — same query, different results.
r1 = col.query(query_texts=["exciting story"], n_results=3)
r2 = col.query(query_texts=["exciting story"], n_results=3, where={"category": "movies"})

print("Query: 'exciting story'")
print(f"  No filter    → {[m['category'] for m in r1['metadatas'][0]]}")
print(f"  Movies only  → {[m['category'] for m in r2['metadatas'][0]]}")

# .get() with where= is a plain filter, no similarity search.
# Like: SELECT * FROM sentences WHERE category = 'tech'
tech_docs = col.get(where={"category": "tech"})
print(f"\n  All tech docs ({len(tech_docs['ids'])} found):")
for doc in tech_docs["documents"]:
    print(f"    - {doc[:65]}...")
print()
del col, client
time.sleep(0.3)


# ==========================================================================
# PART 6: Semantic Search — 50 Movies
# ==========================================================================
# The capstone: keyword search vs semantic search on 50 real movies.
#
# Keyword: does the text literally contain the search words?
# Semantic: does the text MEAN something similar?
#
# This difference is the foundation of RAG (next week).
# ==========================================================================

client = chromadb.PersistentClient(path=str(CHROMA_MOVIES_DIR))
col = client.get_or_create_collection("movies")

# ChromaDB embeds the DOCUMENT (description), not the metadata.
# Metadata (title, genre, year) is for filtering and display only.
col.add(documents=[m["description"] for m in movies],
        ids=[f"movie_{i}" for i in range(len(movies))],
        metadatas=[{"title": m["title"], "genre": m["genre"], "year": m["year"]}
                   for m in movies])

# ---- Semantic queries with natural language ----
for q in ["space travel and survival", "romantic comedy in New York"]:
    r = col.query(query_texts=[q], n_results=3)
    print(f"'{q}':")
    for meta in r["metadatas"][0]:
        print(f"  - {meta['title']} ({meta['genre']}, {meta['year']})")
    print()

# ---- Semantic + metadata filter combined ----
# SQL: SELECT * FROM movies WHERE genre='Drama' ORDER BY similarity('crime')
r = col.query(query_texts=["crime and justice"], n_results=3,
              where={"genre": "Drama"})
print("'crime and justice' (Drama only):")
for meta in r["metadatas"][0]:
    print(f"  - {meta['title']} ({meta['year']})")

# ---- THE BIG COMPARISON — Climax ----
print("\n" + "=" * 50)
print("  KEYWORD SEARCH  vs  SEMANTIC SEARCH")
print("=" * 50)

# Keyword: checks if "space travel" appears literally. Result: NOTHING.
print(f"\nKeyword 'space travel':  {[m['title'] for m in movies if 'space travel' in m['description'].lower()] or 'NO RESULTS'}")

# Semantic: understands "astronaut stranded on Mars" IS about space travel.
r = col.query(query_texts=["space travel"], n_results=3)
print(f"Semantic 'space travel': {[m['title'] for m in r['metadatas'][0]]}")

# Note: We don't delete the chroma folders here — Windows locks the files
# while Python is still running. The cleanup at the TOP of this script
# handles it on the next run. This is a known Windows + ChromaDB quirk.
# On Mac/Linux, cleanup works fine at the end too.

print("\nDone! Next week: retrieval + LLM = RAG")
