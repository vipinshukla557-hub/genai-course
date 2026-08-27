# ─────────────────────────────────────────────────────────────────────────────
# WEEK 8 · SECTION 4: RETRIEVAL — SEARCHING FOR RELEVANT CHUNKS
# ─────────────────────────────────────────────────────────────────────────────
# Applied GenAI Engineering Program
#
# WHAT THIS FILE COVERS:
#   - Similarity search — asking ChromaDB a question
#   - Understanding top-K retrieval (how many chunks to fetch)
#   - Similarity search with scores (how confident is the match?)
#   - Metadata filtering (search only specific sections)
#   - Creating a reusable retriever object
#
# PREREQUISITE: Run Sections 1, 2, and 3 first
#   (to create the document and ChromaDB database)
#
# ─────────────────────────────────────────────────────────────────────────────


print("=" * 60)
print("SECTION 4: Retrieval — Searching for Relevant Chunks")
print("=" * 60)
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD THE EXISTING CHROMADB
# ─────────────────────────────────────────────────────────────────────────────
# We created and saved the database in Section 3.
# Now we just LOAD it — no re-embedding needed.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ─── FALLBACK: If sentence-transformers didn't install ───────────────
# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# ─────────────────────────────────────────────────────────────────────

# Same embedding model we used when storing
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the database from disk
vectorstore = Chroma(
    persist_directory="week8_chroma_db",
    embedding_function=embeddings
)

print(f"✅ Loaded ChromaDB: {vectorstore._collection.count()} chunks")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: BASIC SIMILARITY SEARCH
# ─────────────────────────────────────────────────────────────────────────────
# This is the core of retrieval. You give it a question (in plain English),
# and it finds the chunks whose MEANING is closest to your question.
#
# Under the hood, this is what happens:
#   1. Your question gets EMBEDDED (converted to a 384-number vector)
#   2. ChromaDB compares that vector against ALL stored chunk vectors
#   3. It returns the chunks with the HIGHEST similarity scores
#
# It's like Googling, but instead of matching keywords, it matches MEANING.
# So "What holidays do employees get?" would match a chunk about
# "Public Holidays: 12 gazetted holidays per year..." even though
# the exact word "holidays" might not appear in your question.

question = "How many casual leaves do employees get?"

# similarity_search() returns a list of Document objects.
# k=3 means "give me the top 3 most relevant chunks"
results = vectorstore.similarity_search(
    query=question,  # The question to search for
    k=3              # Number of results to return
)

print(f"🔍 Question: '{question}'")
print(f"   Found {len(results)} relevant chunks:")
print()

for i, doc in enumerate(results):
    print(f"--- Result {i + 1} ---")
    # Show the first 200 characters of each chunk
    preview = doc.page_content[:200].strip().replace("\n", " ")
    print(f"  Preview: {preview}...")
    print(f"  Metadata: {doc.metadata}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: SIMILARITY SEARCH WITH SCORES
# ─────────────────────────────────────────────────────────────────────────────
# Sometimes you want to know HOW similar the results are.
# Are they a strong match (score close to 0) or a weak match (score far)?
#
# similarity_search_with_score() returns both the Document AND a distance score.
#
# IMPORTANT about ChromaDB scores:
#   - ChromaDB uses DISTANCE, not similarity.
#   - LOWER distance = MORE similar (closer in meaning)
#   - Distance of 0 = identical
#   - Distance > 1.5 = probably not relevant
#
# Think of it like physical distance:
#   If two restaurants are 0.1 km apart, they're very close.
#   If they're 2 km apart, they're far.

print("=" * 50)
print("Similarity Search WITH Scores")
print("=" * 50)
print()

question2 = "What is the work from home policy for engineers?"

results_with_scores = vectorstore.similarity_search_with_score(
    query=question2,
    k=4  # Let's get 4 results this time
)

# results_with_scores is a list of TUPLES.
# A tuple is like a pair: (Document, score)
# We unpack each tuple into doc and score using: for doc, score in ...

print(f"🔍 Question: '{question2}'")
print()

for doc, score in results_with_scores:
    preview = doc.page_content[:150].strip().replace("\n", " ")
    # :.4f formats the number to 4 decimal places
    print(f"  Score: {score:.4f} {'(strong match)' if score < 0.8 else '(weaker match)'}")
    print(f"  Preview: {preview}...")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: TRY DIFFERENT QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
# Let's test with several questions to see how well retrieval works.
# Notice how it finds relevant chunks even when the wording is different
# from what's in the document.

print("=" * 50)
print("Testing Multiple Questions")
print("=" * 50)
print()

test_questions = [
    "Can I work from home on Fridays?",
    "How much increment do I get for rating 4?",
    "Is moonlighting allowed?",
    "What is the maternity leave duration?",
    "How much is the meal allowance during travel?",
]

for q in test_questions:
    # Get just the top 1 result for each question
    top_result = vectorstore.similarity_search(query=q, k=1)

    # top_result is a list with 1 item, so we get [0] to get the Document
    preview = top_result[0].page_content[:120].strip().replace("\n", " ")
    print(f"Q: {q}")
    print(f"A: {preview}...")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: METADATA FILTERING
# ─────────────────────────────────────────────────────────────────────────────
# Sometimes you want to search only PART of the database.
# For example: "Only search chunks from page 3" or
# "Only search chunks from the HR policy document, not the finance one."
#
# This is called METADATA FILTERING — you filter BEFORE searching.
#
# Our text file only has one source, so the filter isn't very exciting.
# But in a real app with multiple PDFs, this is very powerful.
# Imagine you have 50 documents loaded. A user asks about "refund policy".
# With filtering, you can restrict the search to only the
# "Customer Support Handbook" instead of searching all 50.
#
# Let's demonstrate with our data:

print("=" * 50)
print("Metadata Filtering")
print("=" * 50)
print()

# Search with a filter — only chunks from our specific source file
filtered_results = vectorstore.similarity_search(
    query="What is the expense policy?",
    k=2,
    filter={"source": "namaste_foods_handbook.txt"}
    # This filter says: only search chunks where the metadata
    # field "source" equals "namaste_foods_handbook.txt"
)

print(f"Filtered search (source = namaste_foods_handbook.txt):")
for doc in filtered_results:
    preview = doc.page_content[:150].strip().replace("\n", " ")
    print(f"  Metadata: {doc.metadata}")
    print(f"  Preview: {preview}...")
    print()

# In a multi-document setup, you could filter like:
# filter={"source": "hr_policy.pdf"}          — only HR docs
# filter={"page": 5}                          — only page 5
# filter={"department": "engineering"}         — only engineering docs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: CREATING A RETRIEVER OBJECT
# ─────────────────────────────────────────────────────────────────────────────
# LangChain has a concept called a "retriever" — it's a wrapper around
# the vector store that makes it easy to plug into chains (LCEL pipes).
#
# Instead of calling vectorstore.similarity_search() every time,
# you create a retriever and pass it to a chain. The chain automatically
# calls the retriever when it needs documents.
#
# This becomes important in Week 9 when we build the full RAG chain.

# .as_retriever() converts the vector store into a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # Use cosine similarity (default)
    search_kwargs={"k": 4}     # Always return top 4 chunks
)

# Now you can call retriever.invoke() with a question
# It returns the same list of Document objects as similarity_search()
retriever_results = retriever.invoke("What are the core values of Namaste Foods?")

print("=" * 50)
print("Using a Retriever Object")
print("=" * 50)
print()
print(f"retriever.invoke() returned {len(retriever_results)} documents")
for i, doc in enumerate(retriever_results):
    preview = doc.page_content[:120].strip().replace("\n", " ")
    print(f"  Result {i+1}: {preview}...")
print()


# ─────────────────────────────────────────────────────────────────────────────
# KEY TAKEAWAYS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("KEY TAKEAWAYS — Section 4")
print("=" * 60)
print()
print("1. similarity_search(query, k) finds the top-K most relevant chunks")
print("2. It matches MEANING, not keywords — 'holidays' matches 'leave policy'")
print("3. Lower distance score = better match (ChromaDB uses distance)")
print("4. Metadata filtering lets you search a SUBSET of your documents")
print("5. .as_retriever() wraps the store for use in LCEL chains")
print()
print("Next: Section 5 — Generation (passing chunks to the LLM for answers)")
