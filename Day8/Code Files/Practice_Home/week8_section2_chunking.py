# ─────────────────────────────────────────────────────────────────────────────
# WEEK 8 · SECTION 2: CHUNKING — SPLITTING DOCUMENTS INTO PIECES
# ─────────────────────────────────────────────────────────────────────────────
# Applied GenAI Engineering Program
#
# WHAT THIS FILE COVERS:
#   - WHY we need to split documents (can't stuff a 100-page PDF into GPT)
#   - RecursiveCharacterTextSplitter — the go-to splitter
#   - chunk_size and chunk_overlap — what they mean and how to choose values
#   - Inspecting chunks to see what the splitter actually does
#
# PREREQUISITE: Run Section 1 first to create namaste_foods_handbook.txt
#
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: WHY DO WE CHUNK?
# ─────────────────────────────────────────────────────────────────────────────
# Imagine you're studying for an exam. You have a 500-page textbook.
# Your friend asks: "What's the formula for compound interest?"
#
# You DON'T read all 500 pages to find the answer.
# You go to the INDEX, find "compound interest → page 127",
# open page 127, and read just THAT section.
#
# That's exactly what RAG does:
#   1. Break the textbook into small sections (CHUNKING)
#   2. When someone asks a question, find the most relevant sections (RETRIEVAL)
#   3. Give only those sections to the LLM to answer from (GENERATION)
#
# Why not just give the WHOLE document to the LLM?
#   Problem 1: Context window limit. GPT-4o-mini has a ~128K token limit.
#              A 500-page PDF could easily be 200K+ tokens. It won't fit.
#   Problem 2: Even if it fits, the LLM gets confused by too much text.
#              It's like asking someone to find a needle in a haystack —
#              the more hay, the harder it is to find the needle.
#   Problem 3: Cost. You pay per token. Sending 100K tokens per question
#              gets expensive FAST.

print("=" * 60)
print("SECTION 2: Chunking — Splitting Documents into Pieces")
print("=" * 60)
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: LOAD OUR SAMPLE DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────
# First, let's load the handbook we created in Section 1.

from langchain_community.document_loaders import TextLoader


loader = TextLoader(
    file_path="C:\\Users\\admin\\NewRepo\\genai-course\\Day8\\Code Files\\Practice_Home\\namaste_foods_handbook.txt",
    encoding="utf-8"
)
docs = loader.load()

# Our text file loaded as one big document
full_text = docs[0].page_content
print(f"📄 Loaded document: {len(full_text):,} characters")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: RecursiveCharacterTextSplitter
# ─────────────────────────────────────────────────────────────────────────────
# This is LangChain's most commonly used text splitter, and for good reason.
#
# HOW IT WORKS:
# It tries to split text in a SMART order. It first tries to split on
# double newlines (\n\n) — those usually mark paragraph boundaries.
# If a chunk is still too big, it tries single newlines (\n).
# Then spaces. Then individual characters as a last resort.
#
# The "Recursive" in the name means it keeps trying different separators
# until the chunks are small enough.
#
# TWO KEY PARAMETERS:
#
# chunk_size = 1000
#   → Each chunk should be approximately 1000 characters long.
#   Think of this as: "how big is each sticky note?"
#
#   Too small (e.g., 100):
#     Each chunk has so little text that it loses context.
#     "Casual Leave: 12 days" — but 12 days for what? For whom?
#
#   Too big (e.g., 5000):
#     Chunks contain too many topics mixed together.
#     The retrieval step might pull in irrelevant information.
#
#   1000 is a good starting point for most documents.
#
# chunk_overlap = 200
#   → The last 200 characters of Chunk 1 are REPEATED at the start of Chunk 2.
#   Think of this as: "the last few lines of one page also appear
#   at the top of the next page."
#
#   WHY? Because sentences don't end neatly at chunk boundaries.
#   Without overlap, you might split a sentence in half:
#     Chunk 1 ends: "...employees are entitled to"
#     Chunk 2 starts: "15 days of earned leave per year."
#   Neither chunk has the complete thought!
#   With overlap, Chunk 2 would start: "...employees are entitled to 15 days
#   of earned leave per year."
#
#   200 is a good default. About 20% of chunk_size.

# In LangChain 1.x, text splitters live in the langchain_text_splitters package.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create the splitter with our chosen settings
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Target size for each chunk (in characters)
    chunk_overlap=200,     # How much text to repeat between chunks
    length_function=len,   # Use Python's len() to measure chunk size
    # length_function tells the splitter HOW to measure text length.
    # len() counts characters. You could also use a token counter,
    # but character count is simpler and good enough for most cases.

    separators=["\n\n", "\n", " ", ""]
    # This is the order it tries to split:
    #   "\n\n" = double newline (paragraph break) — best split point
    #   "\n"   = single newline (line break) — decent split point
    #   " "    = space (between words) — acceptable split point
    #   ""     = character by character — last resort, rarely used
)

print("🔧 Splitter configured:")
print(f"   chunk_size = 1000 characters")
print(f"   chunk_overlap = 200 characters")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: SPLIT THE DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────
# .split_documents() takes a list of Document objects and returns a NEW list
# of smaller Document objects. Each small Document (chunk) inherits the
# metadata from its parent document.

chunks = splitter.split_documents(docs)

# Let's see what happened
print(f"📦 Original: 1 document, {len(full_text):,} characters")
print(f"📦 After splitting: {len(chunks)} chunks")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: INSPECT THE CHUNKS
# ─────────────────────────────────────────────────────────────────────────────
# Let's look at each chunk to understand what the splitter did.
# For each chunk, we'll show:
#   - Its length (in characters)
#   - A preview of the text (first 100 characters)
#   - Its metadata

print("--- All Chunks ---")
print()
for i, chunk in enumerate(chunks):
    # enumerate() gives us the index (i) and the chunk in each iteration
    print(f"Chunk {i + 1} of {len(chunks)}:")
    print(f"  Length: {len(chunk.page_content)} characters")
    print(f"  Metadata: {chunk.metadata}")

    # Show the first 100 characters as a preview.
    # .strip() removes leading/trailing whitespace and newlines.
    # .replace("\n", " ") turns newlines into spaces for cleaner display.
    preview = chunk.page_content[:100].strip().replace("\n", " ")
    print(f"  Preview: {preview}...")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: VERIFY THE OVERLAP
# ─────────────────────────────────────────────────────────────────────────────
# Let's prove that overlap is working by comparing the END of Chunk 1
# with the START of Chunk 2.
#
# The last 200 characters of Chunk 1 should match the first 200 characters
# of Chunk 2 (approximately — the splitter tries to break at clean
# boundaries like newlines, so it won't be exactly 200 characters).

if len(chunks) >= 2:
    print("--- Overlap Verification ---")
    print()

    # Get the last 200 characters of chunk 1
    chunk1_end = chunks[0].page_content[-200:]

    # Get the first 200 characters of chunk 2
    chunk2_start = chunks[1].page_content[:200]

    print("End of Chunk 1 (last 200 chars):")
    print(f"  ...{chunk1_end.strip()}")
    print()
    print("Start of Chunk 2 (first 200 chars):")
    print(f"  {chunk2_start.strip()}...")
    print()

    # Check if there's any common text
    # We'll find the overlap by checking if the end of chunk 1
    # appears at the start of chunk 2
    overlap_found = False
    # Start from the full 200 chars and work down, looking for a match
    for length in range(200, 10, -1):
        # Check if the last 'length' characters of chunk 1
        # match the first 'length' characters of chunk 2
        if chunks[0].page_content[-length:] == chunks[1].page_content[:length]:
            print(f"✅ Found {length} characters of overlap between Chunk 1 and Chunk 2!")
            overlap_found = True
            break
    if not overlap_found:
        print("ℹ️  Overlap exists but the splitter adjusted boundaries for clean breaks.")


print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: EXPERIMENT WITH DIFFERENT chunk_size VALUES
# ─────────────────────────────────────────────────────────────────────────────
# Let's see how chunk_size affects the number of chunks.
# This helps build intuition for choosing the right value.

print("--- Experiment: Different chunk_size values ---")
print()

# We'll try 3 different chunk sizes on the same document
for size in [500, 1000, 2000]:
    test_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=int(size * 0.2),  # 20% overlap is a good rule of thumb
    )
    test_chunks = test_splitter.split_documents(docs)
    avg_len = sum(len(c.page_content) for c in test_chunks) / len(test_chunks)
    # sum() adds up all the chunk lengths
    # dividing by len(test_chunks) gives us the average

    print(f"  chunk_size={size:>5}, overlap={int(size*0.2):>4}"
          f"  →  {len(test_chunks):>3} chunks, avg {avg_len:.0f} chars each")

print()
print("Notice: smaller chunk_size = more chunks = more precise retrieval")
print("        larger chunk_size = fewer chunks = more context per chunk")
print()


# ─────────────────────────────────────────────────────────────────────────────
# KEY TAKEAWAYS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("KEY TAKEAWAYS — Section 2")
print("=" * 60)
print()
print("1. We chunk because: context window limits, LLM confusion, cost")
print("2. RecursiveCharacterTextSplitter splits smartly (paragraphs first)")
print("3. chunk_size=1000, overlap=200 is a solid starting point")
print("4. Overlap prevents sentences from being split in half")
print("5. Smaller chunks = more precise but less context")
print("   Bigger chunks = more context but more noise")
print()
print("Next: Section 3 — Embed chunks and store them in ChromaDB")
