"""
config.py
---------
Centralised configuration for the Document Q&A app.

WHY a config file?
    In real software projects, scattered "magic numbers" (chunk_size=1000 in one file,
    chunk_size=800 in another) cause bugs that are hard to track down. We put every
    tunable value in ONE place. To experiment, you change ONE number here and the
    whole pipeline picks it up.

WHAT lives here:
    - API keys (loaded from .env, never hardcoded)
    - Model names (so swapping models is a one-line change)
    - Chunking parameters (chunk_size, overlap)
    - Retrieval parameters (top_k)
    - Paths (where ChromaDB stores its files)
"""

"""
# ─────────────────────────────────────────────
# WHAT'S INSIDE config.py:
# ─────────────────────────────────────────────
# 1. IMPORTS + LOAD ENV        → Read our secret API keys
# 2. PROVIDER SELECTION        → OpenAI or Groq?
# 3. MODEL SETTINGS            → Temperature, max tokens
# 4. EMBEDDING SETTINGS        → Which embedding model to use
# 5. CHUNKING SETTINGS         → How to split documents
# 6. RETRIEVAL SETTINGS        → How many chunks to fetch
# 7. CONVERSATION MEMORY       → How much chat history to keep
# 8. PATHS                     → Where files and databases live
# 9. SAFETY SETTINGS           → Input limits
# 10. VALIDATE CONFIG          → Fail-fast startup check
# ─────────────────────────────────────────────
"""


import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file into os.environ.
# This must run before we read any os.getenv() calls below.
load_dotenv()


# =============================================================================
# PROVIDER SELECTION
# =============================================================================

# Which LLM provider to use. Read from .env so students can switch without
# touching code. Defaults to "openai" if not set.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()

# API keys — never put these in code, always read from environment.
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")


# =============================================================================
# MODEL SETTINGS
# =============================================================================

# Chat model names. We use small, cheap models — they're plenty good for Q&A.
OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
GROQ_CHAT_MODEL: str = "openai/gpt-oss-120b"

# Temperature controls creativity. For grounded Q&A on documents we want
# the model to stick closely to the source — so we set it LOW (0.0–0.2).
# Higher temperature (0.7+) is for creative writing, not factual answering.
LLM_TEMPERATURE: float = 0.1

# Max tokens the model can generate in one answer. Caps cost and prevents
# runaway responses. 800 tokens ≈ 600 words — enough for a thorough answer.
LLM_MAX_TOKENS: int = 800


# =============================================================================
# EMBEDDING SETTINGS
# =============================================================================

# We use sentence-transformers — runs LOCALLY on your machine, no API call,
# no cost. "all-MiniLM-L6-v2" is small (~80MB) and fast. It produces 384-dim
# vectors that work great for semantic search on English documents.
#
# To swap to OpenAI embeddings later, change EMBEDDING_PROVIDER to "openai"
# in your .env and the vector_store module will use text-embedding-3-small.
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"


# =============================================================================
# CHUNKING SETTINGS  (taught in Week 8)
# =============================================================================

# When we split a long document, chunks of ~1000 characters work well
# because they fit comfortably in the LLM's context AND are large enough
# to contain a complete thought (a paragraph, a definition, an example).
CHUNK_SIZE: int = 1000

# Overlap means consecutive chunks share some text. Why? Because the answer
# to a question might sit RIGHT on the boundary between two chunks. Overlap
# ensures we never lose information at the edges.
# Rule of thumb: overlap = 10–20% of chunk_size.
CHUNK_OVERLAP: int = 200


# =============================================================================
# RETRIEVAL SETTINGS
# =============================================================================

# How many chunks to fetch from the vector store for each question.
# Too few (k=1) → may miss the answer. Too many (k=10) → noisy + expensive.
# k=4 is a solid default for most documents.
TOP_K: int = 4


# =============================================================================
# CONVERSATION MEMORY  (Week 6 — user-managed, modern LangChain 1.x pattern)
# =============================================================================

# Maximum tokens of chat history to keep on each turn. When the running
# conversation exceeds this, older turns get dropped (oldest first).
#
# Why 2000? gpt-4o-mini has a 128k context, so 2000 is tiny. But we WANT
# tight history because:
#   - Retrieved chunks already eat ~1500 tokens (4 chunks × ~1000 chars)
#   - We want the LLM focused on the document, not on chitchat from 20 turns ago
#   - Smaller history = faster + cheaper API calls
# Tune this up if your students need longer conversational memory.
MAX_HISTORY_TOKENS: int = 2000


# =============================================================================
# PATHS
# =============================================================================

# Project root — works no matter where you run the script from.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Where ChromaDB persists its files. Persistence means: index once, reuse
# many times. Restart the app, the index is still there.
CHROMA_DB_DIR: Path = PROJECT_ROOT / "chroma_db"

# Where uploaded PDFs are saved temporarily.
UPLOADS_DIR: Path = PROJECT_ROOT / "uploads"

# Make sure the directories exist (creates them if missing, no error if they do).
CHROMA_DB_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)


# =============================================================================
# SAFETY SETTINGS  (basic guardrails, expanded in src/safety.py)
# =============================================================================

# Reject questions longer than this. Stops people from pasting an entire
# book as a "question" and burning your API budget.
MAX_QUESTION_LENGTH: int = 1000

# Reject PDFs larger than this. Free tiers can't handle huge files.
MAX_PDF_SIZE_MB: int = 25


def validate_config() -> None:
    """
    Sanity check: make sure the settings actually work together.
    Called at app startup — fail fast with a clear message if something
    is wrong, instead of crashing later with a confusing error.
    """
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise ValueError(
            "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is missing. "
            "Did you copy .env.example to .env and fill in your key?"
        )
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        raise ValueError(
            "LLM_PROVIDER is 'groq' but GROQ_API_KEY is missing. "
            "Get a free one at https://console.groq.com/keys"
        )
    if LLM_PROVIDER not in ("openai", "groq"):
        raise ValueError(
            f"LLM_PROVIDER must be 'openai' or 'groq', got: {LLM_PROVIDER!r}"
        )
