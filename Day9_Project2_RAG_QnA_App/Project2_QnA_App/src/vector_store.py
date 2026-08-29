"""
vector_store.py
---------------
Steps 2 + 3 of the RAG pipeline: EMBED and STORE (and RETRIEVE).

The full RAG pipeline:
    Load → Split → Embed → Store → Retrieve → Generate
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
                   THIS FILE

WHAT this module does:
    1. Convert text chunks into vectors (embeddings) — text becomes numbers
       that capture meaning. "dog" and "puppy" end up nearby in vector space.
    2. Store those vectors in ChromaDB so we can search them fast.
    3. Given a question, find the top-K most similar chunks.

WHY two embedding options?
    - HuggingFace (sentence-transformers): runs on YOUR machine, free, no
      API key. Slower the first time (downloads the model). Default choice.
    - OpenAI: faster, slightly better quality, but costs money per call.
      Available as a one-line swap if you want to try it.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.config import (
    EMBEDDING_PROVIDER,
    HF_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    CHROMA_DB_DIR,
    TOP_K,
)


# =============================================================================
# EMBEDDING MODEL FACTORY
# =============================================================================

def get_embedding_model() -> Embeddings:
    """
    Return the embedding model based on what's configured in .env.

    The function returns a LangChain `Embeddings` object — both HuggingFace
    and OpenAI providers implement the same interface, so the rest of the
    code doesn't care which one is in use.

    Returns:
        A LangChain Embeddings instance.

    Raises:
        ValueError: If EMBEDDING_PROVIDER is set to an unknown value.
    """
    if EMBEDDING_PROVIDER == "huggingface":
        # model_kwargs={'device': 'cpu'} forces CPU mode — works on any laptop.
        # encode_kwargs={'normalize_embeddings': True} makes cosine similarity
        # work correctly (vectors get unit length).
        return HuggingFaceEmbeddings(
            model_name=HF_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER!r}. "
        f"Use 'huggingface' or 'openai'."
    )


# =============================================================================
# VECTOR STORE: BUILD AND LOAD
# =============================================================================

# Collection name — like a "table name" inside ChromaDB. We use one
# collection per app. If you wanted multiple separate doc-libraries
# (e.g., legal vs HR) you'd give each a different collection name.
COLLECTION_NAME = "documents"


def build_vector_store(chunks: List[Document]) -> Chroma:
    """
    Take chunks, embed them, and store them in ChromaDB.

    This is the "indexing" step. Run it ONCE per document — after that,
    the index is persisted on disk and you load it instead of rebuilding.

    Args:
        chunks: List of Document chunks (from document_loader.split_documents).

    Returns:
        A Chroma vector store object you can query.

    Example:
        >>> chunks = load_and_split("report.pdf")
        >>> store = build_vector_store(chunks)   # embeds + saves
        >>> results = store.similarity_search("revenue", k=3)
    """
    embeddings = get_embedding_model()

    # Chroma.from_documents() does three things in one call:
    #   1. Calls the embedding model on each chunk's text
    #   2. Stores the (vector, text, metadata) triplet in ChromaDB
    #   3. Persists everything to disk at persist_directory
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_DIR),
    )
    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an existing ChromaDB index from disk.

    Use this when the index has already been built (i.e., on subsequent
    app starts). It's much faster than re-embedding everything.

    Returns:
        A Chroma vector store object.
    """
    embeddings = get_embedding_model()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )


def reset_vector_store() -> None:
    """
    Wipe the entire vector store. Use this when the user wants to start
    fresh with a new document, or in tests.
    """
    store = load_vector_store()
    # delete_collection removes everything in the collection.
    store.delete_collection()


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve_relevant_chunks(
    vector_store: Chroma,
    query: str,
    k: int = TOP_K,
) -> List[Document]:
    """
    Given a question, return the top-K most relevant chunks from the store.

    How it works under the hood:
        1. Embed the query (turn it into a vector)
        2. Compare that vector to every stored chunk vector (cosine similarity)
        3. Return the K closest chunks

    Args:
        vector_store: A loaded Chroma store.
        query: The user's question.
        k: How many chunks to retrieve. Defaults to config.TOP_K.

    Returns:
        List of the K most relevant Document chunks, most relevant first.

    Example:
        >>> store = load_vector_store()
        >>> chunks = retrieve_relevant_chunks(store, "What is RAG?", k=3)
        >>> for chunk in chunks:
        ...     print(chunk.metadata["page"], chunk.page_content[:100])
    """
    # similarity_search returns Documents in descending order of relevance.
    return vector_store.similarity_search(query, k=k)


# -----------------------------------------------------------------------------
# Quick self-test:
#     python -m src.vector_store
# (Assumes you've already indexed a PDF via document_loader)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading existing vector store...")
    store = load_vector_store()

    query = "What is this document about?"
    print(f"\nSearching for: {query!r}")
    results = retrieve_relevant_chunks(store, query, k=3)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {doc.metadata.get('filename', '?')}, "
              f"Page: {doc.metadata.get('page', '?')}")
        print(doc.page_content[:200])
