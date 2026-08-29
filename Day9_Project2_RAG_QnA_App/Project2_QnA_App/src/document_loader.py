"""
document_loader.py
------------------
Step 1 of the RAG pipeline: LOAD and SPLIT.

The full RAG pipeline:
    Load → Split → Embed → Store → Retrieve → Generate
    ^^^^^^^^^^^^
    THIS FILE

WHY split documents?
    LLMs have a context window — a maximum number of tokens they can read
    at once. A 100-page PDF would never fit. So we chop it into "chunks"
    of ~1000 characters each. Each chunk becomes one searchable unit.

WHY overlap?
    Imagine the answer to "What is the company's refund policy?" sits
    exactly across the boundary between chunk 5 and chunk 6. Without
    overlap, BOTH chunks would be incomplete. Overlap (200 chars) means
    the last 200 chars of chunk 5 are repeated at the start of chunk 6,
    so the answer is preserved intact in at least one chunk.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.safety import validate_pdf_file


def load_pdf(file_path: str | Path) -> List[Document]:
    """
    Load a PDF file and return a list of Document objects, one per page.

    Each Document has:
        - page_content: the extracted text
        - metadata: a dict with source filename, page number, etc.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of Document objects, one per page.

    Example:
        >>> docs = load_pdf("annual_report.pdf")
        >>> print(f"Loaded {len(docs)} pages")
        >>> print(docs[0].metadata)
        {'source': 'annual_report.pdf', 'page': 0}
    """
    # Run safety check first — fail fast on bad input.
    safe_path = validate_pdf_file(file_path)

    # PyPDFLoader is LangChain's wrapper around the pypdf library.
    # It returns ONE Document per page, with page numbers in metadata.
    loader = PyPDFLoader(str(safe_path))
    pages = loader.load()

    # Add the original filename to every page's metadata.
    # We need this later to tell users WHICH document an answer came from.
    filename = safe_path.name
    for page in pages:
        page.metadata["filename"] = filename

    return pages


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of Documents into smaller, overlapping chunks.

    Uses RecursiveCharacterTextSplitter which is smarter than naive splitting:
    it tries to break on paragraph boundaries first, then sentences, then
    words — only falling back to mid-word splits as a last resort. This
    keeps chunks readable and meaningful.

    Args:
        documents: List of Document objects (e.g., from load_pdf).

    Returns:
        A new list of smaller Document chunks. Each chunk inherits the
        metadata of the page it came from (filename, page number).

    Example:
        >>> pages = load_pdf("report.pdf")        # 50 pages
        >>> chunks = split_documents(pages)        # ~200 chunks
        >>> print(f"Got {len(chunks)} chunks")
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # The order of separators matters: it tries the FIRST one first.
        # Paragraph break > line break > sentence > word > character.
        separators=["\n\n", "\n", ". ", " ", ""],
        # length_function tells the splitter how to measure chunk size.
        # We use len() (character count). For token-accurate splitting,
        # you could pass a tokenizer-based function — overkill for now.
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    return chunks


def load_and_split(file_path: str | Path) -> List[Document]:
    """
    Convenience function: do both steps in one call.

    Most callers want "give me the chunks" — they don't care about the
    intermediate page list. This wrapper hides the two-step process.

    Args:
        file_path: Path to the PDF.

    Returns:
        List of Document chunks ready to be embedded.
    """
    pages = load_pdf(file_path)
    chunks = split_documents(pages)
    return chunks


# -----------------------------------------------------------------------------
# Quick self-test you can run from the command line:
#     python -m src.document_loader path/to/file.pdf
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.document_loader <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"Loading {pdf_path} ...")
    pages = load_pdf(pdf_path)
    print(f"  → loaded {len(pages)} pages")

    print("Splitting into chunks ...")
    chunks = split_documents(pages)
    print(f"  → got {len(chunks)} chunks")

    print("\nFirst chunk preview:")
    print("-" * 60)
    print(chunks[0].page_content[:300])
    print("-" * 60)
    print(f"Metadata: {chunks[0].metadata}")
