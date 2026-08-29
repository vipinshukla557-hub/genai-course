"""
test_engine.py
--------------
A command-line smoke test for the RAG engine.

WHY this exists:
    Before we build the Streamlit UI, we want to PROVE the engine works.
    Streamlit adds a lot of complexity (sessions, reruns, streaming to a
    web socket). If something is broken, you don't know whether it is your
    engine or your UI. So we test the engine in isolation here, on the
    command line, where errors are loud and obvious.

    This is not a deliverable. It is a debugging tool. In real software
    teams, every backend service has a small CLI like this so you can poke
    at it without spinning up the whole frontend.

USAGE:
    # One-shot mode — index a PDF and ask one question:
    python test_engine.py path/to/sample.pdf "What is this document about?"

    # Interactive chat mode — index then have a multi-turn conversation
    # (demonstrates the Week 6 user-managed memory):
    python test_engine.py path/to/sample.pdf
"""

import sys

from langchain_core.messages import HumanMessage, AIMessage

from src.config import validate_config
from src.document_loader import load_and_split
from src.vector_store import build_vector_store, reset_vector_store
from src.rag_chain import answer_question


def index_pdf(pdf_path: str):
    """Load + split + index. Returns the vector store."""
    print("[1/3] Validating config...")
    validate_config()

    print(f"[2/3] Loading and splitting {pdf_path}...")
    chunks = load_and_split(pdf_path)
    print(f"      Got {len(chunks)} chunks.")

    print("[3/3] Building vector store (first run downloads ~80MB embedding model)...")
    try:
        reset_vector_store()
    except Exception:
        pass
    return build_vector_store(chunks)


def print_result(result: dict) -> None:
    """Pretty-print an answer + sources."""
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result["answer"])
    print()

    if result["sources"]:
        print("=" * 60)
        print(f"SOURCES ({len(result['sources'])} chunks):")
        print("=" * 60)
        for i, chunk in enumerate(result["sources"], start=1):
            page = chunk.metadata.get("page", 0) + 1
            source_file = chunk.metadata.get("filename", "unknown")
            preview = chunk.page_content[:200].replace("\n", " ")
            print(f"\n[{i}] {source_file} — page {page}")
            print(f"    {preview}...")
    print()


def one_shot_mode(pdf_path: str, question: str) -> None:
    """Run a single question and exit."""
    vector_store = index_pdf(pdf_path)
    print(f"\n[Q] {question}\n")
    result = answer_question(vector_store, question)
    print_result(result)


def interactive_mode(pdf_path: str) -> None:
    """
    Multi-turn chat — keeps user-managed history across turns.

    This demonstrates the Week 6 modern-LangChain memory pattern:
    we OWN the message list. Each user input becomes a HumanMessage,
    each model reply becomes an AIMessage, both get appended. The
    engine trims when the list grows too long.
    """
    vector_store = index_pdf(pdf_path)
    print("\n" + "=" * 60)
    print("Interactive chat mode. Type 'quit' to exit.")
    print("Try a follow-up like 'tell me more' or 'what about page 5?'")
    print("=" * 60)

    chat_history: list = []

    while True:
        try:
            question = input("\n[You] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        result = answer_question(
            vector_store,
            question,
            chat_history=chat_history,
        )

        print_result(result)

        # ---- Update the chat history ourselves (Week 6 pattern) ----
        # No framework, no hidden state. We append, the engine trims.
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=result["answer"]))


def main() -> None:
    if len(sys.argv) == 2:
        # Interactive mode
        interactive_mode(sys.argv[1])
    elif len(sys.argv) == 3:
        # One-shot mode
        one_shot_mode(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print('  python test_engine.py <pdf_path> "<question>"   # one-shot')
        print("  python test_engine.py <pdf_path>                # interactive chat")
        sys.exit(1)


if __name__ == "__main__":
    main()
