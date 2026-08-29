"""
verify_environment.py
---------------------
Run this AFTER installing requirements.txt to confirm everything imports
correctly on YOUR machine. This catches version mismatches before you
hit them mid-recording.

USAGE:
    python verify_environment.py

WHAT IT CHECKS:
    1. Python version
    2. All third-party imports the project uses (LangChain 1.x, ChromaDB,
       sentence-transformers, Streamlit, etc.)
    3. The actual classes/functions we import are reachable
    4. Project's own modules import cleanly (no circular imports, no typos)

If everything passes, you're safe to record.
If anything fails, you'll see exactly which package/version is wrong.
"""

import sys
import importlib

# ANSI colour codes for terminal output (work in modern PowerShell too)
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def header(text: str) -> None:
    print(f"\n{BOLD}{text}{RESET}")
    print("─" * len(text))


def ok(text: str) -> None:
    print(f"  {GREEN}✓{RESET} {text}")


def fail(text: str) -> None:
    print(f"  {RED}✗{RESET} {text}")


def warn(text: str) -> None:
    print(f"  {YELLOW}!{RESET} {text}")


# Track failures so we can give a clean exit code at the end.
failures: list[str] = []


def try_import(module_path: str, attrs: list[str]) -> None:
    """Try to import a module and access named attributes from it."""
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        fail(f"{module_path}  ({e})")
        failures.append(module_path)
        return

    missing = [a for a in attrs if not hasattr(mod, a)]
    if missing:
        fail(f"{module_path}  (missing: {', '.join(missing)})")
        failures.append(f"{module_path}.{missing[0]}")
        return

    # Try to get the version
    version = getattr(mod, "__version__", "?")
    ok(f"{module_path}  v{version}  ({', '.join(attrs)})")


# =============================================================================
# 1. Python version
# =============================================================================

header("1. Python version")
print(f"  Python {sys.version.split()[0]}  ({sys.platform})")
if sys.version_info < (3, 10):
    fail("Need Python 3.10 or newer")
    failures.append("python-version")
else:
    ok("Python version OK")


# =============================================================================
# 2. Third-party imports
# =============================================================================

header("2. Third-party packages")

try_import("openai", ["OpenAI"])
try_import("dotenv", ["load_dotenv"])

try_import("langchain_core.prompts", ["ChatPromptTemplate", "MessagesPlaceholder"])
try_import("langchain_core.output_parsers", ["StrOutputParser"])
try_import("langchain_core.documents", ["Document"])
try_import("langchain_core.messages", [
    "HumanMessage", "AIMessage", "SystemMessage", "BaseMessage", "trim_messages",
])

try_import("langchain_community.document_loaders", ["PyPDFLoader"])
try_import("langchain_text_splitters", ["RecursiveCharacterTextSplitter"])
try_import("langchain_openai", ["ChatOpenAI", "OpenAIEmbeddings"])
try_import("langchain_chroma", ["Chroma"])
try_import("langchain_huggingface", ["HuggingFaceEmbeddings"])

try_import("chromadb", [])  # any version OK
try_import("pypdf", [])
try_import("streamlit", [])
try_import("tiktoken", [])
try_import("sentence_transformers", ["SentenceTransformer"])


# =============================================================================
# 3. Project's own modules
# =============================================================================

header("3. Project modules")

# Add the project root to sys.path so we can import `src.*` even when running
# this file from anywhere.
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try_import("src.config", [
    "validate_config", "OPENAI_API_KEY", "TOP_K", "MAX_HISTORY_TOKENS",
])
try_import("src.safety", ["validate_question", "validate_pdf_file", "UnsafeInputError"])
try_import("src.document_loader", ["load_pdf", "split_documents", "load_and_split"])
try_import("src.vector_store", [
    "build_vector_store", "load_vector_store", "reset_vector_store",
    "retrieve_relevant_chunks",
])
try_import("src.rag_chain", [
    "answer_question", "stream_answer", "trim_history",
    "SYSTEM_PROMPT", "RagResult",
])


# =============================================================================
# 4. Quick smoke test of the LangChain prompt builder
# =============================================================================

header("4. Prompt template smoke test")
try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage

    tmpl = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Context: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    msgs = tmpl.format_messages(
        context="some context",
        chat_history=[HumanMessage("Hi"), AIMessage("Hello!")],
        question="What's up?",
    )
    if len(msgs) == 4:  # system + 2 history + 1 human
        ok(f"Prompt builds correctly (yielded {len(msgs)} messages)")
    else:
        fail(f"Expected 4 messages, got {len(msgs)}")
        failures.append("prompt-template")
except Exception as e:
    fail(f"Prompt build failed: {e}")
    failures.append("prompt-template")


# =============================================================================
# 5. Summary
# =============================================================================

header("Summary")
if failures:
    print(f"\n{RED}{BOLD}FAILED{RESET} — {len(failures)} issue(s):")
    for f in failures:
        print(f"  • {f}")
    print()
    print("Fix these, re-run this script, and you're good to go.")
    sys.exit(1)
else:
    print(f"\n{GREEN}{BOLD}ALL CHECKS PASSED.{RESET}")
    print("\nYou can now run:")
    print("  python test_engine.py path/to/your.pdf \"Your question?\"")
    print("  streamlit run streamlit_app.py")
    sys.exit(0)
