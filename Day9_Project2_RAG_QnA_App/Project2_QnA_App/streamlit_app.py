"""
streamlit_app.py
----------------
The Week 9 final deliverable: a web UI for our Document Q&A app.

This file does ZERO new RAG work. All the heavy lifting lives in src/.
Streamlit is just the "skin" wrapped around the engine — that's the
same separation real software teams use.

WHAT THIS APP DOES:
    1. User uploads a PDF.
    2. App builds the index (one-time work for that document).
    3. User asks questions in a chat interface.
    4. Answers stream in token-by-token.
    5. Each answer shows the source pages it came from.

RUN IT:
    streamlit run streamlit_app.py
"""

# ─────────────────────────────────────────────
# WHAT'S INSIDE streamlit_app.py:
# ─────────────────────────────────────────────
# 1. DOCSTRING               → What this file is and how to run it
# 2. IMPORTS                  → Streamlit, LangChain messages, our engine
# 3. PAGE CONFIG              → Browser tab title, layout
# 4. CONFIG VALIDATION        → Fail-fast if API key is missing
# 5. SESSION STATE            → How Streamlit remembers things across reruns
# 6. SIDEBAR                  → File upload + indexing + clear button
# 7. MAIN AREA HEADER         → Title + "upload first" guard
# 8. CHAT HISTORY REPLAY      → Re-draw all past messages on every rerun
# 9. CHAT INPUT               → The text box where users type questions
# 10. MEMORY WIRING           → Convert session state to LangChain messages
# 11. STREAMING + SOURCES     → Show answer word-by-word + page citations
# 12. ERROR HANDLING           → Catch safety rejections and crashes
# ─────────────────────────────────────────────

import os
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.config import validate_config, UPLOADS_DIR
from src.document_loader import load_and_split
from src.vector_store import (
    build_vector_store,
    load_vector_store,
    reset_vector_store,
)
from src.rag_chain import stream_answer
from src.safety import UnsafeInputError


# =============================================================================
# PAGE CONFIG  (must be the first Streamlit call)
# =============================================================================

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide",
)


# =============================================================================
# CONFIG VALIDATION  (run at app startup)
# =============================================================================

try:
    validate_config()
except ValueError as e:
    st.error(f"Configuration error: {e}")
    st.stop()


# =============================================================================
# SESSION STATE  (the app's memory)
# =============================================================================

# Streamlit re-runs the entire script on every interaction. To keep things
# alive across re-runs, we store them in `st.session_state` — a dict-like
# object scoped to one user's browser session.
if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None  # name of the currently indexed PDF
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"}


# =============================================================================
# SIDEBAR — file upload and controls
# =============================================================================

with st.sidebar:
    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Maximum 25 MB. The file stays on the server only while you use the app.",
    )

    if uploaded_file is not None:
        # Has the user uploaded a different file than the one we already indexed?
        is_new_file = st.session_state.indexed_filename != uploaded_file.name

        if is_new_file:
            # Save the uploaded bytes to disk so PyPDFLoader can read it.
            saved_path = UPLOADS_DIR / uploaded_file.name
            saved_path.write_bytes(uploaded_file.getbuffer())

            with st.status("Indexing document...", expanded=True) as status:
                st.write("📄 Loading and splitting...")
                try:
                    chunks = load_and_split(saved_path)
                except UnsafeInputError as e:
                    status.update(label="Failed", state="error")
                    st.error(str(e))
                    st.stop()

                st.write(f"   → {len(chunks)} chunks created")

                st.write("🧹 Clearing old index...")
                try:
                    reset_vector_store()
                except Exception:
                    pass

                st.write("🔢 Embedding chunks (this may take a minute on first run)...")
                build_vector_store(chunks)

                st.write("✅ Done!")
                status.update(label="Ready to chat", state="complete")

            st.session_state.indexed_filename = uploaded_file.name
            st.session_state.messages = []  # fresh chat for fresh document
            st.rerun()

        # Show what's currently indexed
        st.success(f"Indexed: **{st.session_state.indexed_filename}**")

    st.divider()

    # Reset button — clears chat and index
    if st.button("🗑️ Clear everything"):
        st.session_state.messages = []
        st.session_state.indexed_filename = None
        try:
            reset_vector_store()
        except Exception:
            pass
        # Wipe uploaded files too
        for f in UPLOADS_DIR.glob("*.pdf"):
            try:
                f.unlink()
            except Exception:
                pass
        st.rerun()

    st.divider()
    st.caption(
        "Built with LangChain + ChromaDB + sentence-transformers. "
        "Project 2 of the Applied GenAI Engineering Program."
    )


# =============================================================================
# MAIN AREA — chat interface
# =============================================================================

st.title("📄 Document Q&A")
st.caption("Upload a PDF on the left, then ask questions about it.")

# If nothing is indexed yet, show a friendly welcome and stop.
if st.session_state.indexed_filename is None:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()


# --- Render chat history ---
# Every time the script re-runs, we re-draw all past messages from session_state.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If this is an assistant message with sources, show them in a collapsible.
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    page = src.metadata.get("page", "?")
                    if isinstance(page, int):
                        page = page + 1
                    filename = src.metadata.get("filename", "?")
                    st.markdown(f"**{i}. {filename} — page {page}**")
                    st.markdown(f"> {src.page_content[:400]}...")


# --- Chat input ---
# st.chat_input renders a sticky text box at the bottom of the page.
prompt = st.chat_input("Ask a question about the document...")

if prompt:
    # 1) Show the user's message and store it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Generate the assistant response
    with st.chat_message("assistant"):
        try:
            store = load_vector_store()

            # ---- Build chat history for the LLM (Week 6 pattern) ----
            # We translate our display-friendly dicts into LangChain Message
            # objects. The CURRENT user prompt is excluded — the engine
            # receives it as the `question` argument, not as history.
            # This is user-managed memory: WE decide what to send. No
            # ConversationBufferMemory hiding things behind the scenes.
            chat_history = []
            for msg in st.session_state.messages[:-1]:  # skip the just-appended user msg
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

            # The engine will trim this further if it's too long — that's
            # also user-managed memory: trim_messages is a stateless utility,
            # not a hidden framework feature.
            token_stream, sources = stream_answer(
                store, prompt, chat_history=chat_history
            )

            # st.write_stream renders each yielded chunk live and returns
            # the full concatenated text once streaming finishes.
            full_answer = st.write_stream(token_stream)

            # Show sources
            if sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for i, src in enumerate(sources, 1):
                        page = src.metadata.get("page", "?")
                        if isinstance(page, int):
                            page = page + 1
                        filename = src.metadata.get("filename", "?")
                        st.markdown(f"**{i}. {filename} — page {page}**")
                        st.markdown(f"> {src.page_content[:400]}...")

            # 3) Persist the assistant message so it survives the next rerun
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources,
            })

        except UnsafeInputError as e:
            st.error(f"Rejected: {e}")
            # Remove the user's message we appended, so it doesn't sit
            # in history without a reply.
            st.session_state.messages.pop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.session_state.messages.pop()
