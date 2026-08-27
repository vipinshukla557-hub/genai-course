# ─────────────────────────────────────────────────────────────────────────────
# WEEK 8 · SECTION 5: FULL RAG GENERATION — THE COMPLETE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
# Applied GenAI Engineering Program
#
# WHAT THIS FILE COVERS:
#   - Building the generation prompt (system + context + question)
#   - Passing retrieved chunks as context to the LLM
#   - Getting grounded answers with source attribution
#   - Testing edge case: asking something NOT in the document
#   - Putting it all together: the complete RAG pipeline in one flow
#
# PREREQUISITE: Run Sections 1–3 first (to create the ChromaDB database)
#
# ENVIRONMENT:
#   OpenAI gpt-4o-mini (primary)
#   Requires OPENAI_API_KEY in your .env file
#
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
# We need the OpenAI API key to call the LLM.
# Same pattern as Week 1 — load from .env file.

import os
from dotenv import load_dotenv

# load_dotenv() reads your .env file and puts the values into
# environment variables. This keeps your API key out of the code.
load_dotenv()

# Quick check that the key is set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY not found in .env file!")
    print("   Create a .env file with: OPENAI_API_KEY=your-key-here")
    exit()

print("=" * 60)
print("SECTION 5: Full RAG Generation — The Complete Pipeline")
print("=" * 60)
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: SET UP ALL THE PIECES
# ─────────────────────────────────────────────────────────────────────────────
# We need: embedding model, vector store, LLM, and a prompt template.
# Let's load everything.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─── FALLBACK: If sentence-transformers didn't install ───────────────
# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# ─────────────────────────────────────────────────────────────────────

# 1. Embedding model (same one we used to store chunks)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load the vector store from disk
vectorstore = Chroma(
    persist_directory="week8_chroma_db",
    embedding_function=embeddings
)

# 3. Create the LLM
# temperature=0 means we want consistent, factual answers (no creativity)
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Fast, cheap, good for RAG
    temperature=0          # 0 = deterministic (same input → same output)
)

# ─── SWAP: To use Groq (free) instead of OpenAI ─────────────────────
# from langchain_groq import ChatGroq
# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# # Requires GROQ_API_KEY in your .env file
# ─────────────────────────────────────────────────────────────────────

print("✅ All pieces loaded: embeddings, vector store, LLM")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: BUILD THE RAG PROMPT
# ─────────────────────────────────────────────────────────────────────────────
# This is the most important part of RAG. The prompt tells the LLM:
#   1. WHAT to do — answer questions based on provided context
#   2. WHAT NOT to do — don't make stuff up if the answer isn't there
#   3. HOW to behave — be helpful, cite sources, admit uncertainty
#
# The prompt has THREE parts:
#   SYSTEM: Instructions for the LLM (always follow these rules)
#   CONTEXT: The retrieved chunks (this changes with every question)
#   HUMAN: The user's actual question
#
# Think of it like giving a new employee a task:
#   "Here's our company policy (context).
#    A customer is asking about leave (question).
#    Answer ONLY from this policy. If it's not in here, say so."

rag_prompt = ChatPromptTemplate.from_messages([
    # SYSTEM message — sets the rules the LLM must follow
    ("system",
     """You are a helpful assistant that answers questions based ONLY on the
provided context. Follow these rules strictly:

1. Answer the question using ONLY the information in the context below.
2. If the context does not contain the answer, say: "I don't have enough
   information in the provided documents to answer this question."
3. Keep your answer clear and concise.
4. When possible, mention which section or part of the document your
   answer comes from.
5. Do NOT use any knowledge from your training data — only the context.

CONTEXT:
{context}"""),

    # HUMAN message — the user's question
    # {question} will be replaced with the actual question
    ("human", "{question}")
])

# {context} and {question} are PLACEHOLDERS. When we run the chain,
# LangChain replaces them with actual values. This is the same
# ChatPromptTemplate pattern from Week 5.

print("✅ RAG prompt template created")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: THE COMPLETE RAG FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
# Let's build a function that does the ENTIRE RAG pipeline:
#   1. Take a question
#   2. Retrieve relevant chunks
#   3. Format them into the prompt
#   4. Send to the LLM
#   5. Return the answer + sources

def ask_rag(question, k=4):
    """
    Ask a question and get an answer grounded in the document.

    Parameters:
        question (str): The user's question in plain English
        k (int): How many chunks to retrieve (default: 4)

    Returns:
        Nothing — prints the answer and sources directly
    """
    # ── Step 1: Retrieve relevant chunks ──
    # similarity_search finds the top-k chunks closest in meaning
    retrieved_chunks = vectorstore.similarity_search(query=question, k=k)

    # ── Step 2: Format the context ──
    # We need to combine all chunk texts into one big string
    # that goes into the {context} placeholder in our prompt.
    #
    # We add "---" between chunks so the LLM can see where one
    # chunk ends and another begins. We also add the source info.
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        # f-string builds a formatted string
        # chunk.metadata.get("source", "unknown") gets the source
        # from metadata, or "unknown" if it's not there.
        source = chunk.metadata.get("source", "unknown")
        context_parts.append(
            f"[Source: {source}]\n{chunk.page_content}"
        )
    # "\n\n---\n\n".join() puts "---" between each chunk
    context_text = "\n\n---\n\n".join(context_parts)

    # ── Step 3: Build the LCEL chain ──
    # Remember from Week 5: the pipe operator | connects components
    # prompt | llm | parser = take prompt → send to LLM → parse as string
    chain = rag_prompt | llm | StrOutputParser()

    # ── Step 4: Run the chain ──
    # .invoke() runs the chain with our context and question
    answer = chain.invoke({
        "context": context_text,  # Fills the {context} placeholder
        "question": question      # Fills the {question} placeholder
    })

    # ── Step 5: Display results ──
    print(f"🔍 Question: {question}")
    print()
    print(f"💬 Answer:")
    print(f"   {answer}")
    print()
    print(f"📚 Sources used ({len(retrieved_chunks)} chunks):")
    for i, chunk in enumerate(retrieved_chunks):
        preview = chunk.page_content[:80].strip().replace("\n", " ")
        print(f"   {i+1}. {preview}...")
    print()
    print("-" * 60)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: TEST WITH REAL QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
# Let's ask questions about our Namaste Foods handbook and see how
# the RAG pipeline answers them. The LLM should use ONLY the document
# content, not its own knowledge.

print("=" * 60)
print("TESTING THE RAG PIPELINE")
print("=" * 60)
print()

# Question 1: Should be answerable from the leave policy chapter
ask_rag("How many sick leaves do I get per year? Can I carry them forward?")

# Question 2: Should find the WFH policy section
ask_rag("How many days can the engineering team work from home?")

# Question 3: Should find the performance review section
ask_rag("What increment do I get if my rating is 4?")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: TEST THE EDGE CASE — QUESTION NOT IN THE DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────
# This is the critical test. What happens when someone asks about
# something that's NOT in the handbook?
#
# Without RAG, the LLM would happily make something up (hallucinate).
# With our RAG prompt, it should say "I don't have that information."
#
# This is one of the BIGGEST advantages of RAG over raw LLM usage.

print("=" * 60)
print("EDGE CASE: Question NOT in the document")
print("=" * 60)
print()

ask_rag("What is the company's stock option (ESOP) policy?")

# The handbook doesn't mention ESOPs at all.
# A good RAG system should say it doesn't have that information.
# If the LLM makes something up, that means our prompt needs to be stricter.


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: THE COMPLETE PICTURE
# ─────────────────────────────────────────────────────────────────────────────
# Let's zoom out and see what we built today:
#
#   1. LOAD      →  Read a document (PDF or text file)
#   2. SPLIT     →  Break it into ~1000-character chunks with overlap
#   3. EMBED     →  Convert each chunk into a 384-number vector
#   4. STORE     →  Save vectors + text + metadata in ChromaDB
#   5. RETRIEVE  →  Given a question, find the top-K most similar chunks
#   6. GENERATE  →  Pass chunks as context to the LLM, get a grounded answer
#
# This entire pipeline is what companies build for:
#   - Customer support bots that answer from product docs
#   - Internal tools that search company policies
#   - Legal assistants that find relevant clauses in contracts
#   - Medical tools that search research papers
#
# Next week (Week 9), we wrap this pipeline in a Streamlit web app
# where users can upload their own PDFs and chat with them.

print("=" * 60)
print("KEY TAKEAWAYS — Section 5 (and all of Week 8)")
print("=" * 60)
print()
print("1. The RAG prompt has 3 parts: system rules + context + question")
print("2. 'Answer ONLY from context' prevents hallucination")
print("3. Source attribution tells users WHERE the answer came from")
print("4. Edge case test: the system correctly says 'I don't know'")
print("5. The 6-step pipeline: Load → Split → Embed → Store → Retrieve → Generate")
print()
print("🎉 Congratulations! You've built a complete RAG pipeline from scratch.")
print("   Next week, we turn this into a web app anyone can use.")
