# ─────────────────────────────────────────────────────────────────────────────
# WEEK 8 · LIVE DEMO — PART 2: RETRIEVE AND GENERATE ANSWERS
# ─────────────────────────────────────────────────────────────────────────────
# WHAT WE BUILD:
#   1. Embed our chunks and store in ChromaDB
#   2. Search for relevant chunks given a question
#   3. Pass chunks as context to the LLM
#   4. Get an answer grounded in the document
#   5. Test: what happens when the answer ISN'T in the document?
#
# PREREQUISITE: Run Part 1 first (creates company_handbook.txt)
# ─────────────────────────────────────────────────────────────────────────────


# ── STEP 1: Load and chunk the document ─────────────────────────────

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

docs = TextLoader("company_handbook.txt", encoding="utf-8").load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

chunks = splitter.split_documents(docs)

print("Chunks ready:", len(chunks))
print()


# ── STEP 2: Embed chunks and store in ChromaDB ─────────────────────

import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db_path = "demo_chroma_db"

if os.path.exists(db_path):
    shutil.rmtree(db_path)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=db_path
)

print("Chunks embedded and stored in ChromaDB:", vectorstore._collection.count())
print()


# ── STEP 3: Search for relevant chunks ──────────────────────────────

question = "How many sick leaves do I get"

results = vectorstore.similarity_search(query=question, k=3)

print("Question:", question)
print("Relevant chunks retrieved:", len(results))
print()


# Show a preview of each retrieved chunk

for i in range(len(results)):
    preview = results[i].page_content[:100]
    preview = preview.strip().replace("\n", " ")
    print(" ", i + 1, ".", preview, "...")

print()


# ── STEP 4: Pass chunks to the LLM ─────────────────────────────────

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information
in the provided documents."

# CONTEXT:
# {context}"""),
    ("human", "{question}")
])

chain = rag_prompt | llm | StrOutputParser()

chunk_texts = []

for i in range(len(results)):
    chunk_texts.append(results[i].page_content)

context = "\n\n---\n\n".join(chunk_texts)

answer = chain.invoke({
    "context": context,
    "question": question
})

print("Question:", question)
print("Answer:", answer)
print()


# ── STEP 5: Test the edge case ──────────────────────────────────────

print("=" * 50)
print("EDGE CASE — question NOT in the document:")
print()

edge_question = "What is the company's ESOP or stock option policy?"

edge_results = vectorstore.similarity_search(query=edge_question, k=3)

edge_texts = []

for i in range(len(edge_results)):
    edge_texts.append(edge_results[i].page_content)

edge_context = "\n\n---\n\n".join(edge_texts)

edge_answer = chain.invoke({
    "context": edge_context,
    "question": edge_question
})

print("Question:", edge_question)
print("Answer:", edge_answer)
print()


# ─────────────────────────────────────────────────────────────────────
# DONE! Full RAG pipeline:
#   Load → Split → Embed → Store → Retrieve → Generate
# ─────────────────────────────────────────────────────────────────────
