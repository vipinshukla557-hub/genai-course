# ─────────────────────────────────────────────────────────────────────────────
# WEEK 8 · LIVE DEMO — PART 1: LOAD AND SPLIT DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────
# WHAT WE BUILD:
#   1. Create a sample company document
#   2. Load it with LangChain
#   3. Split it into chunks
#   4. Inspect the chunks
#   5. Understand chunk overlap
# ─────────────────────────────────────────────────────────────────────────────


# ── STEP 1: Create a sample document ────────────────────────────────
# In real projects, companies usually already have PDFs or text files.
# For learning, we will create our own sample document.

sample_text = """
NAMASTE FOODS — EMPLOYEE HANDBOOK

=== LEAVE POLICY ===

All full-time employees get the following leaves per year:

Casual Leave: 12 days per year. Cannot carry forward to next year.
Must apply at least 1 day in advance.

Sick Leave: 10 days per year. No prior notice needed but medical
certificate required if taking more than 2 consecutive days.
Unused sick leave carries forward up to 30 days maximum.

Earned Leave: 15 days per year. Can carry forward up to 45 days.
Can be encashed when leaving the company. Apply 7 days in advance
if taking 3 or more consecutive days.

Maternity Leave: 26 weeks as per the Maternity Benefit Act, 2017.

=== WORK FROM HOME POLICY ===

Effective April 2024, Namaste Foods follows a hybrid model:

Engineering Team: 3 days in office (Tue, Wed, Thu), 2 days remote.
Operations Team: 5 days in office. No remote work.
Marketing Team: 2 days in office (Mon, Wed), 3 days remote.

Core hours: 10 AM to 6 PM IST, regardless of location.

Home office allowance: One-time Rs 15,000 for monitor, keyboard, desk.
Internet reimbursement: Rs 1,000 per month for remote days.

=== PERFORMANCE REVIEWS ===

Reviews happen twice a year: April (mid-year) and October (annual).

Rating 5 — Exceptional: 15-20% increment. Top 5% performers.
Rating 4 — Exceeds Expectations: 10-15% increment.
Rating 3 — Meets Expectations: 5-8% increment.
Rating 2 — Needs Improvement: 0% increment, improvement plan.
Rating 1 — Unsatisfactory: 0% increment, formal PIP.

Promotion requires rating 4+ and at least 18 months in current role.

=== EXPENSE POLICY ===

Business Travel: Economy class flights only. Daily meal allowance
is Rs 1,500 in metro cities and Rs 1,000 in non-metro cities.

Client Entertainment: Manager approval needed above Rs 5,000.
Receipts required within 7 days. Alcohol is never reimbursable.

Software: Any subscription above Rs 2,000/month needs IT approval.
"""


# Create a text file and save the sample text into it

file = open("company_handbook.txt", "w", encoding="utf-8")
file.write(sample_text)
file.close()

print("Document saved successfully")
print()


# ── STEP 2: Load the document with LangChain ────────────────────────

from langchain_community.document_loaders import TextLoader

loader = TextLoader("company_handbook.txt", encoding="utf-8")
docs = loader.load()

print("Number of documents loaded:", len(docs))

print("Document size:", len(docs[0].page_content), "characters")

print("Metadata:", docs[0].metadata)

# ── STEP 3: Split document into chunks ──────────────────────────────

#from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)

chunks = splitter.split_documents(docs)

print("Chunks ready:", len(chunks))

print()

# ── STEP 4: Inspect the chunks ──────────────────────────────────────
# Loop through all chunks one by one

for i in range(len(chunks)):
    chunk = chunks[i]
    preview = chunk.page_content[:80]
    # Remove extra spaces
    preview = preview.strip()
    # Replace new lines with spaces
    preview = preview.replace("\n", " ")

    print("Chunk:", i + 1)

    print("Characters:", len(chunk.page_content))

    print("Preview:", preview)

    print()


# ── STEP 5: Understand overlap ──────────────────────────────────────
# Some text from previous chunk is repeated
# in the next chunk to preserve meaning/context.
#
# Let's compare Chunk 2 and Chunk 3 to see this.

print("----- OVERLAP CHECK -----")
print()

print("End of Chunk 2:")
print(chunks[1].page_content[-150:])
print()

print("Start of Chunk 3:")
print(chunks[2].page_content[:150])
print()

print("This repeated part is called OVERLAP.")

print()

# ─────────────────────────────────────────────────────────────────────
# DONE
# We:
# 1. Created a document
# 2. Loaded it with LangChain
# 3. Split it into chunks
# 4. Inspected chunk overlap
# ─────────────────────────────────────────────────────────────────────