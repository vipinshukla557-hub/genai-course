# ─────────────────────────────────────────────────────────────────────────────
# WEEK 8 · SECTION 1: DOCUMENT LOADING
# ─────────────────────────────────────────────────────────────────────────────
# Applied GenAI Engineering Program
#
# WHAT THIS FILE COVERS:
#   - Creating a sample document so we have something to work with
#   - Loading documents using LangChain's document loaders
#   - Understanding the Document object (page_content + metadata)
#   - PyPDFLoader for PDFs, TextLoader for plain text
#
# ENVIRONMENT:
#   Python 3.13, LangChain 1.x
#   Required packages: langchain-community, pypdf
#
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: CREATE A SAMPLE DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────
# Before we can load documents, we need a document to load!
# We'll create a sample "company handbook" text file right here in code.
# This way you don't need to go find a PDF — the code creates one for you.
#
# In real projects, you'd get PDFs from clients, HR departments, legal teams,
# knowledge bases, etc. But for learning, we create our own.

# This is the content of our fake company handbook.
# We're using a fictional food delivery company called "Namaste Foods"
# based in Mumbai. The content has multiple sections so we can later
# see how chunking and retrieval work across different topics.

sample_handbook = """
NAMASTE FOODS — EMPLOYEE HANDBOOK
Last Updated: January 2025

=== CHAPTER 1: COMPANY OVERVIEW ===

Namaste Foods is a premium food delivery platform headquartered in Andheri, Mumbai.
Founded in 2020, we connect over 50,000 restaurants across 15 Indian cities with
hungry customers who want restaurant-quality food delivered to their doorstep.

Our mission is simple: deliver happiness, one meal at a time. We believe that great
food should be accessible to everyone, whether you're ordering a Rs 99 thali from
a local dhaba or a Rs 2,500 tasting menu from a five-star hotel.

Our core values are:
1. Customer Obsession — the customer's biryani should arrive hot, always.
2. Speed Without Shortcuts — fast delivery, but never at the cost of food quality.
3. Respect for Partners — our restaurant partners and delivery riders are family.
4. Data-Driven Decisions — we measure everything from delivery time to dal temperature.

=== CHAPTER 2: LEAVE POLICY ===

All full-time employees are entitled to the following leaves per calendar year:

Casual Leave (CL): 12 days per year. Can be taken in half-day units. Cannot be
carried forward to the next year. Must be applied at least 1 day in advance.

Sick Leave (SL): 10 days per year. Can be taken without prior notice but requires
a medical certificate if taken for more than 2 consecutive days. Unused sick leave
can be carried forward up to a maximum of 30 days.

Earned Leave (EL): 15 days per year. Earned at the rate of 1.25 days per month.
Can be carried forward up to a maximum of 45 days. Can be encashed at the time
of separation. Must be applied at least 7 days in advance for 3+ consecutive days.

Maternity Leave: 26 weeks as per the Maternity Benefit Act, 2017. Applicable to
women employees who have worked for at least 80 days in the 12 months before the
expected delivery date.

Paternity Leave: 5 working days. Must be taken within 30 days of the child's birth.

Public Holidays: 12 gazetted holidays per year as per the Maharashtra state list.
Additionally, employees may choose 2 restricted holidays from the provided list.

=== CHAPTER 3: WORK FROM HOME POLICY ===

Effective April 2024, Namaste Foods follows a hybrid work model:

Engineering Team: 3 days in office (Tuesday, Wednesday, Thursday), 2 days remote.
Operations Team: 5 days in office. Remote work not applicable due to on-ground needs.
Marketing and Sales: 2 days in office (Monday, Wednesday), 3 days remote.
HR and Finance: 3 days in office (Monday, Tuesday, Thursday), 2 days remote.

All employees must be available on Slack during core hours: 10:00 AM to 6:00 PM IST,
regardless of whether they are working from office or from home.

Home office setup: The company provides a one-time Rs 15,000 allowance for purchasing
a monitor, keyboard, mouse, or desk. Receipts must be submitted within 60 days of
purchase. Internet reimbursement of Rs 1,000 per month is provided for remote days.

=== CHAPTER 4: EXPENSE REIMBURSEMENT ===

Business Travel: All domestic flights must be booked in economy class. Hotels must
be booked through the company portal. Daily meal allowance during travel is Rs 1,500
in metro cities (Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Kolkata) and Rs 1,000
in non-metro cities.

Client Entertainment: Prior approval from your manager is required for expenses above
Rs 5,000. Receipts must be submitted within 7 days of the expense. Alcohol is not
reimbursable under any circumstances.

Software and Tools: Any software subscription above Rs 2,000 per month requires IT
team approval. Free trials are encouraged before purchasing.

=== CHAPTER 5: PERFORMANCE REVIEWS ===

Performance reviews happen twice a year: April (mid-year) and October (annual).

Rating Scale:
5 — Exceptional: Consistently exceeds all expectations. Top 5% of performers.
4 — Exceeds Expectations: Regularly delivers above what is expected.
3 — Meets Expectations: Solid, reliable performance. This is a good rating.
2 — Needs Improvement: Some goals not met. Improvement plan will be created.
1 — Unsatisfactory: Significant gaps. Formal PIP (Performance Improvement Plan) initiated.

Promotion eligibility requires a minimum rating of 4 in the most recent annual review
and at least 18 months in the current role. Skip-level promotions require VP approval.

Salary increments are linked to performance ratings:
Rating 5: 15-20% increment
Rating 4: 10-15% increment
Rating 3: 5-8% increment
Rating 2: 0% increment
Rating 1: 0% increment, formal PIP

=== CHAPTER 6: CODE OF CONDUCT ===

All employees must maintain the highest standards of professional behaviour.

Conflict of Interest: Employees must not hold any financial interest in a competing
food delivery platform (e.g., Swiggy, Zomato). Moonlighting for competitors is
strictly prohibited and is grounds for immediate termination.

Data Privacy: Customer data including names, addresses, phone numbers, and order
history is strictly confidential. Sharing customer data outside approved systems is
a terminable offence. All data handling must comply with the IT Act, 2000 and the
Digital Personal Data Protection Act, 2023.

Social Media: Employees may share their work experiences on social media but must
not disclose financial results, unreleased products, or internal strategies. When
in doubt, consult the PR team before posting.

Gifts and Hospitality: Employees may accept gifts from vendors or partners only
if the value is below Rs 3,000. Gifts above this amount must be declared to the
Ethics Committee. Cash gifts are never acceptable.
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: SAVE THE SAMPLE DOCUMENT TO A FILE
# ─────────────────────────────────────────────────────────────────────────────
# We'll save this text to a .txt file so we can load it with LangChain.
# In real projects, you'd skip this step — you'd already have the PDF
# sitting in a folder somewhere.

import os  # os module lets us work with files and directories
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# This is the filename we'll save our sample document as.
# We put it in the same folder where this script lives.
sample_file_path = BASE_DIR / "namaste_foods_handbook.txt"

# open() creates or overwrites a file.
#   "w" means "write mode" — it creates the file if it doesn't exist,
#   or overwrites it if it does.
#   encoding="utf-8" makes sure special characters (like ₹) don't break.
# The "with" statement automatically closes the file when we're done.
with open(sample_file_path, "w", encoding="utf-8") as f:
    f.write(sample_handbook)

# Confirm the file was created and show its size.
# os.path.getsize() returns the file size in bytes.
file_size = os.path.getsize(sample_file_path)
print(f"✅ Sample document saved: {sample_file_path}")
print(f"   File size: {file_size:,} bytes")
print(f"   Character count: {len(sample_handbook):,} characters")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: LOAD WITH TextLoader
# ─────────────────────────────────────────────────────────────────────────────
# LangChain provides "document loaders" — classes that know how to read
# different file types and convert them into a standard format called
# a "Document" object.
#
# TextLoader is the simplest loader. It reads a .txt file and returns
# the entire file content as ONE Document object.
#
# Why not just use open() and f.read()? Because LangChain's Document
# object carries METADATA alongside the text — things like the source
# filename, page number (for PDFs), etc. This metadata becomes crucial
# later when we want to show users WHERE an answer came from.

# Import TextLoader from LangChain's community package.
# In LangChain 1.x, document loaders live in langchain_community.
from langchain_community.document_loaders import TextLoader

# Create a loader instance. We tell it which file to load.
# encoding="utf-8" ensures it handles all characters correctly.
text_loader = TextLoader(
    file_path=sample_file_path,  # Path to our text file
    encoding="utf-8"             # Character encoding — always use utf-8
)

# .load() actually reads the file and returns a LIST of Document objects.
# For TextLoader, the list will have exactly ONE Document (the whole file).
text_docs = text_loader.load()

# Let's see what we got back.
# len() tells us how many Document objects are in the list.
print(f"📄 TextLoader returned {len(text_docs)} document(s)")
print()

# Let's look at the first (and only) Document object.
# Every Document has two parts:
#   1. page_content — the actual text (a string)
#   2. metadata — a dictionary with info ABOUT the document
first_doc = text_docs[0]  # Get the first item from the list (index 0)

print("--- Document Object ---")
print(f"Type: {type(first_doc)}")           # Should be: langchain_core.documents.Document
print(f"Content length: {len(first_doc.page_content):,} characters")
print(f"Metadata: {first_doc.metadata}")     # Shows source filename
print()

# Let's peek at the first 300 characters of the content.
# [:300] is Python slicing — it gives us characters from position 0 to 299.
print("--- First 300 characters ---")
print(first_doc.page_content[:300])
print("...")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: LOAD A PDF WITH PyPDFLoader
# ─────────────────────────────────────────────────────────────────────────────
# In real RAG projects, 90% of the documents you'll work with are PDFs.
# PyPDFLoader reads a PDF file page by page. Each PAGE becomes its own
# Document object, with the page number stored in metadata.
#
# This is important! If a PDF has 50 pages, PyPDFLoader returns a list
# of 50 Document objects. Each one has:
#   - page_content = the text from that page
#   - metadata = {"source": "file.pdf", "page": 0}  (page is 0-indexed)
#
# REQUIREMENT: You need the 'pypdf' package installed.
#   pip install pypdf
#
# NOTE: If you don't have a PDF file handy, skip this step.
# The rest of the sections will work with the text file we created above.
# But if you have ANY PDF (your resume, a textbook chapter, anything),
# try loading it here!

from langchain_community.document_loaders import PyPDFLoader

# ⬇️  CHANGE THIS to the path of any PDF you have on your computer.
#     Examples:
#       "my_resume.pdf"
#       "D:/Documents/report.pdf"
#       "company_policy.pdf"
#
# If you don't have a PDF right now, that's fine — comment out this
# whole block and move on. We'll use the text file for the rest.

pdf_file_path = BASE_DIR / "Resume_Vipin_Shukla.pdf"  # ← Change this to YOUR PDF path

if os.path.exists(pdf_file_path):
    # Create a PDF loader instance
    pdf_loader = PyPDFLoader(file_path=pdf_file_path)

    # .load() reads every page and returns a list of Documents
    pdf_docs = pdf_loader.load()

    print(f"📄 PyPDFLoader returned {len(pdf_docs)} document(s) (one per page)")
    print()

    # Let's look at the first page
    first_page = pdf_docs[0]
    print(f"--- Page 1 ---")
    print(f"Content length: {len(first_page.page_content)} characters")
    print(f"Metadata: {first_page.metadata}")
    print(f"Preview: {first_page.page_content[:200]}...")
    print()

    # And the last page (if there are multiple pages)
    if len(pdf_docs) > 1:
        last_page = pdf_docs[-1]  # [-1] means "last item in the list"
        print(f"--- Last Page (Page {len(pdf_docs)}) ---")
        print(f"Content length: {len(last_page.page_content)} characters")
        print(f"Metadata: {last_page.metadata}")
        print(f"Preview: {last_page.page_content[:200]}...")
        print()
else:
    print(f"⚠️  PDF file '{pdf_file_path}' not found.")
    print("   That's fine! We'll use the text file for the rest of this session.")
    print("   If you want to try PDF loading, change pdf_file_path above")
    print("   to point to any PDF on your computer.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: UNDERSTANDING THE DOCUMENT OBJECT
# ─────────────────────────────────────────────────────────────────────────────
# Let's really understand what a Document object IS, because everything
# in the RAG pipeline works with Documents.
#
# Think of a Document like a sticky note:
#   - Front side = the TEXT (page_content)
#   - Back side = INFO ABOUT the text (metadata)
#
# Analogy: Imagine you're a librarian. You have the actual book content
# (page_content), but you also have a catalogue card that says "this book
# is from shelf 3, row 2, published in 2024" (metadata). Both parts
# travel together.

from langchain_core.documents import Document

# You can also CREATE Document objects manually.
# This is useful when you're loading data from a database, API, or
# any source that doesn't have a built-in LangChain loader.

# Example: Creating documents from a list of Bollywood movie descriptions
movie_docs = [
    Document(
        page_content="Dangal (2016) is a biographical sports drama about Mahavir Singh Phogat, "
                     "a wrestler who trains his daughters Geeta and Babita to become world-class "
                     "wrestlers. Starring Aamir Khan. Directed by Nitesh Tiwari.",
        metadata={"source": "bollywood_db", "genre": "sports", "year": 2016}
    ),
    Document(
        page_content="3 Idiots (2009) follows three engineering students at ICE, a prestigious "
                     "institution. The film explores the pressures of the Indian education system "
                     "and the importance of following your passion. Starring Aamir Khan, "
                     "R. Madhavan, and Sharman Joshi.",
        metadata={"source": "bollywood_db", "genre": "comedy-drama", "year": 2009}
    ),
    Document(
        page_content="Lagaan (2001) is set in colonial India where villagers must win a cricket "
                     "match against British officers to avoid paying triple tax (lagaan). "
                     "Starring Aamir Khan. Nominated for the Academy Award for Best Foreign Film.",
        metadata={"source": "bollywood_db", "genre": "sports-drama", "year": 2001}
    ),
]

print("--- Manually Created Documents ---")
for i, doc in enumerate(movie_docs):
    # enumerate() gives us both the index number (i) and the item (doc)
    # as we loop through the list. i starts at 0, 1, 2, ...
    print(f"\nDocument {i + 1}:")                      # i+1 so it shows 1, 2, 3
    print(f"  Content: {doc.page_content[:80]}...")     # First 80 chars
    print(f"  Metadata: {doc.metadata}")                # The info dictionary
print()


# ─────────────────────────────────────────────────────────────────────────────
# KEY TAKEAWAYS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("KEY TAKEAWAYS — Section 1")
print("=" * 60)
print()
print("1. Document = page_content (text) + metadata (info about the text)")
print("2. TextLoader → whole file as ONE document")
print("3. PyPDFLoader → each PAGE as a separate document")
print("4. Metadata is crucial — it tells you WHERE an answer came from")
print("5. You can also create Document objects manually")
print()
print("Next: Section 2 — Chunking (splitting documents into smaller pieces)")
