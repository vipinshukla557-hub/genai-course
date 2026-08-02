"""
Day 5 Practice

Implements all 4 next steps from Test.PY:
1. Modified Pydantic models with new fields
2. Custom text inputs
3. Groq instead of OpenAI (swap in SETUP below)
4. New domain extraction chain - Resume Parser
"""

# ============================================================
# SETUP
# ============================================================
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

import httpx
# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ===================================================================
# PRACTICE 1: Invoice Parser - NEW FIELDS ADDED
# New fields added to LineItem: discount_percent
# New fields added to Invoice: payment_terms, bank_details, due_date
# ===================================================================


#
# Real-world scenario: Your company receives invoices as
# unstructured text (emails, scanned PDFs). You need to
# extract structured data for the accounting system.
#
# This exercise teaches:
# - Nested Pydantic models (a model inside a model)
# - List fields (multiple line items in one invoice)
# - Optional fields (some invoices have GST, some don't)
# - Computed/derived fields
# ============================================================

# ============================================================
# Define nested models
# LineItem is used INSIDE Invoice
# Think of it like a table row inside a bigger form
# ============================================================
class LineItem(BaseModel):
    description: str = Field(
        description="What was purchased"
    )
    quantity: int = Field(
        description="Number of units"
    )
    unit_price: float = Field(
        description="Price per unit in INR"
    )
    total: float = Field(
        description="quantity * unit_price in INR"
    )
    discount_percent: Optional[float] = Field(
        description="Discount percentage applied on this line item, if any"
    )

# ============================================================
# The main Invoice model contains a LIST of LineItem objects
# List[LineItem] -> Pydantic knows to expect an array of objects
# Optional[str] -> this field can be None (not all invoices have GST)
# ============================================================
class Invoice(BaseModel):
    vendor_name: str = Field(
        description="Company or person who sent the invoice"
    )
    invoice_number: str = Field(
        description="Invoice reference number"
    )
    date: str = Field(
        description="Invoice date in DD-MM-YYYY format"
    )
    due_date: Optional[str] = Field(
            description="Payment due date in DD-MM-YYYY format if mentioned"
    )
    line_items: List[LineItem] = Field(
        description="List of items/services in the invoice"
    )
    subtotal: float = Field(
        description="Sum of all line item totals in INR"
    )
    gst_number: Optional[str] = Field(
        default=None,
        description="GST registration number if mentioned, otherwise null"
    )
    gst_amount: Optional[float] = Field(
        default=None,
        description="GST amount in INR if mentioned, otherwise null"
    )
    grand_total: float = Field(
        description="Final payable amount including GST in INR"
    )
    
# ============================================================
# Create parser and chain
# ============================================================
invoice_parser = PydanticOutputParser(pydantic_object=Invoice)

invoice_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an accounts payable assistant. "
     "Extract invoice details from unstructured text. "
     "Convert all amounts to INR. "
     "Always respond in the exact format requested."),
    ("human",
     "Extract invoice data from this text:\n\n"
     "{invoice_text}\n\n"
     "{format_instructions}")
])

invoice_chain = invoice_prompt | model | invoice_parser

# ============================================================
# Test with a realistic messy invoice text
# Notice: dates are in different formats, amounts have Rs prefix,
# GST is mentioned informally. The LLM must normalize all of this.
# ============================================================
invoice_text = """
From: DesignCraft Studio
Invoice #: DC-2026-0112
Date: 3rd August 2026
Due Date: 18th August 2026
GSTIN: 29AABCD5678G1ZQ

Items:
- Logo design package: 1 unit x Rs 15,000 = Rs 15,000 (10% early bird discount applied = Rs 13,500)
- Social media creatives (10 posts): 10 units x Rs 800/post = Rs 8,000
- Brand guidelines document: 1 unit x Rs 5,000 = Rs 5,000

Subtotal: Rs 26,500
GST @18%: Rs 4,770
Total Payable: Rs 31,270

Payment via NEFT: Axis Bank, A/C: 9210045678, IFSC: UTIB0001234
Please pay within 15 days to avail early bird discount.
"""

print("=" * 60)
print("EXERCISE 1: Invoice Parser")
print("=" * 60)

invoice_result = invoice_chain.invoke({
    "invoice_text": invoice_text,
    "format_instructions": invoice_parser.get_format_instructions()
})

print(f"Vendor:     {invoice_result.vendor_name}")
print(f"Invoice #:  {invoice_result.invoice_number}")
print(f"Date:       {invoice_result.date}")
print(f"Due Date:      {invoice_result.due_date}")
print(f"GST #:      {invoice_result.gst_number}")
print(f"\nLine Items:")
for i in range(len(invoice_result.line_items)):
    item = invoice_result.line_items[i]
    print(f"  {i+1}. {item.description}: {item.quantity} x Rs {item.unit_price} = Rs {item.total}")
print(f"\nSubtotal:   Rs {invoice_result.subtotal}")
print(f"GST:        Rs {invoice_result.gst_amount}")
print(f"Total:      Rs {invoice_result.grand_total}")
print()

# ============================================================
# PRACTICE 2: Support Ticket - CUSTOM TEXT INPUTS
# ============================================================

#
# Sometimes you want a field to be one of a fixed set of values.
# For example, sentiment must be "positive", "negative", or "neutral".
# Not "kinda good" or "somewhat bad".
#
# Python Enum + Pydantic enforces this strictly.
# If the LLM returns a value not in the enum, Pydantic rejects it.
# ============================================================

# ============================================================
# Define enums for strict categorization
# These are like dropdown menus: the value MUST be one of these options
# ============================================================
class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"
    mixed = "mixed"

class Priority(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"

class Department(str, Enum):
    delivery = "delivery"
    food_quality = "food_quality"
    payment = "payment"
    app_issues = "app_issues"
    other = "other"

# ============================================================
# The model uses enum types instead of plain str
# Pydantic will reject any value not in the enum
# ============================================================
class SupportTicket(BaseModel):
    customer_name: str = Field(description="Customer's name")
    order_id: Optional[str] = Field(
        default=None,
        description="Order ID if mentioned"
    )
    department: Department = Field(
        description="Which department should handle this"
    )
    priority: Priority = Field(
        description="Urgency level based on the complaint"
    )
    sentiment: Sentiment = Field(
        description="Overall customer sentiment"
    )
    issue_summary: str = Field(
        description="Brief summary in under 20 words"
    )
    suggested_response: str = Field(
        description="Draft a 1-2 sentence response to the customer"
    )

ticket_parser = PydanticOutputParser(pydantic_object=SupportTicket)

ticket_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI routing system for a food delivery app's customer support. "
     "Analyze the complaint and create a support ticket. "
     "Be precise with department and priority classification."),
    ("human",
     "Create a support ticket from this message:\n\n"
     "{message}\n\n"
     "{format_instructions}")
])

ticket_chain = ticket_prompt | model | ticket_parser

# ============================================================
# Test with different types of complaints
# ============================================================
complaints = [
    "Hi, I'm Priya. I placed order SW-98765 on Swiggy 2 hours ago and it still shows 'preparing'. The restaurant is only 2 km away. My food must be cold by now. What is happening?",
    "The Zomato app keeps crashing every time I try to open my order history. I've reinstalled it twice. Running on iPhone 14, iOS 17. Order ZOM-33210 is missing from my history too.",
    "I ordered a veg thali but received a non-veg meal. I'm a strict vegetarian. Order ID: ZOM-77541. This is completely unacceptable and I want a full refund immediately."
]

print("=" * 60)
print("EXERCISE 2: Support Ticket with Enum Validation")
print("=" * 60)

for i in range(len(complaints)):
    print(f"\n--- Complaint {i+1} ---")
    ticket = ticket_chain.invoke({
        "message": complaints[i],
        "format_instructions": ticket_parser.get_format_instructions()
    })
    print(f"Customer:   {ticket.customer_name}")
    print(f"Department: {ticket.department.value}")
    print(f"Priority:   {ticket.priority.value}")
    print(f"Sentiment:  {ticket.sentiment.value}")
    print(f"Summary:    {ticket.issue_summary}")
    print(f"Response:   {ticket.suggested_response}")
print()

# ============================================================
# PRACTICE 4: New Domain - Resume Parser
# Extracts structured data from unstructured resume text
# ============================================================

class Experience(BaseModel):
    company: str = Field(
        description="Company name"
    )
    role: str = Field(
        description="Job title or role"
    )
    duration: str = Field(
        description="Duration e.g. Jan 2021 - Mar 2023"
    )
    highlights: List[str] = Field(
        description="Key achievements or responsibilities"
    )

class Resume(BaseModel):
    name: str = Field(
        description="Candidate's full name"
    )
    email: Optional[str] = Field(
        default=None, description="Email address if mentioned"
    )
    phone: Optional[str] = Field(
        default=None, description="Phone number if mentioned"
    )
    total_experience_years: float = Field(
        description="Total years of work experience"
    )
    current_role: str = Field(
        description="Most recent job title"
    )
    skills: List[str] = Field(
        description="Technical and soft skills mentioned"
    )
    experience: List[Experience] = Field(
        description="Work experience entries"
    )
    education: str = Field(
        description="Highest qualification and institution"
    )
    is_open_to_relocation: bool = Field(
        description="Whether candidate mentions willingness to relocate"
        )

resume_parser = PydanticOutputParser(pydantic_object=Resume)

resume_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an HR assistant that extracts structured data from resumes. "
     "Be precise and extract only what is explicitly mentioned."),
    ("human",
     "Extract resume data from this text:\n\n"
     "{resume_text}\n\n"
     "{format_instructions}")
])

resume_chain = resume_prompt | model | resume_parser

resume_text = """
Ananya Sharma
ananya.sharma@email.com | +91-9876543210 | Bangalore

SUMMARY
Full-stack developer with 4.5 years of experience building scalable web applications.
Open to relocation for the right opportunity.

EXPERIENCE

Senior Software Engineer - Infosys, Bangalore
Jan 2023 - Present
- Led migration of monolithic app to microservices, reducing latency by 40%
- Mentored a team of 4 junior developers
- Built REST APIs serving 2M+ daily requests using FastAPI and PostgreSQL

Software Engineer - Wipro, Hyderabad
Jul 2020 - Dec 2022
- Developed React dashboards for 3 enterprise clients
- Automated CI/CD pipelines using Jenkins and Docker
- Reduced deployment time from 2 hours to 15 minutes

SKILLS
Python, FastAPI, React, PostgreSQL, Docker, Kubernetes, AWS, Git, Agile

EDUCATION
B.Tech in Computer Science - VIT University, 2020
"""

print("=" * 60)
print("PRACTICE 4: Resume Parser (New Domain)")
print("=" * 60)

resume_result = resume_chain.invoke({
    "resume_text": resume_text,
    "format_instructions": resume_parser.get_format_instructions()
})

print(f"Name:            {resume_result.name}")
print(f"Email:           {resume_result.email}")
print(f"Phone:           {resume_result.phone}")
print(f"Current Role:    {resume_result.current_role}")
print(f"Experience:      {resume_result.total_experience_years} years")
print(f"Open to Relocate: {'Yes' if resume_result.is_open_to_relocation else 'No'}")
print(f"Education:       {resume_result.education}")
print(f"\nSkills: {', '.join(resume_result.skills)}")
print(f"\nWork History:")
for exp in resume_result.experience:
    print(f"  {exp.role} @ {exp.company} ({exp.duration})")
    for h in exp.highlights:
        print(f"    - {h}")
print()

print("=" * 60)
print("All 4 practice exercises completed!")
print("=" * 60)
