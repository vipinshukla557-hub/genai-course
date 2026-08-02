"""
Week 5: LangChain Fundamentals - Homework & Practice
Applied GenAI Engineering Program | GenAIEducate

This file contains advanced scenarios for practice after the session.
Each section builds on what we covered in class and pushes further.
Run each section one at a time by uncommenting the relevant block.

Topics covered:
1. Multi-field invoice parser (nested Pydantic models)
2. Enum validation for strict field values
3. Multiple extraction from a single text (batch processing)
4. Conversation history with MessagesPlaceholder (full chatbot loop)
5. with_structured_output() with complex models
6. RetryWithErrorOutputParser (advanced auto-retry)
7. Comparing raw OpenAI SDK vs LangChain (side-by-side)

Run: python week5_homework.py
"""

# ============================================================
# SETUP
# ============================================================
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# EXERCISE 1: Invoice Parser with Nested Pydantic Models
# ============================================================
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
From: TechServe Solutions Pvt Ltd
Invoice #: TS-2026-0847
Date: 15th July 2026
GSTIN: 27AABCT1234F1ZP

Items:
- Cloud hosting (AWS) for June 2026: 3 months x Rs 12,000/month = Rs 36,000
- API integration development: 40 hours x Rs 2,500/hour = Rs 1,00,000
- SSL certificate renewal: 1 unit x Rs 4,500 = Rs 4,500

Subtotal: Rs 1,40,500
GST @18%: Rs 25,290
Total Payable: Rs 1,65,790

Please process payment within 15 days.
Bank: HDFC Bank, A/C: 50100123456789, IFSC: HDFC0001234
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
# EXERCISE 2: Enum Validation for Strict Categories
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
    "Hi, I'm Rahul from Mumbai. My order ZOM-44521 was charged twice on my card. Rs 800 deducted two times. Please refund the extra amount. I've attached my bank statement.",
    "WORST EXPERIENCE EVER. Ordered from Barbeque Nation on Swiggy, order SW-12093. Got someone else's order. Completely different items. My kids were waiting for their pizza and got some random dal chawal. Fix this NOW.",
    "Hey, just wanted to say the delivery was super quick today! Got my dosa in 18 minutes. The delivery partner Amit was really polite too. Keep it up!"
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
# EXERCISE 3: Batch Processing - Multiple Extractions
# ============================================================
#
# .batch() lets you process multiple inputs at once
# Instead of calling .invoke() in a loop (one at a time),
# .batch() sends all requests and processes them efficiently.
# Same chain, different way to run it.
# ============================================================

print("=" * 60)
print("EXERCISE 3: Batch Processing")
print("=" * 60)

# ============================================================
# Prepare a list of inputs (each is a dictionary)
# .batch() takes a list and returns a list of results
# ============================================================
batch_inputs = [
    {
        "message": complaint,
        "format_instructions": ticket_parser.get_format_instructions()
    }
    for complaint in complaints
]

# ============================================================
# Process all three complaints in one .batch() call
# This is more efficient than three separate .invoke() calls
# because LangChain can parallelize the API calls
# ============================================================
batch_results = ticket_chain.batch(batch_inputs)

print(f"Processed {len(batch_results)} complaints in one batch call")
for i in range(len(batch_results)):
    print(f"  {i+1}. {batch_results[i].customer_name} -> {batch_results[i].priority.value} priority")
print()


# ============================================================
# EXERCISE 4: Full Chatbot Loop with MessagesPlaceholder
# ============================================================
#
# This builds a complete interactive chatbot that:
# - Maintains conversation history across turns
# - Uses MessagesPlaceholder to inject history into the prompt
# - Grows the history list after each exchange
#
# This is the exact pattern you will use in Week 6.
# ============================================================

print("=" * 60)
print("EXERCISE 4: Chatbot with Conversation History")
print("=" * 60)

# ============================================================
# The prompt has three parts:
# 1. System message -> sets the chatbot's personality
# 2. MessagesPlaceholder -> slot for all previous messages
# 3. Human message -> the current question
# ============================================================
chatbot_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly customer support agent for Zomato. "
     "Be helpful, empathetic, and concise. "
     "If you don't have enough information, ask clarifying questions. "
     "Keep responses under 3 sentences."),
    MessagesPlaceholder("chat_history"),
    ("human", "{user_input}")
])

chatbot_chain = chatbot_prompt | model | StrOutputParser()

# ============================================================
# Simulate a multi-turn conversation
# We start with an empty history and grow it each turn
# After each exchange, we add the user message AND AI response
# to the history list
# ============================================================
chat_history = []

# ============================================================
# List of user messages to simulate a conversation
# ============================================================
user_messages = [
    "Hi, I have a problem with my recent order",
    "Order ID is ZOM-55123. I got paneer butter masala but I ordered chicken biryani.",
    "Yes please, I'd like a replacement delivered as soon as possible",
    "Thanks! How long will it take?"
]

for i in range(len(user_messages)):
    user_msg = user_messages[i]

    # ============================================================
    # Call the chain with current history + new question
    # ============================================================
    response = chatbot_chain.invoke({
        "chat_history": chat_history,
        "user_input": user_msg
    })

    print(f"User: {user_msg}")
    print(f"Bot:  {response}")
    print()

    # ============================================================
    # Add this exchange to history for next turn
    # HumanMessage -> records what the user said
    # AIMessage -> records what the AI responded
    # Next time the chain runs, it will see all previous exchanges
    # ============================================================
    chat_history.append(HumanMessage(content=user_msg))
    chat_history.append(AIMessage(content=response))


# ============================================================
# EXERCISE 5: with_structured_output() with Complex Models
# ============================================================
#
# with_structured_output() works great for complex models too
# No parser needed, no format instructions, cleaner code
# But you still need well-defined Pydantic models
# ============================================================

print("=" * 60)
print("EXERCISE 5: Complex Extraction with with_structured_output()")
print("=" * 60)

# ============================================================
# A complex model for analyzing product reviews
# Includes nested lists, optional fields, and computed insights
# ============================================================
class ProductReview(BaseModel):
    product_name: str = Field(description="Name of the product reviewed")
    reviewer_name: str = Field(description="Name of the reviewer if mentioned")
    rating_out_of_5: float = Field(description="Rating from 1.0 to 5.0")
    pros: List[str] = Field(description="List of positive points mentioned")
    cons: List[str] = Field(description="List of negative points mentioned")
    would_recommend: bool = Field(
        description="Whether the reviewer would recommend this product"
    )
    key_quote: str = Field(
        description="The most impactful sentence from the review"
    )

# ============================================================
# Use with_structured_output() instead of PydanticOutputParser
# The model directly returns a ProductReview object
# ============================================================
structured_model = model.with_structured_output(ProductReview)

review_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a product review analyzer. "
     "Extract structured insights from customer reviews."),
    ("human", "Analyze this product review:\n\n{review}")
])

review_chain = review_prompt | structured_model

# ============================================================
# Test with an Amazon India style review
# ============================================================
review_text = """
Review by Arjun K. for boAt Rockerz 450 (Bluetooth Headphones)

I bought these headphones during the Great Indian Festival sale for Rs 1,499.
The sound quality is really good for this price range, especially the bass.
Battery life is fantastic, I've been using them for 3 days straight without
charging. The cushions are comfortable even for long work-from-home calls.

However, the build quality feels a bit plasticky. The folding mechanism is
slightly loose. Also, the Bluetooth range is not great, it cuts out if I
walk more than 5 meters from my laptop. The carrying pouch they included
is basically useless.

Overall, for the price, these are an absolute steal. Would definitely
recommend for anyone looking for budget wireless headphones.
Rating: 4/5
"""

review_result = review_chain.invoke({"review": review_text})

print(f"Product:     {review_result.product_name}")
print(f"Reviewer:    {review_result.reviewer_name}")
print(f"Rating:      {review_result.rating_out_of_5}/5")
print(f"Recommend:   {'Yes' if review_result.would_recommend else 'No'}")
print(f"Key Quote:   \"{review_result.key_quote}\"")
print(f"\nPros:")
for pro in review_result.pros:
    print(f"  + {pro}")
print(f"\nCons:")
for con in review_result.cons:
    print(f"  - {con}")
print()


# ============================================================
# EXERCISE 6: RetryWithErrorOutputParser (Advanced Auto-Retry)
# ============================================================
#
# OutputFixingParser (from live demo) sends the bad output + error
# to the LLM to fix.
#
# RetryWithErrorOutputParser goes further: it also sends the
# ORIGINAL PROMPT back to the LLM. This gives the LLM full
# context about what was asked, making corrections more accurate.
#
# Use this when OutputFixingParser is not enough.
# ============================================================

print("=" * 60)
print("EXERCISE 6: RetryWithErrorOutputParser")
print("=" * 60)

from langchain_classic.output_parsers import RetryWithErrorOutputParser

# ============================================================
# A strict model where the LLM might struggle with the format
# ============================================================
class MeetingAction(BaseModel):
    attendee: str = Field(description="Person responsible for the action")
    action: str = Field(description="What they need to do")
    deadline: str = Field(description="Due date in DD-MM-YYYY format")
    priority: str = Field(description="high, medium, or low")

action_parser = PydanticOutputParser(pydantic_object=MeetingAction)

# ============================================================
# Create the retry parser
# retry_chain -> the prompt template to use when retrying
# This gives the LLM the original prompt context, not just the error
# ============================================================
retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=action_parser,
    llm=model
)

# ============================================================
# Create the prompt for the original extraction
# ============================================================
action_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract action items from meeting notes."),
    ("human",
     "Extract the main action item from this:\n\n"
     "{meeting_notes}\n\n"
     "{format_instructions}")
])

meeting_notes = "Vikram said he'll finish the API docs by end of next week. This is high priority since the client demo is on the 28th."

# ============================================================
# Format the prompt (we need the formatted prompt for the retry parser)
# .invoke() on a prompt template returns a formatted prompt value
# ============================================================
formatted_prompt = action_prompt.invoke({
    "meeting_notes": meeting_notes,
    "format_instructions": action_parser.get_format_instructions()
})

# ============================================================
# Simulate a bad response and fix it with the retry parser
# This response has the deadline in wrong format (should be DD-MM-YYYY)
# and is missing proper JSON structure
# ============================================================
bad_response = "Vikram needs to finish API docs by next Friday, high priority"

# ============================================================
# retry_parser.parse_with_prompt() takes both the bad output
# AND the original prompt, giving the LLM full context to fix it
# ============================================================
fixed_action = retry_parser.parse_with_prompt(
    bad_response,
    formatted_prompt
)

print(f"Attendee: {fixed_action.attendee}")
print(f"Action:   {fixed_action.action}")
print(f"Deadline: {fixed_action.deadline}")
print(f"Priority: {fixed_action.priority}")
print()


# ============================================================
# EXERCISE 7: Raw OpenAI SDK vs LangChain - Side by Side
# ============================================================
#
# This exercise shows the SAME task done two ways:
# 1. Raw OpenAI SDK (what you learned in Month 1)
# 2. LangChain LCEL chain
#
# Purpose: understand when LangChain adds value and when
# the raw SDK is simpler. There is no "always better" answer.
# ============================================================

print("=" * 60)
print("EXERCISE 7: Raw SDK vs LangChain Comparison")
print("=" * 60)

# ============================================================
# METHOD 1: Raw OpenAI SDK
# You handle everything: messages, API call, response extraction,
# JSON parsing, error handling
# ============================================================
import json
from openai import OpenAI

client = OpenAI()

# ============================================================
# With the raw SDK, you manually:
# 1. Build the messages list
# 2. Call the API
# 3. Extract the content from the response
# 4. Parse the JSON string
# 5. Handle any parsing errors
# ============================================================
raw_response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": "You are a data extraction system. Respond only with valid JSON."
        },
        {
            "role": "user",
            "content": (
                "Extract the person's name and city from this text. "
                "Respond as JSON with keys 'name' and 'city'.\n\n"
                "Text: Hi, I'm Meera and I live in Bangalore."
            )
        }
    ]
)

# ============================================================
# Manual extraction: dig into the response object for the text
# Then parse the JSON string into a Python dictionary
# If the JSON is malformed, this crashes with no auto-fix
# ============================================================
raw_text = raw_response.choices[0].message.content
raw_data = json.loads(raw_text)
print("RAW SDK Result:")
print(f"  Name: {raw_data['name']}")
print(f"  City: {raw_data['city']}")
print()

# ============================================================
# METHOD 2: LangChain LCEL
# Components handle the flow. Cleaner. Reusable. Swappable.
# ============================================================

class PersonInfo(BaseModel):
    name: str = Field(description="Person's full name")
    city: str = Field(description="City where they live")

lc_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data extraction system."),
    ("human", "Extract the person's name and city from this text:\n\n{text}")
])

# ============================================================
# with_structured_output -> no parser, no JSON handling, no manual extraction
# The chain returns a PersonInfo object directly
# ============================================================
lc_chain = lc_prompt | model.with_structured_output(PersonInfo)

lc_result = lc_chain.invoke({"text": "Hi, I'm Meera and I live in Bangalore."})
print("LANGCHAIN Result:")
print(f"  Name: {lc_result.name}")
print(f"  City: {lc_result.city}")
print()

# ============================================================
# THE VERDICT:
# For this simple case? The raw SDK is fine. Fewer dependencies.
# For complex pipelines with retries, streaming, conversation
# history, model swapping? LangChain saves you hundreds of lines.
# Know both. Use the right tool for the job.
# ============================================================


print("=" * 60)
print("All 7 exercises completed!")
print("=" * 60)
print()
print("NEXT STEPS:")
print("1. Try modifying the Pydantic models to add new fields")
print("2. Test with your own text inputs")
print("3. Try swapping OpenAI for Groq (3-line change)")
print("4. Build your own extraction chain for a domain you care about")
print("5. Read the LangChain docs: python.langchain.com")
