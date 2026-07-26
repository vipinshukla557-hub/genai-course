# ============================================================
# SETUP: Install these packages first (run in terminal)
# pip install langchain langchain-openai python-dotenv
# For the optional Groq/Llama configuration below:
# pip install langchain-groq
# ============================================================
 
import os
from dotenv import load_dotenv, find_dotenv
 
load_dotenv(find_dotenv())
 
# ============================================================
# PART 1: The Simplest LCEL Chain
# ============================================================
 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Optional Groq support (leave commented when using OpenAI):
from langchain_groq import ChatGroq
 
# model = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0
# )

# To use Groq instead, add GROQ_API_KEY to .env, then comment out the
# ChatOpenAI model above and uncomment the following model configuration:
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)
 
simple_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Song composer. Write songs in a poetic way."),
    ("human", "Compose a song about {topic} in 4 lines.")
])
 
 
simple_chain = simple_prompt | model | StrOutputParser()
 
print("PART 1: Simple LCEL Chain")
result = simple_chain.invoke({"topic": "LangChain"})
print(result)

print("Streaming Output:")
for chunk in simple_chain.stream({"topic": "Cricket"}):
    print(chunk, end="", flush=True)
print()


# ====part2:Structured Output Parsing with LCEL====
from pydantic import BaseModel, Field
 
class CustomerComplaint(BaseModel):
    customer_name: str = Field(description="Full name of the customer")
    issue_type: str = Field(description="Category: delivery_delay, wrong_item, quality, payment, other")
    summary: str = Field(description="One-line professional summary of the complaint")
 
structured_model = model.with_structured_output(CustomerComplaint)
 
 
extraction_prompt = ChatPromptTemplate.from_messages([
    ("system","You are a customer service data extraction system for Swiggy. Extract structured information from the following complaint"),
    ("human","Extract the customer details from complaint:\n\n{complaint}")
])
 
modern_chain = extraction_prompt | structured_model
 
messy_complaint = """
hii my name is Priya Sharma and I ordered biryani from Hyderabad
House yesterday, order number SW-78342. The delivery guy came
after 1.5 hours and the food was completely cold!! I paid Rs 450
for this. This is not acceptable at all. I want a full refund
immediately. Very disappointed with swiggy service.
"""
print("Structured output:")
 
result = modern_chain.invoke({"complaint": messy_complaint})
 
print(f"Name:    {result.customer_name}")
print(f"Issue:   {result.issue_type}")
print(f"Summary: {result.summary}")
# ============================================================
# PART 3: Chatbot with history
# ============================================================
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Swiggy customer service assistant."),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])

chat_chain = chat_prompt | model | StrOutputParser()
 
history = [
    HumanMessage(content="I ordered biryani but got pizza instead"),
    AIMessage(content="I'm sorry to hear that! Could you share your order ID so I can look into this?"),
    HumanMessage(content="Sure, it's SW-99201"),
    AIMessage(content="Thank you. I can see order SW-99201. I'll initiate a replacement for your biryani right away.")
]
 
print("\nPART 3: Chatbot with Conversation History")
 
response = chat_chain.invoke({
    "chat_history": history,
    "question": "Thanks. How long will the replacement take?"
})
 
print(f"question: How long will the replacement take?")
 
print(response)
