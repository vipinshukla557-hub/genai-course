# ============================================================
# PART 1: Manual Message History
# ============================================================

# --- Setup (same as every week) ---
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Create the model ---
model = ChatOpenAI(model="gpt-4o-mini")

# --- Create the prompt template ---

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Swiggy customer support agent. Be friendly and concise."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])


# Build our chain

chain = prompt | model

chat_history = []

# turn 1  (First conversation turn  )

print("----Turn 1----")

response = chain.invoke({
    "chat_history": chat_history,
    "user_input": "Hi, My name is Debjit."
})

print("AI:", response.content)

chat_history.append(HumanMessage(content="Hi, My name is Debjit."))
chat_history.append(response)

# turn 2 (Second conversation turn)

print("\n----Turn 2----")

response = chain.invoke({
    "chat_history": chat_history,
    "user_input": "My order #123434 was supposed to be arrive by now but it never came"
})

print("AI:", response.content)

# append turn 2 messages to chat_history
chat_history.append(HumanMessage(content="My order #123434 was supposed to be arrive by now but it never came"))
chat_history.append(response)

# turn 3 (Third conversation turn)

print("\n----Turn 3----")

response = chain.invoke({
    "chat_history": chat_history,
    "user_input": "What was my first question, respond as is?"
})


# --- Append turn 3 messages to chat_history ---
chat_history.append(HumanMessage(content="What was my first question, respond as is?"))
chat_history.append(response)

print("AI:", response.content)

print("chat history length:", len(chat_history), "messages")

# ============================================================
# PART 2: Handling Long Conversations
# ============================================================

print("\n\n========== PART 2: Handling Long History ==========\n")

# --- Solution 1: Simple slicing (keep last 4 messages = last 2 turns) ---

print("Before slicing:", len(chat_history), "messages")

trimmed_history = chat_history[-4:]

print("After slicing:", len(trimmed_history), "messages")
print("Kept messages:")

for i in range(len(trimmed_history)):
    msg = trimmed_history[i]
    print(f"  {i}: {msg.type} - {msg.content}")



# --- Solution 2: trim_messages (the production approach) ---

from langchain_core.messages import trim_messages

trimmed = trim_messages(chat_history, max_tokens=100, token_counter=model,strategy="last", include_system=True)

print("trimmed message max 100 tokens")

print("Before trimming:", len(chat_history), "messages")
print("After trimming:", len(trimmed), "messages")

for i in range(len(trimmed)):
    msg = trimmed[i]
    print(f"  {i}: {msg.type} - {msg.content}")


## Summarising the conversation to reduce token count




# ============================================================
# PART 3: Saving and Loading History (JSON Persistence)
# ============================================================

print("\n\n========== PART 3: JSON Persistence ==========\n")

import json

def save_history(history, filename):
    """Save chat history to a JSON file."""
    data = []
    for i in range(len(history)):
        msg = history[i]
        data.append({
            "type": msg.type,
            "content": msg.content
        })

    # "w" → open file for writing (creates if doesn't exist)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} messages to {filename}")


def load_history(filename):
    """Load chat history from a JSON file."""
    # "r" → open file for reading
    with open(filename, "r") as f:
        data = json.load(f)

    history = []
    for i in range(len(data)):
        item = data[i]
        # item["type"] is either "human" or "ai"
        if item["type"] == "human":
            history.append(HumanMessage(content=item["content"]))
        else:
            history.append(AIMessage(content=item["content"]))

    print(f"Loaded {len(history)} messages from {filename}")
    return history


save_history(chat_history, "chat_history.json")


# Failure occurs here

chat_history = []

print("Cleared chat history, length:", len(chat_history))

chat_history = load_history("chat_history.json")

response = chain.invoke({
    "chat_history": chat_history,
    "user_input": "What was my name, order number?"
})

print("AI:", response.content)




