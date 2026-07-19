"""
=============================================================================
Applied GenAI Engineering Program
Week 4 · Section 1: The Messages Array — How Conversations Actually Work
=============================================================================

WHAT THIS FILE DOES:
  Shows you exactly how the OpenAI-compatible API structures conversations.
  Every message has a ROLE and CONTENT. The AI has no memory — the messages
  array IS the conversation. Understanding this is the foundation of
  everything we build today.

WHY THIS MATTERS:
  When you build a chatbot, you don't just send one message — you send the
  ENTIRE conversation every single time. If you don't understand the messages
  array, you can't build a chatbot. It's that simple.

CONCEPTS COVERED:
  1. The three roles: system, user, assistant
  2. Multi-turn conversations (sending history back each time)
  3. Why the API is stateless — every call starts fresh

PREREQUISITES:
  - .env file with OPENAI_API_KEY=your_key_here
  - pip install openai python-dotenv
=============================================================================
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── OPENAI CLIENT SETUP ────────────────────────────────────────────────────
# The OpenAI SDK is the industry standard — most providers are compatible.
# To switch to Groq (free): uncomment the two lines below and swap the model.
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",   # ← uncomment for Groq
    api_key=os.getenv("GROQ_API_KEY"),             # ← change to GROQ_API_KEY for Groq
)
# MODEL = "gpt-4o-mini"
MODEL = "llama-3.3-70b-versatile"                 # ← Groq model


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: THE THREE ROLES
# ═══════════════════════════════════════════════════════════════════════════
# Every message in the API has a "role":
#
#   "system"    → Instructions for the AI. Sets persona, rules, tone.
#                 The AI follows this but never shows it to the user.
#
#   "user"      → What the human says. This is the input.
#
#   "assistant" → What the AI said previously. You include past AI responses
#                 so the AI knows what it already said.
#
# Think of it like a movie script:
#   system    = the director's notes (how to act)
#   user      = the other actor's lines
#   assistant = your character's previous lines

print("=" * 60)
print("  PART 1: Single Message — The Simplest API Call")
print("=" * 60)

# The simplest possible API call: one system message + one user message
messages = [
    {"role": "system", "content": "You are a helpful assistant. Keep answers to 2 sentences."},
    {"role": "user", "content": "What is Python?"},
]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.7,
)

print(f"\n  User:      What is Python?")
print(f"  Assistant: {response.choices[0].message.content}")
print(f"  Tokens:    {response.usage.total_tokens}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: MULTI-TURN — THE AI HAS NO MEMORY
# ═══════════════════════════════════════════════════════════════════════════
# This is the most important concept in building chatbots:
#
#   The AI does NOT remember previous calls.
#   Every API call is completely independent.
#   If you want the AI to "remember" the conversation,
#   YOU must send the entire history every time.
#
# Watch what happens when we DON'T send history:

print("\n" + "=" * 60)
print("  PART 2: What Happens WITHOUT History")
print("=" * 60)

# First call: tell the AI your name
messages_call_1 = [
    {"role": "system", "content": "You are a friendly assistant. Keep answers brief."},
    {"role": "user", "content": "My name is Saurav."},
]

response_1 = client.chat.completions.create(
    model=MODEL,
    messages=messages_call_1
    )
ai_reply_1 = response_1.choices[0].message.content
print(f"\n  Call 1 — User:      My name is Saurav.")
print(f"  Call 1 — Assistant: {ai_reply_1}")

# Second call: ask the AI what your name is — but WITHOUT sending history
messages_call_2 = [
    {"role": "system", "content": "You are a friendly assistant. Keep answers brief."},
    {"role": "user", "content": "What is my name?"},
]

response_2 = client.chat.completions.create(
    model=MODEL, 
    messages=messages_call_2
    )
ai_reply_2 = response_2.choices[0].message.content
print(f"\n  Call 2 — User:      What is my name?")
print(f"  Call 2 — Assistant: {ai_reply_2}")
print(f"\n  ⚠️  The AI doesn't know! Because we didn't send the history.")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: MULTI-TURN — WITH HISTORY (THE RIGHT WAY)
# ═══════════════════════════════════════════════════════════════════════════
# Now let's do it correctly: include the previous exchange in the messages.

print("\n" + "=" * 60)
print("  PART 3: With History — The AI 'Remembers'")
print("=" * 60)

# Same second question, but now we include the full conversation
messages_call_2_fixed = [
    {"role": "system", "content": "You are a friendly assistant. Keep answers brief."},
    {"role": "user", "content": "My name is Saurav."},           # ← previous user message
    {"role": "assistant", "content": ai_reply_1},                 # ← previous AI reply
    {"role": "user", "content": "What is my name?"},              # ← new question
]

response_2_fixed = client.chat.completions.create(
    model=MODEL,
    messages=messages_call_2_fixed
    )
ai_reply_2_fixed = response_2_fixed.choices[0].message.content
print(f"\n  Call 2 (with history) — User:      What is my name?")
print(f"  Call 2 (with history) — Assistant: {ai_reply_2_fixed}")
print(f"\n  ✅ Now it knows! Because we sent the full conversation.")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4: VISUALIZE THE MESSAGES ARRAY GROWING
# ═══════════════════════════════════════════════════════════════════════════
# Let's run a 3-turn conversation and print the messages array at each step.

print("\n" + "=" * 60)
print("  PART 4: Watch the Messages Array Grow")
print("=" * 60)

# Start with system message
history = [
    {"role": "system", "content": "You are a math tutor. Keep explanations to 1-2 sentences."},
]

questions = [
    "What is 2 + 2?",
    "Now multiply that result by 3.",
    "Is that result a prime number?",
]

for i, question in enumerate(questions, 1):
    # Add the user's message to history
    history.append({"role": "user", "content": question})

    # Send the FULL history to the API
    response = client.chat.completions.create(
        model=MODEL, 
        messages=history
        )
    ai_reply = response.choices[0].message.content

    # Add the AI's reply to history (so the NEXT call includes it)
    history.append({"role": "assistant", "content": ai_reply})

    # Show what's happening
    print(f"\n  Turn {i}:")
    print(f"    User:      {question}")
    print(f"    Assistant: {ai_reply}")
    print(f"    Messages in array: {len(history)}  (1 system + {i} user + {i} assistant)")

print(f"\n  📊 Final messages array has {len(history)} messages.")
print(f"     Every turn adds 2 messages (user + assistant).")
print(f"     This is why long conversations use more tokens — and cost more.")


# ═══════════════════════════════════════════════════════════════════════════
# KEY TAKEAWAYS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)
print("""
  1. Every API call needs a messages array with role + content.
  2. The AI has NO memory — you must send the full history each time.
  3. The messages array grows by 2 each turn (user + assistant).
  4. More messages = more tokens = more cost.
  5. This is exactly how ChatGPT works behind the scenes —
     it sends your entire conversation with every message.
""")
