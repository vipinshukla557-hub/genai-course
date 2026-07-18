"""
=============================================================================
Applied GenAI Engineering Program
Week 4 · Section 2: Your First Chat Loop — A Real Chatbot
=============================================================================

WHAT THIS FILE DOES:
  Builds a working CLI chatbot that maintains conversation history.
  You type, the AI responds, and it remembers everything you've said.
  This is the skeleton every chatbot in the world is built on.

WHY THIS MATTERS:
  In Section 1, we made individual API calls and manually built the
  messages array. Now we automate that into a loop. Type a message,
  get a response, type another — just like ChatGPT, but in your terminal.

CONCEPTS COVERED:
  1. The infinite input loop (while True + input())
  2. Automatically building the messages array each turn
  3. Graceful exit with /quit
  4. Clean terminal output

PREREQUISITES:
  - .env file with OPENAI_API_KEY=your_key_here
  - pip install openai python-dotenv
=============================================================================
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
     base_url="https://api.groq.com/openai/v1",   # ← uncomment for Groq
    api_key=os.getenv("GROQ_API_KEY"),             # ← change to GROQ_API_KEY for Groq
)
# MODEL = "gpt-4o-mini"
MODEL = "llama-3.3-70b-versatile"                 # ← Groq model


# ═══════════════════════════════════════════════════════════════════════════
# THE CHAT LOOP — THE HEART OF EVERY CHATBOT
# ═══════════════════════════════════════════════════════════════════════════
#
# Here's the pattern:
#   1. Start with a system message
#   2. Loop forever:
#      a. Get input from the user
#      b. Add it to history
#      c. Send history to the API
#      d. Get the AI's response
#      e. Add the response to history
#      f. Print the response
#   3. Exit when the user types /quit
#
# That's it. Every chatbot — ChatGPT, Claude, Gemini — follows this
# exact pattern. The only difference is the UI wrapping around it.

def run_chatbot():
    """A simple chatbot that maintains conversation history."""

    # ── System prompt — sets the AI's behavior ─────────────────────────
    system_prompt = (
        "You are a helpful, friendly assistant. "
        "Keep your answers concise — 2 to 4 sentences unless the user "
        "asks for more detail. Be warm but professional."
    )

    # ── Conversation history — starts with just the system message ─────
    history = [
        {"role": "system", "content": system_prompt},
    ]

    # ── Welcome message ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  🤖 Simple Chatbot")
    print("  Type your message and press Enter.")
    print("  Type /quit to exit.")
    print("=" * 60)

    # ── The loop ───────────────────────────────────────────────────────
    while True:
        # Step 1: Get user input
        user_input = input("\n  You: ").strip()

        # Step 2: Handle exit command
        if user_input.lower() == "/quit":
            print("\n  👋 Goodbye! Chat ended.")
            break

        # Step 3: Handle empty input (user just pressed Enter)
        if not user_input:
            print("  (Empty message — type something or /quit to exit)")
            continue

        # Step 4: Add the user's message to history
        history.append({"role": "user", "content": user_input})

        # Step 5: Send the FULL history to the API
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            temperature=0.7,
        )

        # Step 6: Extract the AI's reply
        ai_reply = response.choices[0].message.content

        # Step 7: Add the AI's reply to history (for next turn)
        history.append({"role": "assistant", "content": ai_reply})

        # Step 8: Print the response
        print(f"\n  🤖: {ai_reply}")

        # Bonus: show how many messages are in the history
        turns = (len(history) - 1) // 2  # subtract system, divide by 2
        print(f"  [Turn {turns} · {len(history)} messages in history]")


# ═══════════════════════════════════════════════════════════════════════════
# RUN IT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_chatbot()
