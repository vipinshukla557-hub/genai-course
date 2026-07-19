"""
=============================================================================
Applied GenAI Engineering Program
Week 4 · Section 5: Persona Switching + Sliding Window
=============================================================================

WHAT THIS FILE DOES:
  Adds two features to the chatbot:
  1. /persona — switch between different AI personalities mid-conversation
  2. Sliding window — automatically manage long conversations so you never
     exceed the context window

WHY THESE MATTER:
  Persona switching: Real products often have multiple "modes" — a customer
  support bot that can switch to a technical expert, or a writing assistant
  that can switch between formal and casual. The system prompt controls this.

  Sliding window: Every LLM has a context window limit (e.g., 8K, 32K, 128K
  tokens). If your conversation gets too long, the API will reject it.
  A sliding window keeps only the most recent messages, so you never crash.

CONCEPTS COVERED:
  1. Changing the system prompt at runtime
  2. Pre-defined personas with different behaviors
  3. Sliding window: keep last N turns + system prompt
  4. Why the system prompt should always stay (never get windowed out)

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
# PART 1: PERSONA DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════
# Each persona is just a different system prompt. That's all it takes
# to completely change how the AI behaves.

PERSONAS = {
    "default": {
        "name": "Default Assistant",
        "emoji": "🤖",
        "prompt": (
            "You are a helpful, friendly assistant. "
            "Keep your answers concise — 2 to 4 sentences unless asked for more. "
            "Be warm but professional."
        ),
    },
    "coder": {
        "name": "Code Expert",
        "emoji": "💻",
        "prompt": (
            "You are an expert Python programmer and AI engineer. "
            "When explaining code, always show a short example. "
            "Use technical terms but explain them simply. "
            "If the user's code has a bug, point it out kindly. "
            "Keep answers focused and practical."
        ),
    },
    "teacher": {
        "name": "Patient Teacher",
        "emoji": "📚",
        "prompt": (
            "You are a patient, encouraging teacher who explains concepts "
            "to absolute beginners. Use everyday analogies — compare technical "
            "concepts to things like cooking, shopping, or commuting. "
            "Never make the student feel dumb. "
            "Always check: 'Does that make sense?' at the end."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: SLIDING WINDOW — KEEP CONVERSATIONS MANAGEABLE
# ═══════════════════════════════════════════════════════════════════════════
# The idea:
#   - Always keep the system prompt (index 0)
#   - Keep only the last N turns of conversation
#   - A "turn" = 1 user message + 1 assistant message = 2 messages
#
# Example with max_turns=3:
#   history = [system, u1, a1, u2, a2, u3, a3, u4, a4, u5, a5]
#   After windowing: [system, u3, a3, u4, a4, u5, a5]
#   We dropped turns 1-2, kept turns 3-5

def apply_sliding_window(history: list[dict], max_turns: int = 10) -> list[dict]:
    """
    Keep only the system prompt + last max_turns turns of conversation.

    Args:
        history:   The full messages list (system + user/assistant pairs)
        max_turns: Maximum number of turns to keep (1 turn = user + assistant)

    Returns:
        A trimmed copy of history. Does NOT modify the original.
    """
    # System prompt is always the first message
    system_msg = history[0]

    # Everything after the system prompt is conversation
    conversation = history[1:]

    # Each turn is 2 messages (user + assistant), so keep last max_turns * 2
    max_messages = max_turns * 2

    if len(conversation) <= max_messages:
        # No trimming needed
        return history

    # Keep only the last max_messages
    trimmed = conversation[-max_messages:]

    # How many messages did we drop?
    dropped = len(conversation) - max_messages
    dropped_turns = dropped // 2

    print(f"\n  📎 Context window management: dropped {dropped_turns} oldest turns, "
          f"keeping last {max_turns}.")

    # Always start with system prompt
    return [system_msg] + trimmed


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: THE CHATBOT — WITH PERSONAS AND SLIDING WINDOW
# ═══════════════════════════════════════════════════════════════════════════

def run_chatbot():
    """Chatbot with persona switching and sliding window."""

    # ── Settings ───────────────────────────────────────────────────────
    MAX_TURNS = 10  # Keep last 10 turns in context

    # Start with the default persona
    current_persona = "default"
    persona = PERSONAS[current_persona]

    history = [{"role": "system", "content": persona["prompt"]}]

    # ── Welcome message ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {persona['emoji']} {persona['name']} — Ready to chat")
    print("  Commands:")
    print("    /persona         — list available personas")
    print("    /persona <name>  — switch persona (e.g., /persona coder)")
    print("    /history         — show conversation length")
    print("    /quit            — exit")
    print("=" * 60)

    while True:
        user_input = input(f"\n  You: ").strip()

        # ── Handle commands ────────────────────────────────────────────
        if user_input.lower() == "/quit":
            print("\n  👋 Goodbye!")
            break

        if not user_input:
            continue

        # ── /persona (no argument) — list personas ─────────────────────
        if user_input.lower() == "/persona":
            print("\n  Available personas:")
            for key, p in PERSONAS.items():
                marker = " ← current" if key == current_persona else ""
                print(f"    {p['emoji']} {key:10s} — {p['name']}{marker}")
            print(f"\n  Usage: /persona coder")
            continue

        # ── /persona <name> — switch persona ───────────────────────────
        if user_input.lower().startswith("/persona "):
            requested = user_input.split(" ", 1)[1].strip().lower()

            if requested not in PERSONAS:
                print(f"  ❌ Unknown persona: '{requested}'")
                print(f"     Available: {', '.join(PERSONAS.keys())}")
                continue

            current_persona = requested
            persona = PERSONAS[current_persona]

            # Replace the system prompt in history
            # The system prompt is ALWAYS history[0]
            history[0] = {"role": "system", "content": persona["prompt"]}

            print(f"\n  ✅ Switched to {persona['emoji']} {persona['name']}")
            print(f"     The AI will now behave differently — try it!")
            continue

        # ── /history — show conversation stats ─────────────────────────
        if user_input.lower() == "/history":
            total_msgs = len(history) - 1  # exclude system prompt
            turns = total_msgs // 2
            print(f"\n  📊 Conversation: {turns} turns ({total_msgs} messages)")
            print(f"     Window limit: {MAX_TURNS} turns")
            remaining = MAX_TURNS - turns
            if remaining > 0:
                print(f"     {remaining} turns before oldest messages are trimmed")
            else:
                print(f"     Window is full — oldest messages are being trimmed")
            continue

        # ── Regular message — send to AI ───────────────────────────────
        history.append({"role": "user", "content": user_input})

        # Apply sliding window BEFORE sending to API
        messages_to_send = apply_sliding_window(history, max_turns=MAX_TURNS)

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages_to_send,
            temperature=0.7,
        )

        ai_reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": ai_reply})

        print(f"\n  {persona['emoji']}: {ai_reply}")


# ═══════════════════════════════════════════════════════════════════════════
# RUN IT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_chatbot()
