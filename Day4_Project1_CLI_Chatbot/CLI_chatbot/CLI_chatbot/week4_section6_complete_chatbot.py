"""
=============================================================================
Applied GenAI Engineering Program
Week 4 · Section 6: The Complete Smart CLI Chatbot — PROJECT 1
=============================================================================

🏆  THIS IS YOUR FIRST PORTFOLIO PROJECT  🏆

WHAT THIS FILE IS:
  The final, assembled chatbot that combines EVERYTHING from today:
    ✅ Conversation history (Section 1-2)
    ✅ Streaming responses (Section 3)
    ✅ Error handling with retries (Section 4)
    ✅ Cost tracking with /cost command (Section 4)
    ✅ Persona switching with /persona command (Section 5)
    ✅ Sliding window for long conversations (Section 5)
    ✅ Provider swap: Groq ↔ OpenAI in one line (NEW)

  This is the file you push to GitHub. This is what you show in interviews.

HOW TO RUN:
  1. Make sure your .env has OPENAI_API_KEY=your_key_here
  2. python week4_section6_complete_chatbot.py
  3. Chat, switch personas, check cost, and enjoy your first AI project!

COMMANDS:
  /persona          — list available personas
  /persona <name>   — switch to a different persona
  /cost             — show token usage and estimated cost
  /history          — show conversation length
  /provider         — show current API provider
  /clear            — clear conversation history (keep persona)
  /help             — show all commands
  /quit             — exit (shows final cost summary)

PREREQUISITES:
  - .env file with OPENAI_API_KEY=your_key_here
  - Optionally: GROQ_API_KEY=your_key_here (to test provider swap — free!)
  - pip install openai python-dotenv
=============================================================================
"""

import os
import time
from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    BadRequestError,
)

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════
# PROVIDER CONFIGURATION — THE ONE-LINE SWAP
# ═══════════════════════════════════════════════════════════════════════════
# This is the key insight: OpenAI, Groq, Together, Fireworks — they all
# use the same SDK. You change THREE things: base_url, api_key, model.
# Everything else — messages format, streaming, error handling — stays
# EXACTLY the same. This is why the OpenAI SDK is the industry standard.

PROVIDERS = {
    "openai": {
        "name": "OpenAI (GPT-4o-mini)",
        "base_url": None,  # Default — no base_url needed for OpenAI
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4o-mini",
        "input_cost_per_million": 0.150,
        "output_cost_per_million": 0.600,
    },
    "groq": {
        "name": "Groq (Llama 3.3-70b)",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.getenv("GROQ_API_KEY"),
        "model": "llama-3.3-70b-versatile",
        "input_cost_per_million": 0.00,     # Free tier
        "output_cost_per_million": 0.00,    # Free tier
    },
}

# ── Start with OpenAI ──────────────────────────────────────────────────
ACTIVE_PROVIDER = "groq"


def get_client(provider_key: str) -> OpenAI:
    """Create an OpenAI-compatible client for the given provider."""
    provider = PROVIDERS[provider_key]

    kwargs = {"api_key": provider["api_key"]}
    if provider["base_url"]:
        kwargs["base_url"] = provider["base_url"]

    return OpenAI(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# PERSONAS
# ═══════════════════════════════════════════════════════════════════════════

PERSONAS = {
    "default": {
        "name": "Default Assistant",
        "emoji": "🤖",
        "prompt": (
            "You are a helpful, friendly assistant. "
            "Keep your answers concise — 2 to 4 sentences unless the user "
            "asks for more detail. Be warm but professional."
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
            "Keep answers focused and practical — no fluff."
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
# COST TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class CostTracker:
    """Tracks token usage and estimated cost across a chat session."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    def add(self, prompt_tokens: int, completion_tokens: int):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_calls += 1

    def get_summary(self, provider_key: str) -> str:
        provider = PROVIDERS[provider_key]
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens

        input_cost = self.total_prompt_tokens * provider["input_cost_per_million"] / 1_000_000
        output_cost = self.total_completion_tokens * provider["output_cost_per_million"] / 1_000_000
        total_cost = input_cost + output_cost
        cost_inr = total_cost * 85

        lines = [
            f"\n  ╔══════════════════════════════════════════╗",
            f"  ║          💰 Session Cost Summary           ║",
            f"  ╠══════════════════════════════════════════╣",
            f"  ║  Provider:        {provider['name']:>22s}  ║",
            f"  ║  API calls:       {self.total_calls:>22}  ║",
            f"  ║  Input tokens:    {self.total_prompt_tokens:>22,}  ║",
            f"  ║  Output tokens:   {self.total_completion_tokens:>22,}  ║",
            f"  ║  Total tokens:    {total_tokens:>22,}  ║",
            f"  ╠══════════════════════════════════════════╣",
            f"  ║  Est. cost:  ${total_cost:>10.6f} / ₹{cost_inr:.4f}  ║",
            f"  ╚══════════════════════════════════════════╝",
        ]

        if provider["input_cost_per_million"] == 0:
            lines.append(f"  (Using free tier — actual cost = $0.00)")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SAFE API CALL — WITH RETRY AND STREAMING
# ═══════════════════════════════════════════════════════════════════════════

def safe_streaming_call(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_retries: int = 3,
) -> dict | None:
    """
    Makes a streaming API call with error handling and retries.

    Returns dict with "content", "prompt_tokens", "completion_tokens"
    or None on failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                stream=True,
                stream_options={"include_usage": True},
                # ↑ This tells the API to include token counts
                #   in the final chunk of the stream.
                #   Without this, we can't track cost for streaming calls.
            )

            full_response = ""
            usage_data = None

            for chunk in stream:
                # Check for content
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content

                # Check for usage (comes in the final chunk)
                if chunk.usage is not None:
                    usage_data = chunk.usage

            print()  # New line after streaming completes

            # Extract token counts
            prompt_tokens = 0
            completion_tokens = 0
            if usage_data:
                prompt_tokens = usage_data.prompt_tokens
                completion_tokens = usage_data.completion_tokens

            return {
                "content": full_response,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

        except AuthenticationError:
            print("\n  ❌ Authentication failed. Check your API key.")
            return None

        except BadRequestError as e:
            print(f"\n  ❌ Bad request: {e}")
            return None

        except RateLimitError:
            wait = 2 ** attempt
            print(f"\n  ⏳ Rate limited. Waiting {wait}s ({attempt}/{max_retries})...")
            time.sleep(wait)

        except (APITimeoutError, APIConnectionError):
            print(f"\n  ⏳ Connection issue. Retrying ({attempt}/{max_retries})...")
            time.sleep(2)

        except Exception as e:
            print(f"\n  ❌ Unexpected error: {type(e).__name__}: {e}")
            return None

    print("\n  ❌ All retries failed.")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SLIDING WINDOW
# ═══════════════════════════════════════════════════════════════════════════

def apply_sliding_window(history: list[dict], max_turns: int = 10) -> list[dict]:
    """Keep system prompt + last max_turns turns."""
    system_msg = history[0]
    conversation = history[1:]
    max_messages = max_turns * 2

    if len(conversation) <= max_messages:
        return history[:]

    trimmed = conversation[-max_messages:]
    dropped = (len(conversation) - max_messages) // 2
    print(f"  📎 Trimmed {dropped} oldest turns (keeping last {max_turns})")

    return [system_msg] + trimmed


# ═══════════════════════════════════════════════════════════════════════════
# HELP TEXT
# ═══════════════════════════════════════════════════════════════════════════

HELP_TEXT = """
  ╔══════════════════════════════════════════╗
  ║            📋 Available Commands          ║
  ╠══════════════════════════════════════════╣
  ║  /persona          List personas          ║
  ║  /persona <name>   Switch persona         ║
  ║  /cost             Show cost summary       ║
  ║  /history          Show conversation info  ║
  ║  /provider         Show current provider   ║
  ║  /clear            Clear chat history      ║
  ║  /help             Show this menu          ║
  ║  /quit             Exit (shows cost)       ║
  ╚══════════════════════════════════════════╝"""


# ═══════════════════════════════════════════════════════════════════════════
# THE MAIN CHATBOT
# ═══════════════════════════════════════════════════════════════════════════

def run_chatbot():
    """The complete Smart CLI Chatbot — Project 1."""

    # ── Setup ──────────────────────────────────────────────────────────
    provider_key = ACTIVE_PROVIDER
    provider = PROVIDERS[provider_key]
    ai_client = get_client(provider_key)

    current_persona = "default"
    persona = PERSONAS[current_persona]

    history = [{"role": "system", "content": persona["prompt"]}]
    tracker = CostTracker()
    max_turns = 10

    # ── Welcome ────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  {persona['emoji']} Smart CLI Chatbot — Project 1")
    print(f"  Provider: {provider['name']}")
    print(f"  Type /help for commands")
    print("═" * 60)

    # ── Chat loop ──────────────────────────────────────────────────────
    while True:
        user_input = input(f"\n  You: ").strip()

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── /quit ──────────────────────────────────────────────────────
        if cmd == "/quit":
            print(tracker.get_summary(provider_key))
            print("\n  👋 Thanks for chatting! Push this to GitHub 🚀")
            break

        # ── /help ──────────────────────────────────────────────────────
        if cmd == "/help":
            print(HELP_TEXT)
            continue

        # ── /cost ──────────────────────────────────────────────────────
        if cmd == "/cost":
            print(tracker.get_summary(provider_key))
            continue

        # ── /history ───────────────────────────────────────────────────
        if cmd == "/history":
            total_msgs = len(history) - 1
            turns = total_msgs // 2
            print(f"\n  📊 {turns} turns in history ({total_msgs} messages)")
            print(f"     Window: keeping last {max_turns} turns")
            continue

        # ── /provider ──────────────────────────────────────────────────
        if cmd == "/provider":
            print(f"\n  🔌 Current: {provider['name']}")
            print(f"     Model:   {provider['model']}")
            for key, p in PROVIDERS.items():
                status = "✅ active" if key == provider_key else (
                    "✅ key found" if p["api_key"] else "❌ no key in .env"
                )
                print(f"     {key:8s} — {status}")
            continue

        # ── /clear ─────────────────────────────────────────────────────
        if cmd == "/clear":
            history = [{"role": "system", "content": persona["prompt"]}]
            print(f"\n  🗑️  History cleared. Persona ({persona['name']}) retained.")
            continue

        # ── /persona (list) ────────────────────────────────────────────
        if cmd == "/persona":
            print("\n  Available personas:")
            for key, p in PERSONAS.items():
                marker = " ← current" if key == current_persona else ""
                print(f"    {p['emoji']} {key:10s} — {p['name']}{marker}")
            print(f"\n  Usage: /persona coder")
            continue

        # ── /persona <name> (switch) ───────────────────────────────────
        if cmd.startswith("/persona "):
            requested = user_input.split(" ", 1)[1].strip().lower()
            if requested not in PERSONAS:
                print(f"  ❌ Unknown: '{requested}'. Options: {', '.join(PERSONAS.keys())}")
                continue

            current_persona = requested
            persona = PERSONAS[current_persona]
            history[0] = {"role": "system", "content": persona["prompt"]}
            print(f"\n  ✅ Now talking to {persona['emoji']} {persona['name']}")
            continue

        # ── Unknown command ────────────────────────────────────────────
        if cmd.startswith("/"):
            print(f"  ❓ Unknown command: {user_input}. Type /help for options.")
            continue

        # ── Regular message ────────────────────────────────────────────
        history.append({"role": "user", "content": user_input})

        # Apply sliding window
        messages_to_send = apply_sliding_window(history, max_turns=max_turns)

        # Stream the response
        print(f"\n  {persona['emoji']}: ", end="", flush=True)

        result = safe_streaming_call(
            client=ai_client,
            model=provider["model"],
            messages=messages_to_send,
        )

        if result is None:
            history.pop()
            print("  (Failed — try again)")
            continue

        # Save response and track cost
        history.append({"role": "assistant", "content": result["content"]})
        tracker.add(result["prompt_tokens"], result["completion_tokens"])


# ═══════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_chatbot()
