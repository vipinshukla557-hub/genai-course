"""
=============================================================================
Applied GenAI Engineering Program
Week 4 · Section 4: Error Handling + Cost Tracking
=============================================================================

WHAT THIS FILE DOES:
  Adds two production-essential features to the chatbot:
  1. Proper error handling — so the chatbot doesn't crash on API errors
  2. Cost tracking — so you always know how much you've spent

WHY THIS MATTERS:
  A chatbot that crashes when the API has a hiccup is not a real product.
  A chatbot that burns through your API budget with no tracking is a
  financial risk. Both of these are table stakes for any production app.

ERROR TYPES YOU'LL SEE IN THE WILD:
  - AuthenticationError  → Wrong API key. NOT retryable. Fix the key.
  - RateLimitError       → Too many requests. Retryable. Wait and retry.
  - APITimeoutError      → Server took too long. Retryable. Try again.
  - APIConnectionError   → Network issue. Retryable. Check your internet.
  - BadRequestError      → Malformed request (e.g., too many tokens). Fix input.

COST TRACKING:
  GPT-4o-mini is very cheap but not free. Tracking cost from day one
  is an engineering habit — when your app runs 10,000 conversations
  per day, this is how you know what it costs.

PREREQUISITES:
  - .env file with OPENAI_API_KEY=your_key_here
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

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",   # ← uncomment for Groq
    api_key=os.getenv("GROQ_API_KEY"),             # ← change to GROQ_API_KEY for Groq
)
# MODEL = "gpt-4o-mini"
MODEL = "llama-3.3-70b-versatile"                 # ← Groq model


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: THE SAFE API CALL — NEVER LET YOUR APP CRASH
# ═══════════════════════════════════════════════════════════════════════════
# The idea: wrap the API call in try/except. Handle each error type
# differently. Retry the ones that are temporary, stop on the ones
# that mean something is broken.

def safe_api_call(messages: list[dict], max_retries: int = 3) -> dict | None:
    """
    Makes an API call with error handling and retry logic.

    Returns a dict with:
      - "content": the AI's response text
      - "prompt_tokens": tokens used for the input
      - "completion_tokens": tokens used for the output
    Or None if the call failed completely.
    """

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
            )

            return {
                "content": response.choices[0].message.content,
                "prompt_tokens": response.usage.prompt_tokens, #input tokens
                "completion_tokens": response.usage.completion_tokens,  #output tokens
            }

        # ── NOT retryable — stop immediately ───────────────────────────
        except AuthenticationError:
            print("\n  ❌ Authentication failed. Check your API key in .env")
            return None

        except BadRequestError as e:
            print(f"\n  ❌ Bad request: {e}")
            print("     This usually means your input is too long or malformed.")
            return None

        # ── Retryable — wait and try again ─────────────────────────────
        except RateLimitError:
            wait_time = 2 ** attempt  # 2s, 4s, 8s — exponential backoff
            print(f"\n  ⏳ Rate limited. Waiting {wait_time}s before retry "
                  f"({attempt}/{max_retries})...")
            time.sleep(wait_time)

        except APITimeoutError:
            print(f"\n  ⏳ Timeout. Retrying ({attempt}/{max_retries})...")
            time.sleep(1)

        except APIConnectionError:
            print(f"\n  ⏳ Connection error. Retrying ({attempt}/{max_retries})...")
            time.sleep(2)

        # ── Catch-all for anything unexpected ──────────────────────────
        except Exception as e:
            print(f"\n  ❌ Unexpected error: {type(e).__name__}: {e}")
            return None

    # If we get here, all retries failed
    print("\n  ❌ All retries failed. Please try again later.")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: COST TRACKER — KNOW WHAT YOU'RE SPENDING
# ═══════════════════════════════════════════════════════════════════════════
# We keep a running total of tokens used across the entire session.
# When the user types /cost, we show the summary.
#
# Pricing:
#   GPT-4o-mini: $0.150 per 1M input tokens, $0.600 per 1M output tokens

class CostTracker:
    """Tracks token usage and estimated cost across a chat session."""

    # Groq Llama-3.3-70B-Versatile pricing per token
    INPUT_COST_PER_TOKEN = 0.00 / 1_000_000    # free tier
    OUTPUT_COST_PER_TOKEN = 0.00 / 1_000_000   # free tier

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    def add(self, prompt_tokens: int, completion_tokens: int):
        """Record tokens from one API call."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_calls += 1

    def get_summary(self) -> str:
        """Return a formatted cost summary."""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens

        # Calculate cost (as if using GPT-4o-mini)
        input_cost = self.total_prompt_tokens * self.INPUT_COST_PER_TOKEN
        output_cost = self.total_completion_tokens * self.OUTPUT_COST_PER_TOKEN
        total_cost = input_cost + output_cost

        # Convert to INR (approximate)
        cost_inr = total_cost * 96  # rough USD to INR

        return (
            f"\n  ╔══════════════════════════════════════╗\n"
            f"  ║        💰 Session Cost Summary       ║\n"
            f"  ╠══════════════════════════════════════╣\n"
            f"  ║  API calls made:    {self.total_calls:>14}   ║\n"
            f"  ║  Input tokens:      {self.total_prompt_tokens:>14,}   ║\n"
            f"  ║  Output tokens:     {self.total_completion_tokens:>14,}   ║\n"
            f"  ║  Total tokens:      {total_tokens:>14,}   ║\n"
            f"  ╠═════════════════════════════════════╣\n"
            f"  ║  Est. cost (Groq (Llama 3.3-70b)  ║\n"
            f"  ║  USD: ${total_cost:>10.6f}                 ║\n"
            f"  ║  INR: ₹{cost_inr:>10.6f}                  ║\n"
            f"  ╚══════════════════════════════════════╝"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: CHATBOT WITH ERROR HANDLING + COST TRACKING
# ═══════════════════════════════════════════════════════════════════════════

def run_chatbot():
    """Chatbot with error handling and /cost command."""

    system_prompt = (
        "You are a helpful, friendly assistant. "
        "Keep answers concise — 2 to 4 sentences unless asked for more."
    )

    history = [{"role": "system", "content": system_prompt}]
    tracker = CostTracker()

    print("\n" + "=" * 60)
    print("  🤖 Chatbot with Error Handling + Cost Tracking")
    print("  Commands:")
    print("    /cost  — show token usage and estimated cost")
    print("    /quit  — exit the chatbot")
    print("=" * 60)

    while True:
        user_input = input("\n  You: ").strip()

        # ── Handle commands ────────────────────────────────────────────
        if user_input.lower() == "/quit":
            print(tracker.get_summary())
            print("\n  👋 Goodbye!")
            break

        if user_input.lower() == "/cost":
            print(tracker.get_summary())
            continue

        if not user_input:
            continue

        # ── Make the API call (with error handling) ────────────────────
        history.append({"role": "user", "content": user_input})

        result = safe_api_call(history)

        if result is None:
            # API call failed — remove the user message from history
            # so we don't have a dangling user message with no response
            history.pop()
            print("  (Message not sent — try again)")
            continue

        # ── Success! Save response and track cost ──────────────────────
        history.append({"role": "assistant", "content": result["content"]})
        tracker.add(result["prompt_tokens"], result["completion_tokens"])

        print(f"\n  🤖: {result['content']}")

        turns = (len(history) - 1) // 2
        total = result["prompt_tokens"] + result["completion_tokens"]
        print(f"  [Turn {turns} · {total} tokens this turn]")


# ═══════════════════════════════════════════════════════════════════════════
# RUN IT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_chatbot()
