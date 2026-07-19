# ═══════════════════════════════════════════════════════════════
# WEEK 2 — SECTION FILE (Self-Study Reference)
# Tokenization, Cost Estimation, Temperature, Hallucination,
# Context Window, Model Comparison
# ═══════════════════════════════════════════════════════════════
#
# This file has more examples than we covered live in class.
# Run any section you want to explore further.
# ═══════════════════════════════════════════════════════════════

import os
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
import tiktoken

load_dotenv(find_dotenv())
client = OpenAI()

# ┌─────────────────────────────────────────────────────────────┐
# │ GROQ ALTERNATIVE (free):                                    │
# │ client = OpenAI(                                            │
# │     base_url="https://api.groq.com/openai/v1",             │
# │     api_key=os.getenv("GROQ_API_KEY"),                     │
# │ )                                                           │
# │ Change model="gpt-4o-mini" to "llama-3.3-70b-versatile"    │
# └─────────────────────────────────────────────────────────────┘

enc = tiktoken.encoding_for_model("gpt-4o-mini")


# ═══════════════════════════════════════════════════════════════
# SECTION A: TOKENIZATION — More Examples
# ═══════════════════════════════════════════════════════════════

print("=" * 55)
print("TOKENIZATION EXAMPLES")
print("=" * 55)

# Simple words
print("\n--- Simple words ---")
print("'Hello'             ->", len(enc.encode("Hello")), "tokens")
print("'AI'                ->", len(enc.encode("AI")), "tokens")
print("'Artificial'        ->", len(enc.encode("Artificial")), "tokens")
print("'supercalifragilistic' ->", len(enc.encode("supercalifragilistic")), "tokens")

# Indian food (fun to compare!)
print("\n--- Indian food ---")
print("'biryani'           ->", len(enc.encode("biryani")), "tokens")
print("'paneer'            ->", len(enc.encode("paneer")), "tokens")
print("'naan'              ->", len(enc.encode("naan")), "tokens")
print("'chicken tikka'     ->", len(enc.encode("chicken tikka")), "tokens")
print("'dal makhani'       ->", len(enc.encode("dal makhani")), "tokens")

# Language comparison
print("\n--- Same meaning, different languages ---")
print("'Hello'             ->", len(enc.encode("Hello")), "tokens (English)")
print("'नमस्ते'              ->", len(enc.encode("नमस्ते")), "tokens (Hindi)")
print("'こんにちは'          ->", len(enc.encode("こんにちは")), "tokens (Japanese)")
print("'مرحبا'              ->", len(enc.encode("مرحبا")), "tokens (Arabic)")
print("'Bonjour'           ->", len(enc.encode("Bonjour")), "tokens (French)")

# Code and special characters
print("\n--- Code and special chars ---")
print("'print(\"hello\")'    ->", len(enc.encode('print("hello")')), "tokens")
print("'OPENAI_API_KEY'    ->", len(enc.encode("OPENAI_API_KEY")), "tokens")
print("'🚀🤖💡'            ->", len(enc.encode("🚀🤖💡")), "tokens")
print("'sk-abc123def456'   ->", len(enc.encode("sk-abc123def456")), "tokens")

# IPL team names
print("\n--- IPL teams (fun test!) ---")
print("'Mumbai Indians'    ->", len(enc.encode("Mumbai Indians")), "tokens")
print("'Chennai Super Kings' ->", len(enc.encode("Chennai Super Kings")), "tokens")
print("'Royal Challengers Bengaluru' ->", len(enc.encode("Royal Challengers Bengaluru")), "tokens")

# Token-to-word ratio
print("\n--- Rule of thumb ---")
text = "The quick brown fox jumps over the lazy dog near the river"
print(f"'{text}'")
print(f"  Words: {len(text.split())}  Tokens: {len(enc.encode(text))}")
print(f"  Ratio: {len(enc.encode(text))/len(text.split()):.2f} tokens per word")
print("  General rule: 100 English words = ~130 tokens")


# ═══════════════════════════════════════════════════════════════
# SECTION B: COST ESTIMATOR
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("COST ESTIMATION")
print("=" * 55)

# GPT-4o-mini pricing (2025-2026):
#   Input:  $0.15 per 1 million tokens
#   Output: $0.60 per 1 million tokens

prompt = "Explain quantum computing in simple terms."
input_tokens = len(enc.encode(prompt))
output_tokens = 500  # estimated response length

input_cost = (input_tokens / 1_000_000) * 0.15
output_cost = (output_tokens / 1_000_000) * 0.60
total_usd = input_cost + output_cost
total_inr = total_usd * 85

print(f"\nPrompt: '{prompt}'")
print(f"  Input tokens:  {input_tokens}")
print(f"  Output tokens: {output_tokens} (estimated)")
print(f"  Total cost:    ${total_usd:.6f} (₹{total_inr:.4f})")

# Scale it up
print(f"\nAt 10,000 calls per day with similar prompts:")
print(f"  Daily:  ${total_usd * 10000:.2f}")
print(f"  Monthly: ${total_usd * 10000 * 30:.2f}")

# Compare models
print("\nSame prompt on GPT-4o (the expensive one):")
input_cost_4o = (input_tokens / 1_000_000) * 2.50
output_cost_4o = (output_tokens / 1_000_000) * 10.00
total_4o = input_cost_4o + output_cost_4o
print(f"  Total cost: ${total_4o:.6f} (₹{total_4o * 85:.4f})")
print(f"  That is {total_4o / total_usd:.0f}x more expensive!")


# ═══════════════════════════════════════════════════════════════
# SECTION C: TEMPERATURE — More Experiments
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("TEMPERATURE: CONSISTENCY TEST")
print("=" * 55)

prompt = "Name one famous dish from Hyderabad."

# Run the same prompt 3 times at temp 0
# You should get the EXACT same answer every time
print("\nTemp 0 (3 runs, should be identical):")
print("  Run 1:", client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0, max_tokens=50,
).choices[0].message.content.strip())

print("  Run 2:", client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0, max_tokens=50,
).choices[0].message.content.strip())

print("  Run 3:", client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0, max_tokens=50,
).choices[0].message.content.strip())

# Now try temp 1.5 — each run should be different
print("\nTemp 1.5 (3 runs, should vary):")
print("  Run 1:", client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=1.5, max_tokens=50,
).choices[0].message.content.strip())

print("  Run 2:", client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=1.5, max_tokens=50,
).choices[0].message.content.strip())

print("  Run 3:", client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=1.5, max_tokens=50,
).choices[0].message.content.strip())


# ── Practical: Email classification with temp 0 ──

print("\n" + "=" * 55)
print("PRACTICAL: Email Classification (temp=0)")
print("=" * 55)

email_1 = "Hi, your order #12345 has shipped and will arrive Tuesday."
email_2 = "URGENT: Your account will be suspended unless you verify NOW!!!"
email_3 = "Hey! Want to grab lunch tomorrow? Found a great new place."

system = "Classify this email as: work, personal, or spam. Reply with ONE word only."

r1 = client.chat.completions.create(model="gpt-4o-mini",
    messages=[{"role": "system", "content": system},
              {"role": "user", "content": email_1}],
    temperature=0, max_tokens=10)

r2 = client.chat.completions.create(model="gpt-4o-mini",
    messages=[{"role": "system", "content": system},
              {"role": "user", "content": email_2}],
    temperature=0, max_tokens=10)

r3 = client.chat.completions.create(model="gpt-4o-mini",
    messages=[{"role": "system", "content": system},
              {"role": "user", "content": email_3}],
    temperature=0, max_tokens=10)

print(f"  '{email_1[:45]}...' -> {r1.choices[0].message.content.strip()}")
print(f"  '{email_2[:45]}...' -> {r2.choices[0].message.content.strip()}")
print(f"  '{email_3[:45]}...' -> {r3.choices[0].message.content.strip()}")


# ═══════════════════════════════════════════════════════════════
# SECTION D: HALLUCINATION — More Tests
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("HALLUCINATION: FAKE RESEARCH PAPER")
print("=" * 55)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content":
        "Summarize the paper 'Neural Pathways in Cloud Formation' "
        "by Smith et al., published in Nature, 2023."
    }],
)
print(response.choices[0].message.content)
print("\n⚠️  This paper does NOT exist. Every detail was fabricated.")

print("\n" + "=" * 55)
print("HALLUCINATION: FAKE PERSON")
print("=" * 55)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content":
        "Who is Dr. Rajesh Krishnamurthy, the Nobel Prize winner "
        "in quantum biology?"
    }],
)
print(response.choices[0].message.content)
print("\n⚠️  This person does not exist. 'Quantum biology' has no Nobel Prize.")


# ═══════════════════════════════════════════════════════════════
# SECTION E: CONTEXT WINDOW
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("CONTEXT WINDOW: How Fast Does It Fill Up?")
print("=" * 55)

# GPT-4o-mini has a 128,000 token context window.
# That sounds huge. But let us see how fast it fills
# in a chatbot scenario.

system_prompt = "You are a helpful coding assistant."
user_msg = "How do I read a CSV file in Python?"
assistant_reply = ("You can use pandas: import pandas as pd, "
                   "then df = pd.read_csv('file.csv'). "
                   "This gives you a DataFrame you can filter, sort, etc.")

sys_tokens = len(enc.encode(system_prompt))
user_tokens = len(enc.encode(user_msg))
reply_tokens = len(enc.encode(assistant_reply))
one_exchange = sys_tokens + user_tokens + reply_tokens

print(f"  System prompt:   {sys_tokens} tokens")
print(f"  One user message: {user_tokens} tokens")
print(f"  One AI reply:    {reply_tokens} tokens")
print(f"  One exchange:    {one_exchange} tokens total")
print(f"\n  Context window:  128,000 tokens")
print(f"  Exchanges before full: ~{128_000 // one_exchange}")
print(f"\n  Sounds like a lot, but add long code snippets or")
print(f"  documents and it shrinks fast!")

# What happens when you exceed the limit?
print("\n  If you exceed the limit, OpenAI returns an error:")
print("  'This model's maximum context length is 128000 tokens.'")
print("\n  Solutions (covered later in the program):")
print("  1. Sliding window: keep only the last N messages")
print("  2. Summarize: compress old messages into a summary")
print("  3. RAG: retrieve only what is needed (Month 2)")


# ═══════════════════════════════════════════════════════════════
# SECTION F: MODEL COMPARISON (OpenAI vs Groq)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("MODEL COMPARISON: GPT-4o-mini vs Groq Llama 3")
print("=" * 55)

# To run this section, you need a Groq API key.
# Get one free at: console.groq.com/keys
# Add to your .env: GROQ_API_KEY=your-key-here

groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    print("  Skipping: GROQ_API_KEY not found in .env")
    print("  Get a free key at console.groq.com/keys")
else:
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key,
    )

    comparison_prompt = "Explain what an API is to a 10-year-old in 2 sentences."

    gpt_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": comparison_prompt}],
    )

    groq_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": comparison_prompt}],
    )

    print(f"\n  GPT-4o-mini says:")
    print(f"  {gpt_response.choices[0].message.content}")
    print(f"\n  Groq Llama 3 says:")
    print(f"  {groq_response.choices[0].message.content}")
    print(f"\n  Same question, different answers.")
    print(f"  Neither is 'right'. Different models, different patterns.")
    print(f"\n  To switch: change base_url, api_key, and model name.")
    print(f"  Everything else (messages format, response format) is identical.")
    print(f"  This is called an 'OpenAI-compatible API'.")
