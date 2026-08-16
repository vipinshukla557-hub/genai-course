"""
Week 6 Homework: Memory + Conversational AI with LangChain
===========================================================
7 exercises covering everything from the session + more.

Exercise 1: Interactive Chatbot with input() Loop
Exercise 2: Window Memory (Keep Last N Turns)
Exercise 3: Summary Memory (Summarize Old Messages)
Exercise 4: trim_messages with Token Counting
Exercise 5: Multi-Conversation Persistence (Multiple Users)
Exercise 6: Support Chatbot with Persona and Memory
Exercise 7: Compare Full History vs Window vs Summary
"""

# --- Setup (same as every week) ---
import os
import json
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import trim_messages

# --- If you want to use Groq instead, comment out the line below
# --- and uncomment these 2 lines:
from langchain_groq import ChatGroq
model = ChatGroq(model="llama-3.3-70b-versatile")

# model = ChatOpenAI(model="gpt-4o-mini")


# ============================================================
# EXERCISE 1: Interactive Chatbot with input() Loop
# ============================================================
# In the live demo, we hardcoded user messages. In a real chatbot,
# the user types messages in real time using input().
#
# This exercise builds a proper interactive chatbot:
# - while True loop that keeps asking for user input
# - Appends each message to history
# - Type "quit" to exit
#
# Scenario: A Zomato food recommendation chatbot

def exercise_1():
    print("=" * 60)
    print("Exercise 1: Interactive Zomato Food Recommendation Bot")
    print("Type 'quit' to exit")
    print("=" * 60)

    # --- Prompt template with system persona and history slot ---
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Zomato food recommendation assistant for Pune city. "
         "You help users find restaurants and dishes based on their preferences. "
         "Be friendly, use casual language, and suggest specific restaurant names "
         "and dishes with approximate prices in Rs. Keep responses under 3 sentences."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])

    chain = prompt | model

    # --- Empty history list: the chatbot's memory ---
    chat_history = []

    # --- The chat loop ---
    while True:
        # input() pauses the program and waits for the user to type something
        user_input = input("\nYou: ")

        # --- Check if user wants to quit ---
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        # --- Send the message to the chain ---
        response = chain.invoke({
            "chat_history": chat_history,
            "user_input": user_input
        })

        # --- Print the bot's response ---
        print("Bot:", response.content)

        # --- Save both messages to history ---
        # This is how the bot remembers the conversation.
        # Without these two lines, every turn would be independent.
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(response)

        # --- Show history size (helpful for debugging) ---
        print(f"  [History: {len(chat_history)} messages]")


# --- Uncomment the line below to run Exercise 1 ---
exercise_1()


# ============================================================
# EXERCISE 2: Window Memory (Keep Last N Turns)
# ============================================================
# Problem: If the chatbot runs for 100 turns, history has 200 messages.
# That's too many tokens. The API call gets expensive or fails.
#
# Solution: Keep only the last N turns. Older messages are discarded.
# This is called "window memory" because you only see a window
# of the most recent conversation.
#
# Scenario: An IPL cricket discussion chatbot

def exercise_2():
    print("=" * 60)
    print("Exercise 2: IPL Cricket Bot with Window Memory (Last 3 Turns)")
    print("Type 'quit' to exit, 'history' to see current memory")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an IPL cricket expert. You discuss matches, players, "
         "and statistics. Keep responses concise, under 2 sentences."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])

    chain = prompt | model
    chat_history = []

    # --- How many turns to keep ---
    # 1 turn = 1 human message + 1 AI message = 2 messages
    # So last 3 turns = last 6 messages
    WINDOW_SIZE = 3
    max_messages = WINDOW_SIZE * 2  # 3 turns * 2 messages per turn = 6

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "quit":
            break

        # --- Show current history if user types "history" ---
        if user_input.lower() == "history":
            print(f"\nCurrent history ({len(chat_history)} messages, window={WINDOW_SIZE} turns):")
            for i in range(len(chat_history)):
                print(f"  [{i}] {chat_history[i].type}: {chat_history[i].content[:60]}...")
            continue

        # --- Call the chain with current history ---
        response = chain.invoke({
            "chat_history": chat_history,
            "user_input": user_input
        })
        print("Bot:", response.content)

        # --- Append the new turn ---
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(response)

        # --- Apply the window: keep only the last N turns ---
        # This is the key line. If history has more than max_messages,
        # slice it to keep only the most recent ones.
        if len(chat_history) > max_messages:
            # chat_history[-6:] keeps the last 6 messages (3 turns)
            chat_history = chat_history[-max_messages:]
            print(f"  [Window applied: kept last {WINDOW_SIZE} turns, {len(chat_history)} messages]")
        else:
            print(f"  [History: {len(chat_history)} messages]")


# --- Uncomment the line below to run Exercise 2 ---
# exercise_2()


# ============================================================
# EXERCISE 3: Summary Memory (Summarize Old Messages)
# ============================================================
# Window memory throws away old messages completely.
# Summary memory is smarter: it uses the LLM to summarize
# old messages into a single message, keeping the key facts.
#
# How it works:
# 1. When history gets too long, take the older messages
# 2. Ask the LLM to summarize them into one paragraph
# 3. Replace those messages with a single SystemMessage containing the summary
# 4. Keep the recent messages as-is
#
# Scenario: A Flipkart product comparison chatbot

def exercise_3():
    print("=" * 60)
    print("Exercise 3: Flipkart Bot with Summary Memory")
    print("Type 'quit' to exit, 'history' to see current memory")
    print("=" * 60)

    # --- The main chatbot prompt ---
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Flipkart shopping assistant. You help users compare "
         "products and find the best deals. Keep responses concise."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])

    chain = prompt | model

    # --- A separate chain just for summarizing old messages ---
    # This is a second LLM call that takes old messages and
    # compresses them into a short summary.
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a conversation summarizer. Summarize the following "
         "conversation into a brief paragraph. Keep all important facts "
         "like names, product names, prices, and preferences mentioned "
         "by the user. Be concise."),
        MessagesPlaceholder(variable_name="messages_to_summarize")
    ])

    summary_chain = summary_prompt | model

    chat_history = []

    # --- Summarize when history exceeds this many messages ---
    MAX_MESSAGES_BEFORE_SUMMARY = 8  # 4 turns

    # --- How many recent messages to keep as-is (not summarized) ---
    KEEP_RECENT = 4  # last 2 turns stay untouched

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "history":
            print(f"\nCurrent history ({len(chat_history)} messages):")
            for i in range(len(chat_history)):
                content_preview = chat_history[i].content[:80]
                print(f"  [{i}] {chat_history[i].type}: {content_preview}...")
            continue

        response = chain.invoke({
            "chat_history": chat_history,
            "user_input": user_input
        })
        print("Bot:", response.content)

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(response)

        # --- Check if we need to summarize ---
        if len(chat_history) > MAX_MESSAGES_BEFORE_SUMMARY:
            print("  [Summarizing old messages...]")

            # Split: old messages to summarize, recent messages to keep
            # old_messages = everything except the last KEEP_RECENT messages
            old_messages = chat_history[:-KEEP_RECENT]
            recent_messages = chat_history[-KEEP_RECENT:]

            # Ask the LLM to summarize the old messages
            summary_response = summary_chain.invoke({
                "messages_to_summarize": old_messages
            })

            # Replace the entire history with:
            # 1. A SystemMessage containing the summary (so the bot has context)
            # 2. The recent messages (unchanged)
            chat_history = [
                SystemMessage(content=f"Summary of earlier conversation: {summary_response.content}")
            ] + recent_messages

            print(f"  [Summarized! History now has {len(chat_history)} messages]")
        else:
            print(f"  [History: {len(chat_history)} messages]")


# --- Uncomment the line below to run Exercise 3 ---
# exercise_3()


# ============================================================
# EXERCISE 4: trim_messages with Token Counting
# ============================================================
# In the live demo, we used trim_messages briefly.
# This exercise explores it more deeply.
#
# trim_messages is LangChain's built-in utility for managing
# message list length. It trims by TOKEN count (not message count),
# which is more accurate because a 2-word message and a 200-word
# message use very different amounts of the context window.
#
# Key parameters:
# - max_tokens: the maximum number of tokens to keep
# - token_counter: how to count tokens (pass the model itself)
# - strategy: "last" (keep recent, discard old) or "first" (keep old, discard recent)
# - include_system: True means always keep the system message even when trimming

def exercise_4():
    print("=" * 60)
    print("Exercise 4: trim_messages Deep Dive")
    print("=" * 60)

    # --- Build a long conversation history for testing ---
    test_history = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hi, my name is Rahul and I live in Mumbai."),
        AIMessage(content="Hello Rahul! Nice to meet you. How can I help you today?"),
        HumanMessage(content="I want to learn about investing in mutual funds in India."),
        AIMessage(content="Great choice! Mutual funds are a popular investment option in India. "
                         "You can start with SIPs (Systematic Investment Plans) for as low as Rs 500 per month. "
                         "Popular categories include equity funds, debt funds, and hybrid funds."),
        HumanMessage(content="What is the difference between direct and regular mutual funds?"),
        AIMessage(content="Direct funds have a lower expense ratio because you buy directly from the AMC. "
                         "Regular funds are bought through distributors who charge a commission. "
                         "Over 10 years, the difference can be 1-2% in returns."),
        HumanMessage(content="Which app should I use to invest?"),
        AIMessage(content="For direct mutual funds, popular apps in India include Groww, Zerodha Coin, "
                         "and Kuvera. All three are free for direct mutual fund investments. "
                         "Groww has the simplest interface for beginners."),
    ]

    print(f"Original history: {len(test_history)} messages\n")

    # --- Example 1: Keep last 100 tokens ---
    # strategy="last" keeps the most recent messages
    trimmed_last = trim_messages(
        test_history,
        max_tokens=100,
        token_counter=model,
        strategy="last",
        include_system=True  # Always keep the system message
    )

    print("After trim (strategy='last', max_tokens=100):")
    print(f"  Kept {len(trimmed_last)} messages:")
    for i in range(len(trimmed_last)):
        print(f"  [{i}] {trimmed_last[i].type}: {trimmed_last[i].content[:60]}...")

    # --- Example 2: Keep first 100 tokens ---
    # strategy="first" keeps the oldest messages
    trimmed_first = trim_messages(
        test_history,
        max_tokens=100,
        token_counter=model,
        strategy="first",
        include_system=True
    )

    print(f"\nAfter trim (strategy='first', max_tokens=100):")
    print(f"  Kept {len(trimmed_first)} messages:")
    for i in range(len(trimmed_first)):
        print(f"  [{i}] {trimmed_first[i].type}: {trimmed_first[i].content[:60]}...")

    # --- Example 3: Use trim_messages in a chain ---
    # You can integrate trim_messages directly into the chain
    # so it trims automatically on every call.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful financial advisor for Indian investors."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])

    chain = prompt | model

    # Before each call, trim the history to fit within budget
    trimmed = trim_messages(
        test_history,
        max_tokens=200,
        token_counter=model,
        strategy="last",
        include_system=True
    )

    print(f"\n\nUsing trimmed history in a chain call:")
    print(f"  Trimmed from {len(test_history)} to {len(trimmed)} messages")

    response = chain.invoke({
        "chat_history": trimmed,
        "user_input": "What was my name and what did we discuss?"
    })
    print(f"  Bot: {response.content}")


# --- Uncomment the line below to run Exercise 4 ---
# exercise_4()


# ============================================================
# EXERCISE 5: Multi-Conversation Persistence (Multiple Users)
# ============================================================
# In a real app, you have multiple users. Each user has their
# own conversation history. You need to save and load history
# per user, not globally.
#
# This exercise builds a simple system where:
# - Each user gets their own JSON file
# - History is loaded when a user starts chatting
# - History is saved after each turn

def exercise_5():
    print("=" * 60)
    print("Exercise 5: Multi-User Chat Persistence")
    print("=" * 60)

    # --- Helper functions for save/load ---
    def get_history_filename(user_id):
        """Generate a filename for a user's chat history."""
        # Each user gets their own file: chat_history_priya.json, chat_history_rahul.json
        return f"chat_history_{user_id}.json"

    def save_history(history, user_id):
        """Save a user's chat history to their JSON file."""
        filename = get_history_filename(user_id)
        data = []
        for i in range(len(history)):
            msg = history[i]
            data.append({
                "type": msg.type,
                "content": msg.content
            })
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def load_history(user_id):
        """Load a user's chat history. Returns empty list if no file exists."""
        filename = get_history_filename(user_id)

        # --- Check if file exists ---
        # os.path.exists() returns True if the file is there, False if not
        if not os.path.exists(filename):
            print(f"  No previous history for user '{user_id}'. Starting fresh.")
            return []

        with open(filename, "r") as f:
            data = json.load(f)

        history = []
        for i in range(len(data)):
            item = data[i]
            if item["type"] == "human":
                history.append(HumanMessage(content=item["content"]))
            else:
                history.append(AIMessage(content=item["content"]))

        print(f"  Loaded {len(history)} messages for user '{user_id}'.")
        return history

    # --- Setup ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly assistant. Remember user details across conversations."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])

    chain = prompt | model

    # --- Simulate two users chatting ---

    # User 1: Priya
    print("\n--- Session 1: Priya ---")
    priya_history = load_history("priya")

    response = chain.invoke({
        "chat_history": priya_history,
        "user_input": "Hi, I'm Priya from Pune. I love biryani."
    })
    print(f"Bot to Priya: {response.content}")
    priya_history.append(HumanMessage(content="Hi, I'm Priya from Pune. I love biryani."))
    priya_history.append(response)
    save_history(priya_history, "priya")

    # User 2: Rahul
    print("\n--- Session 1: Rahul ---")
    rahul_history = load_history("rahul")

    response = chain.invoke({
        "chat_history": rahul_history,
        "user_input": "Hey, I'm Rahul from Mumbai. I'm a cricket fan."
    })
    print(f"Bot to Rahul: {response.content}")
    rahul_history.append(HumanMessage(content="Hey, I'm Rahul from Mumbai. I'm a cricket fan."))
    rahul_history.append(response)
    save_history(rahul_history, "rahul")

    # --- Simulate app restart ---
    print("\n--- [APP RESTART] ---")
    print("All variables are cleared. Loading from files...\n")

    # Reload Priya's history from file
    print("--- Session 2: Priya ---")
    priya_history = load_history("priya")

    response = chain.invoke({
        "chat_history": priya_history,
        "user_input": "What's my name and where am I from?"
    })
    print(f"Bot to Priya: {response.content}")

    # Reload Rahul's history from file
    print("\n--- Session 2: Rahul ---")
    rahul_history = load_history("rahul")

    response = chain.invoke({
        "chat_history": rahul_history,
        "user_input": "Do you remember what I like?"
    })
    print(f"Bot to Rahul: {response.content}")

    # --- Cleanup ---
    os.remove(get_history_filename("priya"))
    os.remove(get_history_filename("rahul"))
    print("\nCleaned up history files.")


# --- Uncomment the line below to run Exercise 5 ---
# exercise_5()


# ============================================================
# EXERCISE 6: Support Chatbot with Persona and Memory
# ============================================================
# Build a complete customer support chatbot for boAt (the audio
# brand). The bot should:
# - Have a specific persona (friendly boAt support agent)
# - Remember the user's name and product throughout the conversation
# - Track the issue type (warranty, refund, technical)
# - Use window memory to keep conversation manageable
# - Save history to JSON on exit
#
# This is the closest to a real production chatbot.

def exercise_6():
    print("=" * 60)
    print("Exercise 6: boAt Customer Support Chatbot")
    print("Type 'quit' to exit (history will be saved)")
    print("=" * 60)

    HISTORY_FILE = "boat_support_history.json"

    # --- Load existing history if any ---
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
        chat_history = []
        for i in range(len(data)):
            item = data[i]
            if item["type"] == "human":
                chat_history.append(HumanMessage(content=item["content"]))
            elif item["type"] == "system":
                chat_history.append(SystemMessage(content=item["content"]))
            else:
                chat_history.append(AIMessage(content=item["content"]))
        print(f"Resumed previous conversation ({len(chat_history)} messages)")
    else:
        chat_history = []
        print("Starting new conversation")

    # --- The persona prompt ---
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer support agent for boAt, the Indian audio brand. "
         "Your name is Anika. You are friendly, patient, and solution-oriented. "
         "Guidelines:\n"
         "- Always address the customer by name once they share it\n"
         "- Ask for order ID or product name early in the conversation\n"
         "- Categorize the issue: warranty, refund, exchange, technical support\n"
         "- Offer clear next steps\n"
         "- If you cannot resolve, offer to escalate to a senior agent\n"
         "- Keep responses under 3 sentences"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])

    chain = prompt | model

    # --- Window memory settings ---
    WINDOW_SIZE = 5  # Keep last 5 turns = 10 messages
    max_messages = WINDOW_SIZE * 2

    while True:
        user_input = input("\nCustomer: ")

        if user_input.lower() == "quit":
            # --- Save history before exiting ---
            data = []
            for i in range(len(chat_history)):
                msg = chat_history[i]
                data.append({
                    "type": msg.type,
                    "content": msg.content
                })
            with open(HISTORY_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print(f"History saved to {HISTORY_FILE}. Goodbye!")
            break

        # --- Apply window before calling the chain ---
        # This ensures we never send too many messages to the model
        windowed_history = chat_history
        if len(chat_history) > max_messages:
            windowed_history = chat_history[-max_messages:]

        response = chain.invoke({
            "chat_history": windowed_history,
            "user_input": user_input
        })
        print(f"Anika (boAt Support): {response.content}")

        # --- Append to full history (for persistence) ---
        # We keep the FULL history in the file but only send the
        # windowed version to the model.
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(response)

    # --- Cleanup (remove this line if you want to keep the history file) ---
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


# --- Uncomment the line below to run Exercise 6 ---
# exercise_6()


# ============================================================
# EXERCISE 7: Compare Full History vs Window vs Summary
# ============================================================
# This exercise runs the SAME conversation through three
# different memory strategies and compares the results.
#
# You'll see how each strategy handles forgetting:
# - Full history: remembers everything but uses the most tokens
# - Window memory: forgets old information
# - Summary memory: remembers key facts but loses detail

def exercise_7():
    print("=" * 60)
    print("Exercise 7: Memory Strategy Comparison")
    print("=" * 60)

    # --- A pre-written conversation (8 turns) ---
    # We simulate a user planning a trip to Goa
    conversation_turns = [
        "Hi, I'm Meera. I want to plan a trip to Goa.",
        "I'll be traveling with 3 friends. We're all from Bangalore.",
        "Our budget is Rs 15,000 per person for 4 days.",
        "We prefer beaches on the north side of Goa.",
        "Can you suggest some budget hotels near Baga Beach?",
        "What about restaurants? We love seafood.",
        "Any water sports you'd recommend?",
        "Great, now can you summarize everything we discussed? Include my name, group size, budget, and all recommendations."
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a travel planner for trips in India. Be concise, limit responses to 2-3 sentences."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])

    chain = prompt | model

    # --- Summary chain for summary memory ---
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Summarize this conversation into one brief paragraph. "
         "Keep all important facts: names, numbers, preferences, recommendations."),
        MessagesPlaceholder(variable_name="messages_to_summarize")
    ])
    summary_chain = summary_prompt | model

    # ========== Strategy 1: Full History ==========
    print("\n" + "=" * 40)
    print("STRATEGY 1: Full History (keep everything)")
    print("=" * 40)

    full_history = []
    for i in range(len(conversation_turns)):
        user_msg = conversation_turns[i]
        response = chain.invoke({
            "chat_history": full_history,
            "user_input": user_msg
        })
        full_history.append(HumanMessage(content=user_msg))
        full_history.append(response)

    # The last response is the summary
    print(f"\nFinal response (using {len(full_history)} messages):")
    print(full_history[-1].content)

    # ========== Strategy 2: Window Memory (last 3 turns) ==========
    print("\n" + "=" * 40)
    print("STRATEGY 2: Window Memory (last 3 turns)")
    print("=" * 40)

    window_history = []
    for i in range(len(conversation_turns)):
        user_msg = conversation_turns[i]

        # Apply window before each call
        windowed = window_history[-6:] if len(window_history) > 6 else window_history

        response = chain.invoke({
            "chat_history": windowed,
            "user_input": user_msg
        })
        window_history.append(HumanMessage(content=user_msg))
        window_history.append(response)

    print(f"\nFinal response (window=3 turns, full history was {len(window_history)} messages):")
    print(window_history[-1].content)

    # ========== Strategy 3: Summary Memory ==========
    print("\n" + "=" * 40)
    print("STRATEGY 3: Summary Memory (summarize after 6 messages)")
    print("=" * 40)

    summary_history = []
    for i in range(len(conversation_turns)):
        user_msg = conversation_turns[i]

        response = chain.invoke({
            "chat_history": summary_history,
            "user_input": user_msg
        })
        summary_history.append(HumanMessage(content=user_msg))
        summary_history.append(response)

        # Summarize when history gets too long
        if len(summary_history) > 6:
            old_messages = summary_history[:-4]
            recent_messages = summary_history[-4:]

            summary_response = summary_chain.invoke({
                "messages_to_summarize": old_messages
            })

            summary_history = [
                SystemMessage(content=f"Summary of earlier conversation: {summary_response.content}")
            ] + recent_messages

    print(f"\nFinal response (with summarization, {len(summary_history)} messages):")
    print(summary_history[-1].content)

    # ========== Comparison ==========
    print("\n" + "=" * 40)
    print("COMPARISON")
    print("=" * 40)
    print(f"Full History: used {len(full_history)} messages")
    print(f"Window Memory: used 6 messages (last 3 turns) for each call")
    print(f"Summary Memory: used {len(summary_history)} messages (summary + recent)")
    print("\nNotice how:")
    print("- Full history remembers everything (name, budget, all details)")
    print("- Window memory may forget the name and budget from early turns")
    print("- Summary memory keeps key facts but may lose specific recommendations")


# --- Uncomment the line below to run Exercise 7 ---
# exercise_7()


# ============================================================
# QUICK REFERENCE: What We Learned in Week 6
# ============================================================
#
# 1. LLMs are stateless: every API call starts fresh with no memory
#
# 2. Manual message history: maintain a Python list of HumanMessage
#    and AIMessage objects. Pass it via MessagesPlaceholder.
#
# 3. Three strategies for managing long histories:
#    a) Full history: keep everything (accurate but expensive)
#    b) Window memory: keep last N turns using list slicing [-N:]
#    c) Summary memory: use LLM to summarize old messages
#
# 4. trim_messages: LangChain's built-in utility for trimming
#    by token count (more accurate than message count)
#    from langchain_core.messages import trim_messages
#
# 5. Persistence: save history as JSON, load on startup
#    Convert HumanMessage/AIMessage to dicts for JSON serialization
#
# 6. Each strategy has tradeoffs:
#    - Full: most accurate, most expensive, can exceed context window
#    - Window: cheapest, loses old context completely
#    - Summary: balanced, keeps key facts, costs an extra LLM call
