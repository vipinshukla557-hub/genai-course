# 🤖 Smart CLI Chatbot

A command-line AI chatbot built with Python and the OpenAI-compatible SDK. Supports multiple AI providers (Groq, OpenAI), persona switching, streaming responses, cost tracking, and automatic conversation management.

**Built as Project 1 of the Applied GenAI Engineering Program.**

---

## Features

- **Streaming responses** — see the AI's reply word by word, just like ChatGPT
- **Conversation history** — the AI remembers what you've discussed
- **Persona switching** — switch between Default, Coder, and Teacher personalities mid-chat
- **Cost tracking** — know exactly how many tokens you've used and what it costs
- **Error handling** — retries on rate limits and timeouts, never crashes
- **Sliding window** — automatically trims old messages so you never exceed the context limit
- **Provider swap** — switch between OpenAI and Groq (free) with a one-line config change

## How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   You type   │────>│  Messages Array   │────>│ OpenAI API  │
│  in terminal │     │  (full history)   │     │  (or Groq)  │
└─────────────┘     └──────────────────┘     └──────┬──────┘
       ▲                                            │
       │            ┌──────────────────┐            │
       └────────────│  Stream response │<───────────┘
                    │  token by token  │
                    └──────────────────┘
```

The AI has no memory. Every API call sends the **entire conversation history** as input. The chatbot manages this history automatically — adding new messages, trimming old ones when the conversation gets too long, and swapping the system prompt when you change personas.

## Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/smart-cli-chatbot.git
cd smart-cli-chatbot

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install openai python-dotenv

# 4. Add your API key
echo "OPENAI_API_KEY=your_key_here" > .env

# 5. Run the chatbot
python week4_section6_complete_chatbot.py
```

## Commands

| Command | What it does |
|---------|-------------|
| `/persona` | List available personas |
| `/persona coder` | Switch to Code Expert persona |
| `/persona teacher` | Switch to Patient Teacher persona |
| `/cost` | Show tokens used and estimated cost |
| `/history` | Show conversation length |
| `/provider` | Show current API provider |
| `/clear` | Clear conversation history |
| `/help` | Show all commands |
| `/quit` | Exit (shows final cost summary) |

## Getting an API Key

This project uses **OpenAI (GPT-4o-mini)** by default:

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up and add billing (GPT-4o-mini is very cheap — a full chat session costs fractions of a paisa)
3. Create an API key under API Keys
4. Add it to your `.env` file as `OPENAI_API_KEY=your_key_here`

**Free alternative:** Use [Groq](https://console.groq.com) (free tier, no billing needed). Change `ACTIVE_PROVIDER = "groq"` in the code and add `GROQ_API_KEY` to your `.env`.

## Project Structure

```
smart-cli-chatbot/
├── week4_section6_complete_chatbot.py   # The complete chatbot (run this)
├── .env                                  # Your API key (never commit this!)
├── .gitignore                            # Protects .env from being pushed
└── README.md                             # This file
```

## Key Concepts Demonstrated

- **OpenAI-compatible SDK** — one SDK works with Groq, OpenAI, Together, and more
- **Messages array** — system/user/assistant roles structure every conversation
- **Streaming** — `stream=True` for real-time token-by-token output
- **Error handling** — retry logic with exponential backoff for rate limits
- **Context management** — sliding window keeps conversations within token limits

## Tech Stack

- Python 3.10+
- OpenAI SDK (works with any OpenAI-compatible provider)
- OpenAI API (GPT-4o-mini) — with Groq as free swap-in alternative
- python-dotenv for environment variable management
