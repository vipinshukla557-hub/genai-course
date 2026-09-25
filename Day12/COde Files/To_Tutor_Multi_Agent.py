
# ── IMPORTS ──────────────────────────────────────────────────

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
# load_dotenv()

from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
# from langgraph.prebuilt import create_react_agent

# ── LLM SETUP ───────────────────────────────────────────────

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ── TOOL: Web Search ─────────────────────────────────────────


from ddgs import DDGS #duck duck go search

def web_search(query: str) -> str:
    """Search the web and return top results."""
    search = DDGS()
    results = search.text(query, max_results=3)

    # results → a list of dicts, each with 'title', 'body', 'href'
    output = ""
    for i in range(len(results)):
        output = output + f"Title: {results[i]['title']}\n"
        output = output + f"Summary: {results[i]['body']}\n"
        output = output + f"URL: {results[i]['href']}\n\n"

    return output


# ── AGENT 1: Research Agent ──────────────────────────────────

research_agent = create_agent(
    model=model,
    tools=[web_search],
    name="researcher",
    system_prompt="You are a research specialist. When given a topic, "
           "search the web and return the key facts you find. "
           "Be thorough but concise. Always cite your sources."
)


# ── AGENT 2: Writer Agent ───────────────────────────────────

writer_agent = create_agent(
    model=model,
    tools=[],
    name="writer",
    system_prompt="You are a professional writer. When given research notes, "
           "write a clear, well-structured summary with bullet points. "
           "Keep it under 200 words. Use simple language."
)
# add a tool to download the generated text into an .md file on your workspace

# ── SUPERVISOR ───────────────────────────────────────────────

workflow = create_supervisor(
    agents=[research_agent, writer_agent],
    model=model,
    prompt=(
        "You are a team lead managing a researcher and a writer. "
        "For any user request:\n"
        "1. Send the topic to the researcher first\n"
        "2. Once research is done, send it to the writer for a summary\n"
        "3. Return the writer's summary as the final answer"
    )
)

app = workflow.compile()


# ── RUN IT ───────────────────────────────────────────────────

result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What are the top 3 AI trends in India in 2026?"
        }
    ]
})


# ── PRINT THE RESULT ─────────────────────────────────────────

print("\n" + "=" * 50)
print("FINAL ANSWER:")
print("=" * 50)

all_messages = result["messages"]
final_message = all_messages[-1]
print(final_message.content)

print("\n" + "=" * 50)
print("FULL CONVERSATION TRACE:")
print("=" * 50)
for msg in all_messages:
    sender = getattr(msg, "name", msg.type)
    print(f"\n[{sender}]")
    print(msg.content[:200])