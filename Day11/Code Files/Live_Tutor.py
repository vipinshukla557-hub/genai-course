
# ═══════════════════════════════════════════════════════════════════════════
#                   PART 1: GRAPHS, ROUTING & REACT AGENT
# ═══════════════════════════════════════════════════════════════════════════


# ── Step 1: Imports for our first graph ─────────────────────────────────

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

# ── Step 2: Define the State ────────────────────────────────────────────

class Mystate(TypedDict):
    messages: Annotated[list, operator.add]

# ── Step 3: Create two nodes ───────────────────────────────────────────

def greet(state):
    print("[greet] Running....")
    return {"messages":["Namaste! welcome to langgraph"]}

def farewell(state):
    print(" [farewell] Running.....")
    return {"messages":["See you later!"]}

# ── Step 4: Build and run the graph ─────────────────────────────────────

workflow = StateGraph(Mystate)

workflow.add_node("greet",greet)
workflow.add_node("farewell", farewell)

workflow.add_edge(START,"greet")
workflow.add_edge("greet","farewell")
workflow.add_edge("farewell", END)

graph = workflow.compile()


result = graph.invoke({"messages":["User has entered the graph."]})

print("Final state message", result["messages"])

# ── Step 5: Conditional Routing — Define the State ──────────────────────

class TicketState(TypedDict):
    complaint: str
    category: str
    response: str

# ── Step 6: Nodes + routing function ───────────────────────────────────

def classify(state):
    text = state["complaint"].lower()
    cat = "billing" if any(w in text for w in ["refund","charge","bill"]) else "general"
    print(" [classify] ", cat)
    return {"category": cat}

# handler nodes

def billing_handler(state):
    return {"response": "Billing: refund of x amount initiated. allow x number of days"}

def general_handler(state):
    return {"response":"Support: Team will follow up in 72 hours"}


def route_ticket(state):
    return state["category"]

# ── Step 7: Build the conditional graph + test ─────────────────────────

router = StateGraph(TicketState)
router.add_node("classify", classify)
router.add_node("billing", billing_handler)
router.add_node("general", general_handler)

router.add_edge(START, "classify")
router.add_conditional_edges("classify", route_ticket, {
    "billing": "billing",
    "general": "general"
})
router.add_edge("billing", END)
router.add_edge("general", END)

ticket_graph = router.compile()

print("\n--- Billing complaint ---")
r1 = ticket_graph.invoke({"complaint": "I was charged twice!", "category": "", "response": ""})
print("Response:", r1["response"])

print("\n--- General complaint ---")
r2 = ticket_graph.invoke({"complaint": "The paneer was raw.", "category": "", "response": ""})
print("Response:", r2["response"])


# ── Step 8: ReAct Agent — imports ──────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.prebuilt import create_react_agent

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Step 9: Define the tools ──────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    # The default ``auto`` backend can select Wikipedia and fail when that
    # host is unavailable. DuckDuckGo is more reliable for this tool.
    search = DuckDuckGoSearchRun(
        api_wrapper=DuckDuckGoSearchAPIWrapper(
            region="us-en",
            backend="duckduckgo",
            time=None,
            max_results=5,
        )
    )
    try:
        return search.run(query)
    except Exception as exc:
        return f"Web search temporarily unavailable: {exc}"

@tool
def calculator(expression: str) -> str:
    """Calculate a math expression like '15 * 100'."""
    try:
        return str(eval(expression))
    except Exception as e:
        return "Error: " + str(e)


# ── Step 10: Create and run the ReAct agent ────────────────────────────

agent = create_react_agent(model, [web_search, calculator])

print("\n--- ReAct Agent (LangGraph) ---")
result = agent.invoke({
    "messages": [HumanMessage(content="What is the population of Pune and what is 10% of that?")]
})

messages = result["messages"]
final_answer = messages[len(messages) - 1]
print("Answer:", final_answer.content[:200])


# ═══════════════════════════════════════════════════════════════════════════
#                         ─── BREAK (5 min) ───
# ═══════════════════════════════════════════════════════════════════════════


# ── Step 11: Create agent WITH checkpointing ───────────────────────────

from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
agent = create_react_agent(model, [web_search, calculator], checkpointer=checkpointer)

config = {"configurable": {"thread_id": "demo-thread"}}


# ── Step 12: Multi-turn conversation (agent remembers!) ────────────────

print("--- Turn 1 ---")
r1 = agent.invoke(
    {"messages": [HumanMessage(content="What is the population of Mumbai?")]},
    config=config
)
answer1 = r1["messages"][len(r1["messages"]) - 1]
print("Answer:", answer1.content[:150])

print("\n--- Turn 2 (agent remembers Turn 1!) ---")
r2 = agent.invoke(
    {"messages": [HumanMessage(content="And what is 20% of that?")]},
    config=config
)
answer2 = r2["messages"][len(r2["messages"]) - 1]
print("Answer:", answer2.content[:150])


# ── Step 13: Different thread = no memory ──────────────────────────────

print("\n--- Different thread (no memory) ---")
r3 = agent.invoke(
    {"messages": [HumanMessage(content="What did we just talk about?")]},
    config={"configurable": {"thread_id": "fresh-thread"}}
)
answer3 = r3["messages"][len(r3["messages"]) - 1]
print("Answer:", answer3.content[:150])
print("  ^ No memory of Mumbai — different thread_id!")


# ── Step 14: LangSmith — quick check ──────────────────────────────────

print("\n--- LangSmith Check ---")

tracing = os.getenv("LANGSMITH_TRACING", "True")
if tracing.lower() == "true":
    print("LangSmith is ON! Check smith.langchain.com for traces.")
else:
    print("LangSmith not configured. Add to .env:")
    print("  LANGSMITH_TRACING=true")
    print("  LANGSMITH_API_KEY=your_key")
    print("  LANGSMITH_PROJECT=week11-agents")
