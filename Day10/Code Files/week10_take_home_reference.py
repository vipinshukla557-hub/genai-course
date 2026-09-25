# ============================================================
# WEEK 10: AI Agents & Custom Tools — Take-Home Reference
# ============================================================
#
# This file is NOT for live coding. It's your after-class
# reference for deeper concepts we didn't have time to cover.
#
# Run each section independently — they're self-contained.
#
# SETUP (same as live session):
#   pip install langchain langchain-openai langgraph langchain-community
#   export OPENAI_API_KEY="your-key"
# ============================================================


# ============================================================
# SECTION 1: THE AGENT LOOP EXPLAINED
# ============================================================
# In the live session we saw the agent "think" and "pick tools."
# Here's what's actually happening under the hood:
#
#   1. THINK — The LLM reads your question + tool schemas
#      and decides: "Do I need a tool, or can I answer directly?"
#
#   2. DO — If it needs a tool, it generates a "tool call":
#      {"name": "check_leave_balance", "args": {"employee_name": "Priya"}}
#      The framework calls YOUR Python function with those args.
#
#   3. SEE — The tool's return value goes back to the LLM as a
#      new message. The LLM reads it and decides: done, or need
#      another tool call?
#
#   4. REPEAT — This loop continues until the LLM decides it has
#      enough information to answer. Then it generates a final
#      text response.
#
# This is called the ReAct pattern (Reason + Act).
# Paper: "ReAct: Synergizing Reasoning and Acting in Language Models"
#        by Yao et al., 2023
#
# INTERVIEW QUESTION: "What's the difference between a chain and an agent?"
# ANSWER: A chain has a fixed sequence of steps decided at build time.
#         An agent decides its steps at runtime based on the input.
#         A chain is a recipe. An agent is a cook who reads the recipe
#         but adjusts based on what's in the fridge.


# ============================================================
# SECTION 2: THREE WAYS AGENTS BREAK (and how to fix them)
# ============================================================

import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Add it to the repository .env file "
        "or set it in your environment before running this example."
    )

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# --- FAILURE 1: The Infinite Loop ---
# What happens when the agent keeps calling tools but never
# finds a satisfying answer?

@tool
def vague_search(query: str) -> str:
    """Search for anything."""
    return f"Here are some general results about {query}..."
    # Problem: The result is so vague that the LLM tries again,
    # and again, and again — burning tokens each time.

# In production, you'd see this error:
#   GraphRecursionError: Recursion limit of 25 reached
#
# FIX 1: Return specific, useful data (not vague filler)
# FIX 2: Set a recursion limit to fail gracefully:
#
# from langgraph.errors import GraphRecursionError
# agent = create_agent(llm, tools=[vague_search])
# try:
#     response = agent.invoke(
#         {"messages": [("user", "Find me everything about AI")]},
#         {"recursion_limit": 10}
#     )
# except GraphRecursionError:
#     print("Agent hit the loop limit — question was too open-ended")


# --- FAILURE 2: Wrong Tool Selection ---
# When two tools have overlapping or vague descriptions,
# the LLM picks the wrong one.

@tool
def get_data(name: str) -> str:
    """Get data for a person."""  # Too vague — could be anything
    return f"Department: Engineering, Role: Developer"

@tool
def get_leave(name: str) -> str:
    """Get data for a person."""  # Same vague description!
    return f"Leave balance: 15 days"

# With identical descriptions, the LLM has no way to choose.
# It's literally a coin flip.
#
# FIX: Unique, specific descriptions:
#   get_data → "Look up employee profile: department, role, join date"
#   get_leave → "Check remaining leave balance in days for this quarter"


# --- FAILURE 3: Hallucinated Arguments ---
# The LLM invents argument values that don't match what your
# tool expects.

@tool
def get_stock_price(ticker: str) -> str:
    """Get current stock price for a company.
    The ticker must be a valid NSE/BSE symbol like RELIANCE, TCS, INFY."""
    prices = {"RELIANCE": 2850, "TCS": 3920, "INFY": 1650}
    price = prices.get(ticker.upper())
    if not price:
        return f"Ticker '{ticker}' not found. Valid tickers: {', '.join(prices.keys())}"
    return f"{ticker.upper()}: ₹{price}"

# If you ask "What's the Reliance stock price?", the LLM might pass:
#   "Reliance" (wrong case), "RELIANCE.NS" (added suffix), or
#   "Reliance Industries" (full name instead of ticker)
#
# FIX: Helpful error messages that teach the LLM what you need.
#   Notice the tool returns valid tickers on failure — the LLM
#   reads that, self-corrects, and retries with the right value.
#
# PRODUCTION PATTERN: "Fail loud, fail helpful."
#   Bad:  return "Error"
#   Good: return f"Invalid ticker '{ticker}'. Try one of: RELIANCE, TCS, INFY"

agent = create_agent(llm, tools=[get_stock_price])
response = agent.invoke({"messages": [("user", "How much is Reliance stock right now?")]})
print(response["messages"][-1].content)


# ============================================================
# SECTION 3: DEBUGGING — Seeing What the Agent Thinks
# ============================================================
# When your agent does something unexpected, you need to see
# the Think → Do → See loop in action.

from langchain_core.globals import set_debug

# Turn on debug mode — shows every LLM call, tool call, and result
set_debug(True)

@tool
def check_inventory(product: str) -> str:
    """Check warehouse inventory for a product. Returns quantity in stock."""
    stock = {"laptop": 42, "mouse": 150, "keyboard": 88}
    qty = stock.get(product.lower())
    if qty is None:
        return f"Product '{product}' not in warehouse. Available: {', '.join(stock.keys())}"
    return f"{product}: {qty} units in stock"

agent = create_agent(llm, tools=[check_inventory])
response = agent.invoke({"messages": [("user", "Do we have enough laptops for a team of 50?")]})
print(response["messages"][-1].content)

set_debug(False)  # Turn off so the rest of this file isn't noisy

# In the debug output, look for:
#   [llm/start]     — what the LLM received (your question + tool schemas)
#   [tool/start]    — which tool it called and with what arguments
#   [tool/end]      — what your tool returned
#   [llm/start]     — the LLM reading the tool result + deciding next step
#   [llm/end]       — the final answer
#
# PRODUCTION TIP: Don't use set_debug(True) in production.
#   Use LangSmith (LangChain's tracing platform) or structured
#   logging instead. set_debug is for development only.


# ============================================================
# SECTION 4: THE TOOL LANDSCAPE — Three Ways to Give LLMs Tools
# ============================================================
# This is the "where does @tool fit in the bigger picture" section.
#
# There are three layers of tool integration in the LLM ecosystem:
#
# ┌─────────────────────────────────────────────────────┐
# │  LAYER 3: MCP (Model Context Protocol)              │
# │  Universal plug-and-play. One server, any LLM app.  │
# │  Think of it like USB-C for AI tools.               │
# │  Example: A Slack MCP server works in Claude,        │
# │  ChatGPT, Cursor, and your custom app — unchanged.  │
# ├─────────────────────────────────────────────────────┤
# │  LAYER 2: Framework Tools (LangChain @tool)          │
# │  What we learned today. Works within LangChain.      │
# │  More portable than native FC, less than MCP.        │
# │  Example: @tool works with OpenAI, Anthropic,        │
# │  Google models — as long as you use LangChain.       │
# ├─────────────────────────────────────────────────────┤
# │  LAYER 1: Native Function Calling                    │
# │  Raw API-level tool use. Provider-specific format.   │
# │  OpenAI has one JSON format, Anthropic another.      │
# │  Most control, least portable.                       │
# └─────────────────────────────────────────────────────┘
#
# INTERVIEW QUESTION: "When would you NOT use an agent?"
# ANSWER: When the task has a fixed, predictable sequence of steps.
#   A document summarizer doesn't need an agent — it's always:
#   load doc → chunk → summarize → combine. No decisions needed.
#   Agents add latency (multiple LLM calls), cost (more tokens),
#   and unpredictability (the LLM might pick wrong tools).
#   Use an agent when the workflow genuinely requires runtime
#   decisions. Use a chain when the steps are known upfront.


# ============================================================
# SECTION 5: PRODUCTION PATTERNS — What Real Agent Tools Look Like
# ============================================================

# PATTERN 1: Input Validation
# Don't trust the LLM to always pass clean inputs.

@tool
def transfer_budget(from_dept: str, to_dept: str, amount: float) -> str:
    """Transfer budget between departments.
    from_dept and to_dept must be valid department names.
    amount must be positive and under 50000."""
    valid_depts = {"Engineering", "Marketing", "HR", "Finance"}
    errors = []
    if from_dept not in valid_depts:
        errors.append(f"'{from_dept}' is not a valid department. Choose from: {valid_depts}")
    if to_dept not in valid_depts:
        errors.append(f"'{to_dept}' is not a valid department. Choose from: {valid_depts}")
    if amount <= 0:
        errors.append("Amount must be positive")
    if amount > 50000:
        errors.append(f"Amount {amount} exceeds single-transfer limit of 50,000")
    if errors:
        return "VALIDATION FAILED:\n" + "\n".join(errors)
    return f"Transferred ₹{amount:,.0f} from {from_dept} to {to_dept}"


# PATTERN 2: Tools With Side Effects Need Guardrails
# Any tool that WRITES data (not just reads) is dangerous.
# The principle of least privilege: give the agent the minimum
# access it needs.
#
# BAD:  Give the agent a tool that can delete database records
# OK:   Give the agent a tool that can flag records for review
# GOOD: Give the agent a read-only tool + require human approval
#       for any write operation
#
# INTERVIEW QUESTION: "How do you handle tools with side effects?"
# ANSWER: Three strategies:
#   1. Confirmation step — agent proposes action, human approves
#   2. Dry-run mode — tool returns what WOULD happen without doing it
#   3. Principle of least privilege — read-only tools by default,
#      write access only with explicit human-in-the-loop


# PATTERN 3: Never Use eval() in a Tool
# If the LLM can control what gets evaluated, it can run
# arbitrary code on your server.

# DANGEROUS — the LLM could pass "os.system('rm -rf /')"
# @tool
# def calculate(expression: str) -> str:
#     """Evaluate a math expression."""
#     return str(eval(expression))

# SAFE — use a math parser or restrict to known operations
# from ast import literal_eval  # Only evaluates literals
# Or better: use a dedicated math library like numexpr


# ============================================================
# SECTION 6: EXERCISE — Design Your Own Tool
# ============================================================
# Pick a scenario from your work and build a tool for it.
# Use this template:

# @tool
# def your_tool_name(param1: str, param2: int = 10) -> str:
#     """
#     [ONE LINE: What this tool does]
#     [ONE LINE: When the LLM should use this tool]
#     [ONE LINE: What the parameters mean]
#     """
#     # Your logic here
#     return "result as a string the LLM can read"

# CHECKLIST:
# □ Does the function name clearly say what it does?
# □ Does the docstring explain WHEN to use it (not just what)?
# □ Are parameter types annotated (str, int, float)?
# □ Does it return helpful error messages on bad input?
# □ Is the return value a string the LLM can understand?
# □ Does it have side effects? If yes, add guardrails.

# IDEAS:
# - Ticket status lookup (Jira/ServiceNow style)
# - Customer account lookup
# - Server health checker
# - Meeting room availability
# - Knowledge base search
# - Approval workflow trigger (with confirmation)


# ============================================================
# SECTION 7: FIVE INTERVIEW QUESTIONS ON AI AGENTS
# ============================================================
#
# Q1: "What's the difference between a chain and an agent?"
# A: A chain executes a fixed sequence of steps defined at build time.
#    An agent uses an LLM to decide which steps to take at runtime.
#    Chains are predictable and efficient. Agents are flexible but
#    slower, costlier, and less predictable.
#
# Q2: "What is the ReAct pattern?"
# A: ReAct = Reason + Act. The LLM alternates between reasoning
#    about what to do (Think), executing an action like a tool call
#    (Do), and observing the result (See). It loops until it has
#    enough information to answer.
#
# Q3: "How do you prevent an agent from running forever?"
# A: Set a recursion_limit (max number of Think→Do→See cycles).
#    Write clear tool descriptions so the agent finds answers quickly.
#    Return specific, useful data from tools so the agent doesn't
#    need to retry. Monitor token usage in production.
#
# Q4: "When should you NOT use an agent?"
# A: When the task has a fixed, known sequence of steps. A document
#    summarizer, a data pipeline, an ETL job — these don't need
#    runtime decisions. Agents add latency (multiple LLM calls),
#    cost (more tokens), and unpredictability. Use the simplest
#    architecture that solves the problem.
#
# Q5: "How do you handle agent tools that have side effects?"
# A: Three strategies: (1) Human-in-the-loop confirmation for
#    write/delete/send operations, (2) Dry-run mode where the tool
#    returns what it WOULD do without executing, (3) Principle of
#    least privilege — give read-only access by default and require
#    explicit approval for writes. Never give an agent unrestricted
#    access to production systems.
