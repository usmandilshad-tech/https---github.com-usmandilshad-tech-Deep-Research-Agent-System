# research_agents.py
from agents import Agent
from sdk import model_cheap, model_smart
from tools import web_search, fetch_url, citation_check

# ----- Fact Finder -----
FACTFINDER_SYS = (
    "You are a meticulous fact-finding researcher.\n"
    "Workflow:\n"
    "1) Propose 3–5 focused web search queries.\n"
    "2) Use web_search for each (k≈5–8) and pick the most relevant URLs.\n"
    "3) Use fetch_url to read 6–10 best sources.\n"
    "4) Extract dated facts, numbers, short quotes with inline (Title, URL) after each bullet.\n"
    "\n"
    "CRITICAL: At the very end, output a section exactly named 'URLS:' and list each unique source URL on its own line.\n"
    "Example:\n"
    "URLS:\n"
    "https://example.com/a\n"
    "https://example.org/b\n"
    "\n"
    "When extraction is complete, if verification is needed, HANDOFF to the SourceChecker."
)

def build_fact_finder(handoffs=None) -> Agent:
    return Agent(
        name="FactFinder",
        handoff_description="Extracts grounded facts with citations from the web.",
        instructions=FACTFINDER_SYS,
        model=model_cheap(),
        tools=[web_search, fetch_url],
        handoffs=handoffs or [],
    )

# ----- Source Checker -----
SOURCECHECK_SYS = (
    "You verify claims against provided sources using citation_check.\n"
    "Return concise bullets: {claim, status: supported/partial/unsupported, best_citation, note}.\n"
    "If verification passes (or after flagging unsupported claims), HANDOFF to DataAnalyst."
)

def build_source_checker(handoffs=None) -> Agent:
    return Agent(
        name="SourceChecker",
        handoff_description="Verifies claims against cited sources.",
        instructions=SOURCECHECK_SYS,
        model=model_cheap(),
        tools=[citation_check],
        handoffs=handoffs or [],
    )

# ----- Analyst -----
ANALYST_SYS = (
    "You analyze verified nuggets to produce concise, cautious insights with micro-citations by bullet index."
)

def build_analyst(handoffs=None) -> Agent:
    return Agent(
        name="DataAnalyst",
        handoff_description="Converts verified nuggets into insights.",
        instructions=ANALYST_SYS,
        model=model_smart(),
        tools=[],
        handoffs=handoffs or [],
    )
