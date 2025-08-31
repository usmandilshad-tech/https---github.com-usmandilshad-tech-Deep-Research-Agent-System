# research_agents.py
from agents import Agent
from sdk import model_cheap, model_smart
from tools import web_search, fetch_url, citation_check

FACTFINDER_SYS = (
    "You are a meticulous fact-finding researcher.\n"
    "Token budget is tight. Follow strictly:\n"
    "1) Propose up to 2 focused web search queries.\n"
    "2) For each, call web_search with k=3 and pick the best URLs.\n"
    "3) Use fetch_url on at most 3 sources total. If a page is very long, only extract 2–3 key facts.\n"
    "4) Output concise bullets (≤12), with short paraphrases and inline (Title, URL). Avoid long quotes.\n"
    "5) At the very end, print a section exactly named 'URLS:' with each unique URL on its own line.\n"
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

ANALYST_SYS = (
    "You analyze verified nuggets to produce concise, cautious insights. Keep to 8–12 bullets max."
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
