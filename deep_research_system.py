from __future__ import annotations
import os, sys, datetime as dt
from dotenv import load_dotenv
from rich import print
from rich.panel import Panel

# --- OpenAI Agents SDK (async client + tracing) ---
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    function_tool,
    trace,
    set_default_openai_client,
    OpenAIResponsesModel,  # model wrapper for Responses API
)

# --- your modules ---
from planning_agent import make_plan
from research_agents import (
    FactFinderAgent,
    SourceCheckerAgent,
    DataAnalystAgent,
    SearchTool,
    DocLoaderTool,
    CitationCheckerTool,
)
from synthesis_agent import synthesize_outline
from report_writer import render_markdown

load_dotenv()

# register a SINGLE async OpenAI client for the Agents SDK
external_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
set_default_openai_client(external_client)

# ---- simple injectable tools (replace with real implementations) ----
class DummySearch(SearchTool):
    def run(self, query: str, k: int = 8):
        return []

class DummyLoader(DocLoaderTool):
    def fetch(self, url: str) -> str:
        return ""

class DummyChecker(CitationCheckerTool):
    pass

# ---- your existing pipeline kept synchronous under the hood ----------
def run_research(question: str) -> str:
    print(Panel.fit(f"[bold]Deep Research[/bold]\n{question}", title="Lead Researcher"))

    # 1) plan
    plan = make_plan(question)
    print(Panel.fit("\n".join(plan.high_level), title="High-level Plan"))
    print(Panel.fit("\n".join(plan.tasks), title="Task List"))

    # 2) research agents
    fact_finder = FactFinderAgent(search=DummySearch(), loader=DummyLoader())
    checker = SourceCheckerAgent(checker=DummyChecker())
    analyst = DataAnalystAgent()

    nuggets, all_urls = [], []

    for i, task in enumerate(plan.tasks, start=1):
        print(Panel.fit(task, title=f"Task {i}"))
        pack = fact_finder.research(task, k=6)
        nuggets.append(pack["extracted"])
        urls = [r.get("url", "") for r in pack["results"] if r.get("url")]
        all_urls.extend(urls)

    # 3) verify & analyze
    verified = checker.verify("\n\n".join(nuggets), list(dict.fromkeys(all_urls)))
    analysis = analyst.analyze(verified)

    # 4) synthesize outline
    outline = synthesize_outline(question, verified)

    # 5) write report
    md = render_markdown({
        "title": "Deep Research Report",
        "date": dt.date.today().isoformat(),
        "author": "Deep Research Agent",
        "window": "Focus: 2022–2025 (prioritized newest credible sources)",
        "executive_summary": "TODO: Add 5–8 bullet executive summary from outline+analysis.",
        "key_findings": outline,
        "analysis": analysis,
        "limitations": "- Some sources are placeholders until real web/search tools are connected.\n- Citations should be re-checked after integrating live fetch.",
        "sources": list(dict.fromkeys(all_urls)) or ["(No links collected — implement SearchTool/DocLoader)"],
    })
    return md

# ---- expose your pipeline as a tool for the Agent to call ----------
@function_tool()
def run_pipeline(question: str) -> str:
    """
    Tool: executes the full deep research pipeline and returns a Markdown report.
    """
    return run_research(question)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[red]Usage:[/red] uv run python deep_research_system.py \"your question here\"")
        sys.exit(1)

    question = sys.argv[1]

    # Construct a model wrapper for the coordinator agent
    llm_model = OpenAIResponsesModel(
        model=os.getenv("MODEL_SMART", "gpt-4o"),
        openai_client=external_client,  # async client required by Agents SDK loop
    )

    # Coordinator agent that simply calls the tool with the user question
    lead = Agent(
        name="Lead Researcher",
        instructions=(
            "You coordinate deep research. When given a question, call the "
            "`run_pipeline` tool with the full question to produce a complete report."
        ),
        model=llm_model,
        tools=[run_pipeline],
    )

    # ✅ tracing: the whole run will appear in OpenAI → Traces
    with trace(workflow_name="Deep Research Run", metadata={"question": question, "app": "DSAS"}):
        result = Runner.run_sync(
            starting_agent=lead,
            input=question
        )

    # The tool returns Markdown text as the final output
    report_md = result.output_text

    out = "report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(Panel.fit(f"Report written to [bold]{out}[/bold]", title="Done"))
