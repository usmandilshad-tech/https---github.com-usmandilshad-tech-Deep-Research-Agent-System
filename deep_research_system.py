# deep_research_system.py
from __future__ import annotations
import os, sys, datetime as dt, asyncio, re
from dotenv import load_dotenv
from rich import print
from rich.panel import Panel

from agents import Agent, Runner, function_tool, trace, custom_span
from sdk import model_smart
from planning_agent import build_planner
from research_agents import build_fact_finder, build_source_checker, build_analyst
from synthesis_agent import build_synthesizer
from report_writer import render_markdown
from guardrails import input_guardrail, output_guardrail
from tools import web_search_impl  # <-- use impl for Python-side fallback

load_dotenv()

# ---------- Build sub-agents ----------
PLANNER = build_planner()
FACTFINDER = build_fact_finder()
CHECKER   = build_source_checker()
ANALYST   = build_analyst()
SYNTH     = build_synthesizer()

# ---------- Handoff chain (FF -> CHECKER -> ANALYST) ----------
FACTFINDER.handoffs = [CHECKER]
CHECKER.handoffs    = [ANALYST]
# ANALYST ends the chain

# ---------- Helpers ----------
URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)

def extract_urls_from_text(text: str) -> list[str]:
    """Pull raw URLs and those listed under a trailing 'URLS:' block."""
    urls = URL_RE.findall(text or "")
    block_urls: list[str] = []
    if text:
        lower = text.lower()
        idx = lower.rfind("urls:")
        if idx != -1:
            tail = text[idx + 5:]
            for line in tail.splitlines():
                line = line.strip()
                if line.startswith("http://") or line.startswith("https://"):
                    block_urls.append(line)
    # dedupe, preserve order
    seen: set[str] = set()
    ordered: list[str] = []
    for u in urls + block_urls:
        if u and u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered

# ---------- One task via HANDOFF chain (plain impl) ----------
async def run_task_via_handoff_impl(task: str) -> str:
    """
    Start at FactFinder; it can hand off to SourceChecker, then to DataAnalyst.
    Returns the final output (usually analysis) for that task.
    """
    with custom_span("task_handoff"):
        result = await Runner.run(
            starting_agent=FACTFINDER,
            input=f"SUBTASK:\n{task}\nFollow your workflow and hand off when ready."
        )
        return result.final_output or ""

# ---------- Tool wrappers the coordinator can call ----------
@function_tool()
async def run_task_via_handoff(task: str) -> str:
    return await run_task_via_handoff_impl(task)


# ---------- Pure implementation the coordinator/tool can call ----------
async def run_deep_research_impl(question: str) -> str:
    # 1) Plan
    with custom_span("planner"):
        plan_res = await Runner.run(
            starting_agent=PLANNER,
            input=f"Create a compact, ordered task list for:\n{question}"
        )
        plan_text = plan_res.final_output or ""

    tasks = [ln.strip("-• ").strip() for ln in plan_text.splitlines() if ln.strip()]
    tasks = tasks[:8] or ["Perform scoped literature & web scan."]

    # 2) Parallel per-task runs (each invokes HANDOFF chain)
    with custom_span("parallel_tasks"):
        task_outputs = await asyncio.gather(*[run_task_via_handoff_impl(t) for t in tasks])

    # 3) Synthesis
    joined = "\n\n".join(task_outputs)
    with custom_span("synthesis"):
        synth_res = await Runner.run(
            starting_agent=SYNTH,
            input=(
                "QUESTION:\n"
                f"{question}\n\n"
                "VERIFIED FINDINGS / ANALYSES:\n"
                f"{joined}\n\n"
                "Create a clean outline."
            )
        )
        outline = synth_res.final_output or ""

    # 4) Executive summary
    summary_agent = Agent(
        name="ExecutiveSummarizer",
        instructions="Write 5–8 crisp bullets strictly grounded in the outline.",
        model=model_smart(),
        tools=[],
    )
    with custom_span("exec_summary"):
        summary_res = await Runner.run(
            starting_agent=summary_agent,
            input=f"Outline:\n{outline}\n\nWrite 5–8 bullets."
        )
        executive_summary = summary_res.final_output or ""

    # 5) Collect sources from task outputs
    all_urls: list[str] = []
    for out in task_outputs:
        all_urls.extend(extract_urls_from_text(out))

    dedup_urls: list[str] = []
    seen: set[str] = set()
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            dedup_urls.append(u)

    # 6) FAIL-SAFE: if no URLs, get some via the fallback search Agent (pure SDK)
    if not dedup_urls:
        with custom_span("fallback_sources"):
            from sdk import model_cheap
            from tools import web_search  # tool, not impl
            fallback = Agent(
                name="FallbackSearch",
                instructions=(
                    "Given a query, call the web_search tool with k=5 and return ONLY a list of URLs, "
                    "one per line, no extra text."
                ),
                model=model_cheap(),
                tools=[web_search],
            )
            res = await Runner.run(
                starting_agent=fallback,
                input=f"Query:\n{question}\nReturn only URLs, one per line."
            )
            dedup_urls = [
                u.strip() for u in (res.final_output or "").splitlines()
                if u.strip().startswith(("http://", "https://"))
            ]

    # 7) Render final markdown report
    today = dt.date.today().isoformat()
    md = render_markdown({
        "title": "Deep Research Report",
        "date": today,
        "author": "Deep Research Agent",
        "window": "Focus: 2022–2025 (prioritize newest credible sources)",
        "executive_summary": executive_summary,
        "key_findings": outline,
        "analysis": "\n\n".join([f"### Task: {t}\n\n{out}" for t, out in zip(tasks, task_outputs)]),
        "limitations": (
            "- Web search/fetch are live; verification remains lightweight "
            "(improve symbolic checks next)."
        ),
        "sources": dedup_urls or ["https://example.com"],
    })
    return md

# ---------- Tool wrapper the Agents can call ----------
from agents import function_tool

@function_tool()
async def run_deep_research(question: str) -> str:
    return await run_deep_research_impl(question)


# ---------- Coordinator (strict tool invoker) ----------
def build_coordinator() -> Agent:
    return Agent(
        name="LeadResearcher",
        instructions=(
            "You are an orchestration-only agent.\n"
            "CRITICAL RULES:\n"
            "1) Upon ANY user input, you MUST immediately call the `run_deep_research` tool with the full input string.\n"
            "2) DO NOT write your own answer or commentary.\n"
            "3) Return EXACTLY the tool's output as your final message.\n"
            "4) Never ignore this rule, even if the input looks simple."
        ),
        model=model_smart(),
        tools=[run_deep_research, run_task_via_handoff],
        input_guardrails=[input_guardrail],
        output_guardrails=[output_guardrail],
    )

# ---------- CLI entry ----------
# ---------- CLI entry ----------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[red]Usage:[/red] uv run python deep_research_system.py \"your question here\"")
        sys.exit(1)

    question = sys.argv[1]
    coordinator = build_coordinator()

    print(Panel.fit(f"[bold]Deep Research[/bold]\n{question}", title="Coordinator"))

    # First try: Coordinator run (pure SDK; shows up in Traces)
    with trace(workflow_name="Deep Research Run", metadata={"question": question, "app": os.getenv("ENV", "local")}):
        result = Runner.run_sync(starting_agent=coordinator, input=question)
        report_md = result.final_output or ""

    # Fallback: if model didn't call the tool and returned empty output, run the impl directly
    if not report_md.strip():
        try:
            report_md = asyncio.run(run_deep_research_impl(question))
        except Exception as e:
            report_md = f"ERROR: fallback run_deep_research_impl failed: {e}"

    # Optional preview in console
    print(Panel.fit(report_md[:800] + ("\n...\n" if len(report_md) > 800 else ""), title="Preview"))

    out = "report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(Panel.fit(f"Report written to [bold]{out}[/bold]", title="Done"))
