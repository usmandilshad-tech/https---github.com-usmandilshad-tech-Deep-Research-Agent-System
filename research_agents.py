from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- simple tool interfaces (implement later) -----------------
class SearchTool:
    """Hook for web search. Replace `run` body with Tavily/SerpAPI/etc."""
    def run(self, query: str, k: int = 8) -> List[Dict[str, str]]:
        # return [{"title": "...","url":"...","snippet":"..."}]
        return []

class DocLoaderTool:
    """Hook to fetch & read URLs/PDFs."""
    def fetch(self, url: str) -> str:
        return ""

class CitationCheckerTool:
    """Hook to verify claims vs sources and produce citations."""
    def check(self, text: str, sources: List[str]) -> Dict[str, Any]:
        return {"verdict": "uncertain", "notes": []}

# ---- shared client ------------------------------------------------
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ---- routing logic (cost-aware) -----------------------------------
def pick_model(task: str, cheap: str, smart: str, reasoning: str) -> str:
    # super simple heuristic (tune later)
    n = len(task)
    if "calculate" in task.lower() or n > 3000:
        return reasoning
    if n < 800 and ("extract" in task.lower() or "search" in task.lower()):
        return cheap
    return smart

# ---- BaseAgent ----------------------------------------------------
class AgentConfig(BaseModel):
    name: str
    system_prompt: str
    default_model: str

class BaseAgent:
    def __init__(self, config: AgentConfig):
        self.config = config

    def call_llm(self, prompt: str, model_override: Optional[str] = None) -> str:
        model = model_override or self.config.default_model
        resp = _client.responses.create(
            model=model,
            input=[{
                "role": "system",
                "content": self.config.system_prompt
            },{
                "role": "user",
                "content": prompt
            }],
            temperature=0.2,
        )
        # Responses API returns output in a list; grab the text
        return resp.output_text

# ---- Specialized Agents -------------------------------------------
class FactFinderAgent(BaseAgent):
    def __init__(self, search: SearchTool, loader: DocLoaderTool):
        super().__init__(AgentConfig(
            name="FactFinder",
            system_prompt=(
                "You are a meticulous fact-finding researcher. "
                "Generate focused search queries, scan results, extract key facts "
                "with dates, figures, and direct citations (URL+title). Always prefer recent, credible sources."
            ),
            default_model=os.getenv("MODEL_CHEAP", "gpt-4o-mini"),
        ))
        self.search = search
        self.loader = loader

    def research(self, task: str, k: int = 8) -> Dict[str, Any]:
        model = pick_model(task,
                           os.getenv("MODEL_CHEAP", "gpt-4o-mini"),
                           os.getenv("MODEL_SMART", "gpt-4o"),
                           os.getenv("MODEL_REASONING", "o4-mini"))

        # ask LLM for search queries
        q = self.call_llm(
            f"Create 3-5 concise web search queries for this task:\n{task}",
            model_override=model
        )
        queries = [s.strip("-• ").strip() for s in q.splitlines() if s.strip()][:5]

        results, notes = [], []
        for query in queries:
            hits = self.search.run(query, k=k)
            results.extend(hits[:k])

        # optionally fetch bodies (be careful with quotas)
        bodies = []
        for r in results[:10]:
            try:
                bodies.append({"url": r.get("url",""), "content": self.loader.fetch(r.get("url",""))})
            except Exception:
                pass

        # synthesize extracted facts from snippets + bodies
        synthesis = self.call_llm(
            "From the following search results and article bodies, extract dated facts, numbers, and short quotes. "
            "Return a bullet list with (fact) — (source title, URL):\n\n"
            f"SEARCH_RESULTS:\n{results}\n\nBODIES:\n{bodies[:5]}",
            model_override=model
        )

        return {
            "queries": queries,
            "results": results,
            "extracted": synthesis
        }

class SourceCheckerAgent(BaseAgent):
    def __init__(self, checker: CitationCheckerTool):
        super().__init__(AgentConfig(
            name="SourceChecker",
            system_prompt=(
                "You verify claims against provided sources. Flag unsupported or ambiguous claims. "
                "Return JSON-like bullets: {claim, status, citation?, note}."
            ),
            default_model=os.getenv("MODEL_CHEAP", "gpt-4o-mini"),
        ))
        self.checker = checker

    def verify(self, claims_markdown: str, urls: List[str]) -> str:
        # (Optionally call external symbolic checker, then ask LLM to format nicely)
        raw = self.checker.check(claims_markdown, urls)
        prompt = (
            "Format the following verification results into concise markdown.\n"
            f"INPUT_CLAIMS:\n{claims_markdown}\n\nRAW_CHECK:\n{raw}"
        )
        return self.call_llm(prompt)

class DataAnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentConfig(
            name="DataAnalyst",
            system_prompt=(
                "You analyze structured nuggets (facts with numbers/dates) and produce short insights, comparisons, and trends. "
                "Be conservative and cite which nuggets support each insight."
            ),
            default_model=os.getenv("MODEL_SMART", "gpt-4o"),
        ))

    def analyze(self, nuggets_markdown: str) -> str:
        return self.call_llm(
            "Turn the following research nuggets into concise insights with micro-citations in brackets:\n"
            f"{nuggets_markdown}"
        )
