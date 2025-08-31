# guardrails.py
import os
import re
from pydantic import BaseModel
from agents import (
    Agent, Runner,
    InputGuardrail, OutputGuardrail, GuardrailFunctionOutput
)

# =========================================================
# Config
# =========================================================
STRICT = os.getenv("GUARDRAILS_STRICT", "0").lower() in ("1", "true", "yes")

# =========================================================
# INPUT GUARDRail (LLM-based): is this a legitimate research query?
# =========================================================
class ResearchGate(BaseModel):
    allowed: bool
    reason: str

research_gate_agent = Agent(
    name="Research Input Gate",
    instructions=(
        "Decide if the user input is an appropriate research question. "
        "BLOCK if it requests illegal/harmful actions (hacking, malware), adult content, explicit PII extraction, "
        "medical/legal advice without disclaimers, or pure homework cheating. "
        "Return JSON {allowed: bool, reason}."
    ),
    output_type=ResearchGate,
)

async def research_input_guardrail(ctx, agent, input_data):
    res = await Runner.run(research_gate_agent, input_data, context=ctx.context)
    out = res.final_output_as(ResearchGate)
    # Tripwire when NOT allowed (always strict for input)
    return GuardrailFunctionOutput(output_info=out, tripwire_triggered=not out.allowed)

input_guardrail = InputGuardrail(guardrail_function=research_input_guardrail)

# =========================================================
# OUTPUT GUARDRail (Deterministic): minimal report quality
# Checks for exact Markdown sections and at least one http(s) link in Sources.
# =========================================================
_HEADING_RE = re.compile(r"^\s*##\s*(?P<h>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_HTTP_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)

def _find_section(text: str, title: str) -> str | None:
    """
    Return the section body for the H2 heading '## {title}', up to (but not including) the next H2.
    """
    if not text:
        return None
    # Find all H2 headings and their positions
    matches = list(_HEADING_RE.finditer(text))
    # Locate the requested title (case-insensitive exact match)
    start_idx = None
    end_idx = None
    for i, m in enumerate(matches):
        if m.group("h").strip().lower() == title.strip().lower():
            start_idx = m.end()
            # Next H2 (or end of text)
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            break
    if start_idx is None:
        return None
    return text[start_idx:end_idx]

def _bool_section(text: str, title: str) -> bool:
    return _find_section(text, title) is not None

def _has_http_in_sources(text: str) -> bool:
    block = _find_section(text, "Sources")
    if not block:
        return False
    return _HTTP_RE.search(block) is not None

async def report_output_guardrail(ctx, agent, output_data):
    # output_data is the agent's final text (Markdown)
    text = output_data if isinstance(output_data, str) else str(output_data)

    has_exec = _bool_section(text, "Executive Summary")
    has_kf   = _bool_section(text, "Key Findings")
    has_srcs = _has_http_in_sources(text)

    ok = has_exec and has_kf and has_srcs

    info = {
        "has_exec_summary": has_exec,
        "has_key_findings": has_kf,
        "has_sources_with_http": has_srcs,
        "note": "Deterministic QA: requires exact H2 headings and at least one http(s) link under '## Sources'."
    }

    # In dev, only WARN if not ok; in strict mode, BLOCK.
    trip = (not ok) and STRICT
    return GuardrailFunctionOutput(output_info=info, tripwire_triggered=False)

output_guardrail = OutputGuardrail(guardrail_function=report_output_guardrail)
