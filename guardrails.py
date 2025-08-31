# guardrails.py
from pydantic import BaseModel
from agents import (
    Agent, Runner,
    InputGuardrail, OutputGuardrail, GuardrailFunctionOutput
)

# -------- Input guardrail: is this a legitimate research query? --------
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
    # tripwire when NOT allowed
    return GuardrailFunctionOutput(output_info=out, tripwire_triggered=not out.allowed)

input_guardrail = InputGuardrail(guardrail_function=research_input_guardrail)

# -------- Output guardrail: does the final report meet minimal quality? --------
class ReportQA(BaseModel):
    has_sources: bool
    has_exec_summary: bool
    has_key_findings: bool
    reason: str

report_qa_agent = Agent(
    name="Report QA",
    instructions=(
        "Given a Markdown research report, check:\n"
        "1) A 'Sources' section with at least 1 URL (http/https)\n"
        "2) An 'Executive Summary' section\n"
        "3) A 'Key Findings' section\n"
        "Return JSON {has_sources, has_exec_summary, has_key_findings, reason}."
    ),
    output_type=ReportQA,
)

async def report_output_guardrail(ctx, agent, output_data):
    res = await Runner.run(report_qa_agent, output_data, context=ctx.context)
    out = res.final_output_as(ReportQA)
    ok = out.has_sources and out.has_exec_summary and out.has_key_findings
    # tripwire when the report is missing key sections
    return GuardrailFunctionOutput(output_info=out, tripwire_triggered=not ok)

output_guardrail = OutputGuardrail(guardrail_function=report_output_guardrail)
