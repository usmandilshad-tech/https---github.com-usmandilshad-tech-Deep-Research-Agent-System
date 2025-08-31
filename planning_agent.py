from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Plan(BaseModel):
    high_level: list[str]
    tasks: list[str]

SYSTEM = (
    "You break complex research questions into an efficient plan. "
    "Prefer a small number of high-leverage tasks. Be specific and ordered."
)

def make_plan(question: str, model: str | None = None) -> Plan:
    resp = _client.responses.create(
        model=model or os.getenv("MODEL_CHEAP", "gpt-4o-mini"),
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Create a plan for:\n{question}"}
        ],
        temperature=0.2
    )
    text = resp.output_text
    lines = [l.strip("-• ").strip() for l in text.splitlines() if l.strip()]
    mid = min(len(lines), max(2, len(lines)//3))
    return Plan(high_level=lines[:mid], tasks=lines[mid:])
