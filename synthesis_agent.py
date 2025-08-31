from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



SYSTEM = (
    "You are a synthesis expert. Combine disjoint findings into a coherent, non-duplicative outline. "
    "Organize by themes, then sub-claims, then key evidence (with source names/URLs inline)."
)

def synthesize_outline(question: str, findings_markdown: str, model: str | None = None) -> str:
    resp = _client.responses.create(
        model=model or os.getenv("MODEL_SMART", "gpt-4o"),
        input=[{"role":"system","content":SYSTEM},
               {"role":"user","content":f"QUESTION:\n{question}\n\nFINDINGS:\n{findings_markdown}\n\nCreate an outline."}],
        temperature=0.2
    )
    return resp.output_text
