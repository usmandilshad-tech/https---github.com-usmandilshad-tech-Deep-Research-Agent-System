# synthesis_agent.py
from agents import Agent
from sdk import model_smart

SYNTH_SYS = (
    "You are a synthesis expert.\n"
    "Combine verified findings into a coherent outline:\n"
    "Themes → sub-claims → key evidence (Source Name, URL)."
)

def build_synthesizer() -> Agent:
    return Agent(
        name="Synthesis",
        instructions=SYNTH_SYS,
        model=model_smart(),
        tools=[],
    )
