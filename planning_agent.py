# planning_agent.py
from agents import Agent
from sdk import model_cheap

PLANNER_SYS = (
    "You break complex research questions into an efficient plan.\n"
    "- Produce 3–6 high-leverage, ordered tasks.\n"
    "- Each task should be specific and independently executable.\n"
    "- Keep it concise."
)

def build_planner() -> Agent:
    return Agent(
        name="Planner",
        instructions=PLANNER_SYS,
        model=model_cheap(),
        tools=[],  # pure LLM
    )
