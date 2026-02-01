from agno.agent import Agent
from agno.models.openai import OpenAIChat

planner_agent = Agent(
    name="planner-agent",
    instructions=(
        "You are a planner agent.\n"
        "Break the user request into ordered, minimal steps.\n"
        "Do NOT execute anything.\n"
        "Return only the steps, one per line."
    ),
    model=OpenAIChat(id="gpt-4o-mini"),
    stream=True,
)
