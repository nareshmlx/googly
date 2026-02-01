from agno.agent import Agent
from agno.models.openai import OpenAIChat
from tools.time_tool import get_current_time

basic_agent = Agent(
    name="basic-agent",
    instructions=(
        "You are a helpful AI tutor. "
        "Explain concepts clearly and step by step."
    ),
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[get_current_time],
    markdown=True,
    stream=True,
)
