from agno.agent import Agent
from agno.models.openai import OpenAIChat
from tools.time_tool import get_current_time

executor_agent = Agent(
    name="executor-agent",
    instructions=(
        "You are an executor agent.\n"
        "Execute ONE step exactly as given.\n"
        "Use tools if required.\n"
        "Return only the result."
    ),
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[get_current_time],
    stream=True,
)
