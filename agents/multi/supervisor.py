from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agents.multi.planner import planner_agent
from agents.multi.executor import executor_agent

supervisor_agent = Agent(
    name="supervisor-agent",
    instructions=(
        "You are a supervisor agent.\n"
        "You receive the user request.\n"
        "1) Hand off to the planner agent to create steps.\n"
        "2) For each step, hand off to the executor agent.\n"
        "3) Collect results and respond to the user.\n"
        "Use handoffs to delegate. Do not execute tools yourself."
    ),
    model=OpenAIChat(id="gpt-4o-mini"),
    # 👇 THIS is the key difference
    handoffs=[planner_agent, executor_agent],
    stream=True,
)
