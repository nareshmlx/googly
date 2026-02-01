import settings
from agents.multi.planner import planner_agent
from agents.multi.executor import executor_agent


def collect_text(agent, prompt, session_id=None):
    response = agent.run(prompt, session_id=session_id)
    text = ""
    current_session_id = None

    for event in response:
        data = vars(event)

        if current_session_id is None:
            current_session_id = data.get("session_id")

        if data.get("content"):
            text += data["content"]

    return text.strip(), current_session_id


def main():
    user_request = "What is the current time and explain what timezone it is in?"

    print("\n🧠 USER REQUEST\n")
    print(user_request, "\n")

    # 1️⃣ Planner
    plan, session_id = collect_text(planner_agent, user_request)

    print("📋 PLAN\n")
    print(plan, "\n")

    # 2️⃣ Executor runs each step
    results = []

    for step in plan.split("\n"):
        if not step.strip():
            continue

        result, _ = collect_text(
            executor_agent,
            step,
            session_id=session_id
        )
        results.append(result)

    # 3️⃣ Combine (fake supervisor)
    print("✅ FINAL ANSWER\n")
    print("\n".join(results))


if __name__ == "__main__":
    main()
