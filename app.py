import json
import settings
from agents.basic_agent import basic_agent

def main():
    print("\n🧠 Agent started. Streaming output:\n")

    response = basic_agent.run(
        "What is the current time? Explain briefly."
    )

    for event in response:
        event_dict = vars(event)

        event_type = event_dict.get("event")

        # 1️⃣ Tool call event
        if event_type == "RunToolCall":
            print(f"\n🛠️ Calling tool: {event_dict.get('tools')}\n")

        # 2️⃣ Normal content tokens
        if event_dict.get("content"):
            print(event_dict["content"], end="", flush=True)

    print("\n\n✅ Agent run complete")

if __name__ == "__main__":
    main()
