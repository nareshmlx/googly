import settings
from agents.multi.supervisor import supervisor_agent

def main():
    print("\n🧠 HANDOFF-BASED MULTI-AGENT RUN\n")

    response = supervisor_agent.run(
        "What is the current time and explain what timezone it is in?"
    )

    for event in response:
        data = vars(event)

        # Show which agent is currently speaking (optional but useful)
        if data.get("agent_name"):
            print(f"\n[{data['agent_name']}]", end=" ")

        if data.get("content"):
            print(data["content"], end="", flush=True)

    print("\n\n✅ Done")

if __name__ == "__main__":
    main()
