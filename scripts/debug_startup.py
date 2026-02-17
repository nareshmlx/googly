
import os
import settings
from agno.db.postgres import PostgresDb
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
import time

print("--- DEBUG STARTUP ---")
print(f"DATABASE_URL: {settings.DATABASE_URL}")

def test_db_connection():
    print("1. Testing DB Connection...")
    start = time.time()
    try:
        db = PostgresDb(db_url=settings.DATABASE_URL)
        # Force connection
        engine = db.db_engine
        with engine.connect() as conn:
            print("   Connection successful!")
    except Exception as e:
        print(f"   Connection FAILED: {e}")
    print(f"   Time taken: {time.time() - start:.2f}s")
    return db

def test_agent_init(db):
    print("\n2. Testing Agent Initialization...")
    start = time.time()
    try:
        agent = Agent(
            name="temp-agent",
            model=OpenAIResponses(id="gpt-4o-mini"),
            db=db,
            user_id="debug_user",
            session_id="debug_session",
        )
        print("   Agent initialized.")
    except Exception as e:
        print(f"   Agent init FAILED: {e}")
    print(f"   Time taken: {time.time() - start:.2f}s")
    return agent

def test_get_history(agent):
    print("\n3. Testing get_chat_history...")
    start = time.time()
    try:
        history = agent.get_chat_history(session_id="debug_session")
        print(f"   History retrieved. Count: {len(history) if history else 0}")
    except Exception as e:
        print(f"   get_chat_history FAILED: {e}")
    print(f"   Time taken: {time.time() - start:.2f}s")

def test_openai_connection():
    print("\n4. Testing OpenAI Connection (via Agent)...")
    start = time.time()
    try:
        agent = Agent(model=OpenAIResponses(id="gpt-4o-mini"))
        response = agent.run("Hello, are you there?")
        print(f"   Response received: {response.content}")
    except Exception as e:
        print(f"   OpenAI Connection FAILED: {e}")
    print(f"   Time taken: {time.time() - start:.2f}s")


if __name__ == "__main__":
    db = test_db_connection()
    if db:
        agent = test_agent_init(db)
        if agent:
            test_get_history(agent)
    
    # Optional: Test OpenAI if strictly needed, but DB is likely culprit
    # test_openai_connection()
    
    print("\n--- DEBUG COMPLETE ---")
