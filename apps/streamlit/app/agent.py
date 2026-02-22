"""
Agent configuration and helper functions.
Separates AI agent logic from the Streamlit UI.
"""

import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.db.postgres import PostgresDb
import settings
from tools.ensemble_tiktok_tool import search_tiktok_hashtag
from tools.open_alex_tool import search_openalex


@st.cache_resource
def get_database():
    """Create and cache the PostgreSQL database connection."""
    try:
        db = PostgresDb(db_url=settings.DATABASE_URL)
        return db
    except Exception:
        return None


def create_agent(
    enable_tiktok: bool, enable_openalex: bool, user_id: str, session_id: str
) -> Agent:
    """
    Create an Agno agent with configured tools and memory.

    Args:
        enable_tiktok: Whether to enable TikTok search tool
        enable_openalex: Whether to enable OpenAlex search tool
        user_id: User identifier for memory
        session_id: Session identifier for memory

    Returns:
        Configured Agent instance
    """
    tools = []
    if enable_tiktok:
        tools.append(search_tiktok_hashtag)
    if enable_openalex:
        tools.append(search_openalex)

    db = get_database()

    instructions = (
        "You are Beauty Social AI, a helpful assistant specialized in beauty trends "
        "and social media insights. Provide clear, friendly, and accurate responses."
    )

    if enable_tiktok:
        instructions += (
            "\n\nYou can search TikTok for beauty content using the search_tiktok_hashtag tool. "
            "Use it when users ask about TikTok videos or trending content."
        )

    if enable_openalex:
        instructions += (
            "\n\nYou can search for research papers using the search_openalex tool. "
            "Use it when users ask for academic papers, scientific research, or deeper technical details."
        )

    return Agent(
        name="beauty-social-ai",
        model=OpenAIResponses(id="gpt-5-mini"),
        instructions=instructions,
        tools=tools if tools else None,
        markdown=True,
        stream=True,
        db=db,
        user_id=user_id,
        session_id=session_id,
        update_memory_on_run=True,
        # Enable chat history persistence - loads previous messages from DB
        add_history_to_context=True,
        num_history_runs=10,  # Include last 10 conversation turns
    )


def get_chat_history(session_id: str, user_id: str) -> list[dict]:
    """
    Retrieve chat history for a specific session from the database.

    Args:
        session_id: The session ID to get history for
        user_id: The user ID

    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    db = get_database()
    if not db:
        return []

    try:
        # Create a temporary agent to access session methods
        agent = Agent(
            name="temp-agent",
            model=OpenAIResponses(id="gpt-5-mini"),
            db=db,
            user_id=user_id,
            session_id=session_id,
        )

        # Get chat history (user and assistant messages only)
        messages = agent.get_chat_history(session_id=session_id)

        if not messages:
            return []

        # Convert to simple dict format
        return [
            {"role": msg.role, "content": msg.content or ""}
            for msg in messages
            if msg.role in ("user", "assistant") and msg.content
        ]
    except Exception:
        return []


def get_all_sessions(user_id: str) -> list[dict]:
    """
    Retrieve all sessions for a user from the database.

    Args:
        user_id: The user ID to get sessions for

    Returns:
        List of session dicts with 'session_id', 'title', and 'message_count'
    """
    db = get_database()
    if not db:
        return []

    try:
        # Get sessions from database
        # Note: This may log errors for sessions created with different agent types
        sessions = db.get_sessions(user_id=user_id)

        if not sessions:
            return []

        result = []
        for session in sessions:
            try:
                # Get first message for title
                title = "New Chat"
                messages = session.runs[0].messages if session.runs else []
                for msg in messages:
                    if msg.role == "user" and msg.content:
                        title = (
                            msg.content[:30] + "..."
                            if len(msg.content) > 30
                            else msg.content
                        )
                        break

                # Count messages
                message_count = sum(len(run.messages) for run in (session.runs or []))

                result.append(
                    {
                        "session_id": session.session_id,
                        "title": title,
                        "message_count": message_count,
                    }
                )
            except Exception:
                # Skip sessions that can't be parsed (e.g., different agent type)
                continue

        return result
    except Exception:
        return []


def get_session_state_data(session_id: str, user_id: str) -> dict:
    """Session state data — kept in st.session_state only (no DB persistence needed)."""
    return {}


def update_session_state_data(
    session_id: str, user_id: str, state_updates: dict
) -> dict:
    """Session state data — kept in st.session_state only (no DB persistence needed)."""
    return {}
