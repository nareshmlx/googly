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


@st.cache_resource
def get_database():
    """Create and cache the PostgreSQL database connection."""
    try:
        db = PostgresDb(db_url=settings.DATABASE_URL)
        return db
    except Exception:
        return None


def create_agent(enable_tiktok: bool, user_id: str, session_id: str) -> Agent:
    """
    Create an Agno agent with configured tools and memory.
    
    Args:
        enable_tiktok: Whether to enable TikTok search tool
        user_id: User identifier for memory
        session_id: Session identifier for memory
    
    Returns:
        Configured Agent instance
    """
    tools = []
    if enable_tiktok:
        tools.append(search_tiktok_hashtag)
    
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
    )
