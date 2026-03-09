"""
Agent configuration and helper functions.
Separates AI agent logic from the Streamlit UI.
"""

import streamlit as st
from agno.db.postgres import PostgresDb
import settings


@st.cache_resource
def get_database():
    """Create and cache the PostgreSQL database connection."""
    try:
        db = PostgresDb(db_url=settings.DATABASE_URL)
        return db
    except Exception:
        return None
