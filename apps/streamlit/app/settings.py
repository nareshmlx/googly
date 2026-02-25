import warnings
from pathlib import Path
from dotenv import load_dotenv
import os

# Load from repo root .env — two levels up: apps/streamlit/app/ -> apps/streamlit/ -> apps/ -> repo root
_env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ENSEMBLE_API_TOKEN = os.getenv("ENSEMBLE_API_TOKEN")

# FastAPI backend base URL — used for project CRUD, SSE chat, KB upload/status
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8003")
FASTAPI_CREATE_PROJECT_TIMEOUT = float(os.getenv("FASTAPI_CREATE_PROJECT_TIMEOUT", "60"))
FASTAPI_DISCOVER_TIMEOUT = float(os.getenv("FASTAPI_DISCOVER_TIMEOUT", "20"))
FASTAPI_KB_STATUS_TIMEOUT = float(os.getenv("FASTAPI_KB_STATUS_TIMEOUT", "20"))

# OpenAlex polite-pool email (required by OpenAlex Terms of Service)
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "user@example.com")

# PostgreSQL Database Configuration (matches docker-compose defaults)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "googly")
DB_PASSWORD = os.getenv("DB_PASSWORD", "googly")
DB_NAME = os.getenv("DB_NAME", "googly")

# Construct database URL for agno
DATABASE_URL = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

if not OPENAI_API_KEY:
    warnings.warn(
        "OPENAI_API_KEY not found in environment — Agent Mode will be unavailable. "
        "Project Mode (FastAPI backend) will still work.",
        stacklevel=1,
    )

# GOOGLE_API_KEY and ENSEMBLE_API_TOKEN are optional
