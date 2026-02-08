from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ENSEMBLE_API_TOKEN = os.getenv("ENSEMBLE_API_TOKEN")

# PostgreSQL Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5532")
DB_USER = os.getenv("DB_USER", "ai")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ai")
DB_NAME = os.getenv("DB_NAME", "ai")

# Construct database URL for agno
DATABASE_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in environment")

# Ensemble API token is optional - only needed if using TikTok search tool
