# draupnir_core/db_config.py
import os
from sqlalchemy import create_engine

# Uses NEON_DB_URL, e.g.:
# postgresql+psycopg://user:pass@host/draupnir?sslmode=require
DB_URL = os.getenv("NEON_DB_URL")

def get_engine():
    if not DB_URL:
        raise RuntimeError("NEON_DB_URL is not set in environment variables.")
    return create_engine(DB_URL, pool_pre_ping=True)
