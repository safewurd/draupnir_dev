# scripts/test_neon_connection.py
from sqlalchemy import text
from draupnir_core.db_config import get_engine

engine = get_engine()
with engine.connect() as conn:
    print("Connected ✅")
    tables = conn.execute(text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema='public' 
        ORDER BY table_name
    """)).fetchall()
    print("Tables:", [t[0] for t in tables])
