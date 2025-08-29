# draupnir_core/db_config.py
import os
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from sqlalchemy import create_engine

# Load .env if available (harmless if missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _raw_url() -> str:
    # Accept multiple env var names for portability across machines
    return (
        os.getenv("NEON_DB_URL")
        or os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or ""
    )


def _normalize_sqlalchemy_url(raw: str) -> str:
    """
    Ensure the URL works with SQLAlchemy + psycopg v3.
    - Convert 'postgresql://' -> 'postgresql+psycopg://'
    - Ensure sslmode=require if not specified
    """
    if not raw:
        return raw
    parsed = urlparse(raw)
    scheme = parsed.scheme
    if scheme == "postgresql":
        scheme = "postgresql+psycopg"
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if "sslmode" not in q:
        q["sslmode"] = "require"
    rebuilt = urlunparse(
        (scheme, parsed.netloc, parsed.path, parsed.params, urlencode(q), parsed.fragment)
    )
    return rebuilt


def get_db_label() -> str:
    """
    Return a human-friendly label for the current DB WITHOUT secrets.
    Example: 'Postgres @ ep-shy-bread-ae7eg150-pooler.c-2.us-east-2.aws.neon.tech/draupnir as neondb_owner'
    """
    raw = _raw_url()
    if not raw:
        return "(no DB configured)"
    p = urlparse(raw)
    vendor = "Postgres" if p.scheme.startswith("postgres") else p.scheme or "DB"
    host = p.hostname or "?"
    db = (p.path or "").lstrip("/") or "?"
    user = p.username or "?"
    return f"{vendor} @ {host}/{db} as {user}"


def get_engine():
    """
    Create a SQLAlchemy engine for the configured Postgres URL.
    """
    url = _raw_url()
    if not url:
        raise RuntimeError("NEON_DB_URL or DATABASE_URL (or POSTGRES_URL) must be set.")
    sa_url = _normalize_sqlalchemy_url(url)
    return create_engine(sa_url, pool_pre_ping=True, pool_recycle=1800)
