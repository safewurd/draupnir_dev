# draupnir_core/db_config.py
import os
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from typing import Optional, Tuple

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Load .env if available (harmless if missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------------
# Module state (for UI switching)
# -------------------------
_ACTIVE_ENV_OVERRIDE: Optional[str] = None      # "development" | "production" | None
_SINGLE_URL_OVERRIDE: Optional[str] = None      # full URL takes precedence if set
_ENGINE: Optional[Engine] = None
_ENGINE_KEY: Optional[str] = None               # normalized URL string for cache key


# -------------------------
# Env helpers
# -------------------------
def _active_env() -> str:
    """
    Returns the logical environment: 'production' or 'development'.
    Order of precedence:
      1) In-process override (set_active_env)
      2) DB_ENV or APP_ENV from environment
      3) default 'development'
    """
    if _ACTIVE_ENV_OVERRIDE:
        env = _ACTIVE_ENV_OVERRIDE
    else:
        env = (os.getenv("DB_ENV") or os.getenv("APP_ENV") or "development")
    env = env.strip().lower()
    return "production" if env in ("prod", "production") else "development"


def get_active_env() -> str:
    """Public getter for the current logical environment."""
    return _active_env()


def set_active_env(env: str) -> None:
    """
    Set the active environment for THIS Streamlit session/process.
    Triggers an engine reset so the next query uses the new DB.
    """
    global _ACTIVE_ENV_OVERRIDE
    env = (env or "").strip().lower()
    _ACTIVE_ENV_OVERRIDE = "production" if env in ("prod", "production") else "development"
    _reset_engine()


def set_single_url(url: Optional[str]) -> None:
    """
    Explicitly set a single connection URL (overrides env selection).
    Pass None or '' to clear and go back to env-based selection.
    """
    global _SINGLE_URL_OVERRIDE
    _SINGLE_URL_OVERRIDE = (url or "").strip() or None
    _reset_engine()


def _raw_single_url() -> str:
    """
    Highest precedence explicit URL (either in-process override or env var).
    """
    if _SINGLE_URL_OVERRIDE:
        return _SINGLE_URL_OVERRIDE
    return (
        os.getenv("NEON_DB_URL")
        or os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or ""
    )


def _url_for_env(env: str) -> str:
    """
    Pick URL based on requested env ('production' or 'development').
    Accepts common *_PROD / *_DEV variable names.
    """
    keys = (
        ("NEON_DB_URL_PROD", "DATABASE_URL_PROD", "POSTGRES_URL_PROD")
        if env == "production"
        else ("NEON_DB_URL_DEV", "DATABASE_URL_DEV", "POSTGRES_URL_DEV")
    )
    for k in keys:
        v = os.getenv(k)
        if v:
            return v
    return ""


def _pick_url_from_env() -> str:
    """
    Selection order:
      1) Explicit single URL (_SINGLE_URL_OVERRIDE or NEON_DB_URL/DATABASE_URL/POSTGRES_URL)
      2) By logical env: *_PROD or *_DEV selected via set_active_env / DB_ENV / APP_ENV
      3) Nothing -> ''
    """
    single = _raw_single_url()
    if single:
        return single
    return _url_for_env(_active_env())


# -------------------------
# URL normalization & engine cache
# -------------------------
def _normalize_sqlalchemy_url(raw: str) -> str:
    """
    Normalize for SQLAlchemy + psycopg v3:
      - 'postgresql://' -> 'postgresql+psycopg://'
      - ensure sslmode=require if none present
      - preserve other query params (e.g., channel_binding=require)
    """
    if not raw:
        return raw
    parsed = urlparse(raw)
    scheme = parsed.scheme
    if scheme == "postgresql":  # map to psycopg v3 driver
        scheme = "postgresql+psycopg"
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    q.setdefault("sslmode", "require")
    return urlunparse((scheme, parsed.netloc, parsed.path, parsed.params, urlencode(q), parsed.fragment))


def _engine_key_from_url(url: str) -> str:
    return _normalize_sqlalchemy_url(url)


def _reset_engine() -> None:
    global _ENGINE, _ENGINE_KEY
    try:
        if _ENGINE is not None:
            _ENGINE.dispose()
    finally:
        _ENGINE = None
        _ENGINE_KEY = None


def dispose_engine() -> None:
    """Public method to dispose the current engine (usually not needed; set_active_env calls this)."""
    _reset_engine()


# -------------------------
# Public API
# -------------------------
def get_db_label() -> str:
    """
    Return a human-friendly label WITHOUT secrets.
    Example: 'Postgres @ ep-xxxx.neon.tech/draupnir as neondb_owner [development]'
    """
    raw = _pick_url_from_env()
    if not raw:
        return "(no DB configured)"
    p = urlparse(raw)
    vendor = "Postgres" if (p.scheme or "").startswith("postgres") else (p.scheme or "DB")
    host = p.hostname or "?"
    db = (p.path or "").lstrip("/") or "?"
    user = p.username or "?"
    return f"{vendor} @ {host}/{db} as {user} [{_active_env()}]"


def get_engine() -> Engine:
    """
    Create or return a cached SQLAlchemy engine for the currently selected DB.
    If the selected URL changes (e.g., UI switch), the engine is rebuilt automatically.
    """
    global _ENGINE, _ENGINE_KEY
    url = _pick_url_from_env()
    if not url:
        raise RuntimeError(
            "No Postgres URL found. Set NEON_DB_URL (or DATABASE_URL/POSTGRES_URL), "
            "or define DB_ENV and the corresponding *_DEV/*_PROD variables in .env."
        )

    key = _engine_key_from_url(url)
    if _ENGINE is None or _ENGINE_KEY != key:
        _ENGINE = create_engine(key, pool_pre_ping=True, pool_recycle=1800)
        _ENGINE_KEY = key
    return _ENGINE
