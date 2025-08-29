# app.py
import os
import streamlit as st

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Draupnir",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- DB label helper (Postgres/Neon) ----------------
# Prefer get_db_label() from db_config if available; otherwise fallback here.
try:
    from draupnir_core.db_config import get_db_label  # uses .env and hides secrets
except Exception:
    from urllib.parse import urlparse

    def get_db_label() -> str:
        raw = (
            os.getenv("NEON_DB_URL")
            or os.getenv("DATABASE_URL")
            or os.getenv("POSTGRES_URL")
            or ""
        )
        if not raw:
            return "(no DB configured)"
        p = urlparse(raw)
        vendor = "Postgres" if (p.scheme or "").startswith("postgres") else (p.scheme or "DB")
        host = p.hostname or "?"
        db = (p.path or "").lstrip("/") or "?"
        user = p.username or "?"
        return f"{vendor} @ {host}/{db} as {user}"

# ---------------- Neon engine (import for side-effect consistency) ----------------
# Your modules call get_engine() internally; importing keeps a single source of truth.
from draupnir_core.db_config import get_engine, set_active_env, get_active_env  # noqa: F401

# ---------------- Core modules ----------------
from draupnir_core import settings as settings_mod
from draupnir_core import portfolio as portfolio_mod
from draupnir_core import summary as summary_mod
from draupnir_core import trades as trades_mod

# Pull tab functions
from draupnir_core.summary import summary_tab
from draupnir_core.portfolio import portfolio_tab
from draupnir_core.trades import trades_tab
from draupnir_core.settings import settings_tab

# ---------------- One-time base table init ----------------
try:
    # Ensures required tables exist in Neon (safe to run multiple times)
    settings_mod.create_settings_tables()
except Exception as ex:
    st.error(f"Failed to initialize base tables: {ex}")

# ---------------- Header ----------------
st.markdown("# 🧠 Draupnir Portfolio Management")
st.markdown("Welcome to your private portfolio management and forecasting system.")

# ---- UI toggle (main page): choose DB environment ----
_current_env = get_active_env()
_env_choice = st.radio(
    "Environment",
    options=["development", "production"],
    index=0 if _current_env == "development" else 1,
    help="Switch between your Neon dev and prod endpoints for this session.",
    horizontal=True,
)
set_active_env(_env_choice)

# Show a safe DB label (no password)
st.caption(f"DB: `{get_db_label()}`")

# ---------------- Tabs ----------------
tabs = st.tabs(["Summary", "Portfolio", "Trade Blotter", "Settings"])

with tabs[0]:
    summary_tab()
with tabs[1]:
    portfolio_tab()
with tabs[2]:
    trades_tab()
with tabs[3]:
    settings_tab()
