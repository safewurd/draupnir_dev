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

# ---------------- Neon engine (no local DB path) ----------------
from draupnir_core.db_config import get_engine  # ensures NEON_DB_URL is used

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

# Show masked Neon URL so you can confirm where you're connected
NEON_URL = os.getenv("NEON_DB_URL", "")
masked = NEON_URL
try:
    if "://" in masked and "@" in masked:
        left, right = masked.split("://", 1)
        creds, rest = right.split("@", 1)
        user = creds.split(":")[0]
        masked = f"{left}://{user}:********@{rest}"
except Exception:
    masked = "(set NEON_DB_URL)"

st.caption(f"DB: `{masked}`")

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
