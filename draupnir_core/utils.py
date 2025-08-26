# draupnir_core/utils.py
"""
Minimal utilities for the slimmed-down app (Summary, Portfolio, Trade Blotter).

- No forecasting/tax defaults.
- Simple helpers for reading/writing settings and formatting values.
"""

from __future__ import annotations

import sqlite3
from typing import Optional, Any
import os

# Use the per-machine DB resolver
from draupnir_core.db_config import get_db_path

os.makedirs("data", exist_ok=True)
DB_PATH = get_db_path()  # absolute path for THIS machine


# -----------------------------
# Settings (key/value) helpers
# -----------------------------
def get_setting_value(key: str) -> Optional[str]:
    """
    Read a setting from global_settings, falling back to settings.
    Returns None if not found.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT value FROM global_settings WHERE key = ? LIMIT 1;", (key,)
        ).fetchone()
        if row and row[0] is not None:
            return str(row[0])

        row2 = conn.execute(
            "SELECT value FROM settings WHERE key = ? LIMIT 1;", (key,)
        ).fetchone()
        if row2 and row2[0] is not None:
            return str(row2[0])

        return None
    finally:
        conn.close()


def set_setting_value(key: str, value: Any) -> None:
    """
    Write a setting to global_settings and mirror to settings (compat).
    Values are stored as TEXT.
    """
    val = "" if value is None else str(value)
    conn = sqlite3.connect(DB_PATH)
    try:
        with conn:
            conn.execute(
                "INSERT INTO global_settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                (key, val),
            )
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                (key, val),
            )
    finally:
        conn.close()


def get_base_currency(default: str = "CAD") -> str:
    """
    Convenience accessor used by Summary/Portfolio to pick a display/valuation currency.
    """
    cur = (get_setting_value("base_currency") or "").strip().upper()
    return cur if cur in ("CAD", "USD") else default


def get_market_data_provider(default: str = "yahoo") -> str:
    """
    Convenience accessor for price/yield source (e.g., 'yahoo').
    """
    provider = (get_setting_value("market_data_provider") or "").strip().lower()
    return provider if provider in ("yahoo", "alpha_vantage", "polygon") else default


# -----------------------------
# Formatting helpers
# -----------------------------
def format_ccy(val: Optional[float]) -> str:
    """
    Format a number to currency-like string with 2 decimals and thousands separators.
    Returns '' for None or NaN-ish values.
    """
    try:
        if val is None:
            return ""
        return f"{float(val):,.2f}"
    except Exception:
        return ""


def format_pct(val: Optional[float], decimals: int = 1) -> str:
    """
    Format a percentage (e.g., 0.1234 -> '12.3%') if you pass decimal form.
    If you pass already-in-percent (e.g., 12.34), set decimals accordingly and divide by 100 yourself first.
    """
    try:
        if val is None:
            return ""
        return f"{float(val) * 100:.{decimals}f}%"
    except Exception:
        return ""
