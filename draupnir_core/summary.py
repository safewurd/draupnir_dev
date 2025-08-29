from typing import Optional, Dict, List

import pandas as pd
import streamlit as st
import yfinance as yf
from sqlalchemy import text

from draupnir_core.db_config import get_engine
from .portfolio import (
    load_trades,
    load_portfolios,
    backfill_yahoo_symbols,
    aggregate_positions,
    value_positions,  # now computes base-ccy book/market/gain/return using trade_value_* columns
)

# -----------------------------
# Base currency (read-only caption)
# -----------------------------

def _read_settings_base_currency() -> Optional[str]:
    try:
        eng = get_engine()
        with eng.connect() as conn:
            r = conn.execute(text(
                "SELECT value FROM global_settings WHERE key IN ('base_currency','BASE_CURRENCY') LIMIT 1"
            )).fetchone()
            if r and r[0]:
                return str(r[0]).strip().upper()
            r2 = conn.execute(text(
                "SELECT value FROM settings WHERE key IN ('base_currency','BASE_CURRENCY') LIMIT 1"
            )).fetchone()
            if r2 and r2[0]:
                return str(r2[0]).strip().upper()
    except Exception:
        return None
    return None

def _get_base_currency() -> str:
    base = _read_settings_base_currency()
    if base in ("CAD", "USD"):
        return base
    return "CAD"

# -----------------------------
# Formatting helpers
# -----------------------------

def _fmt_money(x) -> str:
    return "" if pd.isna(x) else f"{float(x):,.2f}"

def _fmt_pct(x) -> str:
    return "" if pd.isna(x) else f"{float(x):.1f}%"

def _format_summary(df: pd.DataFrame, cols_money: List[str], col_pct: Optional[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for c in cols_money:
        if c in out.columns:
            out[c] = out[c].map(_fmt_money)
    if col_pct and col_pct in out.columns:
        out[col_pct] = out[col_pct].map(_fmt_pct)
    return out

def _render_totals_bar(book_series: pd.Series, mkt_series: pd.Series, gl_series: pd.Series):
    book_total = pd.to_numeric(book_series, errors="coerce").sum()
    mkt_total  = pd.to_numeric(mkt_series,  errors="coerce").sum()
    gl_total   = pd.to_numeric(gl_series,   errors="coerce").sum()
    ret_pct = (gl_total / book_total * 100.0) if (book_total not in (None, 0) and pd.notna(book_total)) else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Book Value (Base)",   f"{book_total:,.2f}")
    c2.metric("Market Value (Base)", f"{mkt_total:,.2f}")
    c3.metric("Gain / Loss (Base)",  f"{gl_total:,.2f}")
    c4.metric("Return %",            "" if ret_pct is None else f"{ret_pct:.1f}%")

# -----------------------------
# Summary Tab
# -----------------------------

def summary_tab():
    st.subheader("📈 Summary")

    base = _get_base_currency()
    st.caption(f"All amounts shown in base currency: **{base}**")

    # 1) Load trades
    trades = load_trades(None)
    if trades.empty:
        st.info("No trades found. Add trades in the Trade Blotter.")
        return

    # 2) Backfill any missing symbols (quiet)
    try:
        backfill_yahoo_symbols(None, trades)
    except Exception:
        pass

    # 3) Aggregate and value (uses trade_value_* for book value)
    pos = aggregate_positions(trades)
    if pos.empty:
        st.info("No open positions.")
        return

    # Attach canonical portfolio labels
    pf = load_portfolios(None)
    if not pf.empty and "portfolio_id" in pos.columns:
        pf_small = pf[["portfolio_id","portfolio_name"]].drop_duplicates()
        pos = pos.merge(pf_small, on="portfolio_id", how="left", suffixes=("", "_canon"))
        pos["portfolio_label"] = pos["portfolio_name"].where(pos["portfolio_name"].notna(), pos["trade_portfolio_name"])
    else:
        pos["portfolio_label"] = pos["trade_portfolio_name"]

    valued = value_positions(pos)
    if valued.empty:
        st.info("No valued positions.")
        return

    # View switcher
    view = st.radio("View", options=["By Portfolio", "By Asset"], horizontal=True)

    if view == "By Portfolio":
        grp = valued.groupby("portfolio_label", dropna=False).agg(
            book_value_base=("book_value_base", "sum"),
            market_value_base=("market_value_base", "sum"),
            gain_loss_base=("gain_loss_base", "sum"),
        ).reset_index()
        grp["return_pct_base"] = (100.0 * grp["gain_loss_base"] / grp["book_value_base"]).where(
            (grp["book_value_base"].notna()) & (grp["book_value_base"] != 0)
        )

        # Totals bar
        _render_totals_bar(grp["book_value_base"], grp["market_value_base"], grp["gain_loss_base"])

        display = grp.rename(columns={
            "portfolio_label": "Portfolio",
            "book_value_base": "Book Value (Base)",
            "market_value_base": "Market Value (Base)",
            "gain_loss_base": "Gain/Loss (Base)",
            "return_pct_base": "Return %",
        })
        display = _format_summary(display, ["Book Value (Base)","Market Value (Base)","Gain/Loss (Base)"], "Return %")
        st.dataframe(display, use_container_width=True, hide_index=True)

    else:  # By Asset
        grp = valued.groupby(["display_ticker"], dropna=False).agg(
            book_value_base=("book_value_base", "sum"),
            market_value_base=("market_value_base", "sum"),
            gain_loss_base=("gain_loss_base", "sum"),
        ).reset_index()
        grp["return_pct_base"] = (100.0 * grp["gain_loss_base"] / grp["book_value_base"]).where(
            (grp["book_value_base"].notna()) & (grp["book_value_base"] != 0)
        )

        # Totals bar
        _render_totals_bar(grp["book_value_base"], grp["market_value_base"], grp["gain_loss_base"])

        display = grp.rename(columns={
            "display_ticker": "Ticker",
            "book_value_base": "Book Value (Base)",
            "market_value_base": "Market Value (Base)",
            "gain_loss_base": "Gain/Loss (Base)",
            "return_pct_base": "Return %",
        })
        display = _format_summary(display, ["Book Value (Base)","Market Value (Base)","Gain/Loss (Base)"], "Return %")
        st.dataframe(display, use_container_width=True, hide_index=True)
