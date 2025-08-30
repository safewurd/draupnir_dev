from functools import lru_cache
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf
from sqlalchemy import text, inspect

from draupnir_core.db_config import get_engine

# -----------------------------
# Quiet Yahoo resolver (unchanged)
# -----------------------------

CAD_SUFFIX_SECONDARY: List[str] = [".V", ".NE", ".CN"]

def _has_price_data(sym: str) -> bool:
    if not sym:
        return False
    try:
        hist = yf.Ticker(sym).history(period="1d", auto_adjust=False, actions=False, raise_errors=False)
        return (not hist.empty) and ("Close" in hist.columns) and pd.notna(hist["Close"]).any()
    except Exception:
        return False

@lru_cache(maxsize=2048)
def resolve_yahoo_symbol(ticker: str, currency: str) -> Optional[str]:
    if not ticker:
        return None
    t = ticker.strip().upper()
    cur = (currency or "").upper()

    if cur == "CAD":
        primary = f"{t}.TO"
        if _has_price_data(primary):
            return primary
        for suff in CAD_SUFFIX_SECONDARY:
            sym = f"{t}{suff}"
            if _has_price_data(sym):
                return sym
        return primary

    if _has_price_data(t):
        return t
    return t

def _table_columns(table: str) -> List[str]:
    insp = inspect(get_engine())
    try:
        return [c["name"] for c in insp.get_columns(table)]
    except Exception:
        return []

# -----------------------------
# Read helpers (unchanged)
# -----------------------------

def fetch_portfolios() -> pd.DataFrame:
    engine = get_engine()
    try:
        return pd.read_sql(
            "SELECT portfolio_id, portfolio_name, account_number, portfolio_owner, institution, tax_treatment "
            "FROM portfolios ORDER BY portfolio_owner, institution, account_number;",
            engine
        )
    except Exception:
        return pd.DataFrame(columns=[
            "portfolio_id","portfolio_name","account_number","portfolio_owner","institution","tax_treatment"
        ])

def get_portfolio_by_name(name: str) -> Optional[dict]:
    if not name:
        return None
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT portfolio_id, portfolio_name, account_number, portfolio_owner, institution, tax_treatment "
            "FROM portfolios WHERE portfolio_name = :n LIMIT 1;"
        ), {"n": name}).fetchone()
    if not row:
        return None
    cols = ["portfolio_id","portfolio_name","account_number","portfolio_owner","institution","tax_treatment"]
    return dict(zip(cols, row))

# -----------------------------
# Insert / Delete
# -----------------------------

def _calc_trade_values(currency: str, price: float, quantity: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (trade_value_cad, trade_value_usd, trade_usdcad)

    NOTE: This is still called at INSERT time to persist the values once.
    """
    # We no longer expose a UI button to mass-recalculate after insert.
    # If you ever restore that, wire a batch updater to call this logic.
    import yfinance as _yf  # local import to avoid polluting module scope
    try:
        hist = _yf.Ticker("USDCAD=X").history(period="1d", auto_adjust=False, actions=False, raise_errors=False)
        fx = float(hist["Close"].dropna().iloc[-1]) if (not hist.empty and "Close" in hist.columns) else None
    except Exception:
        fx = None

    ccy = (currency or "").upper().strip()
    px  = float(price or 0.0)
    qty = float(quantity or 0.0)

    if px <= 0 or qty <= 0 or ccy not in ("CAD", "USD"):
        return None, None, fx

    if ccy == "CAD":
        val_cad = px * qty
        val_usd = (val_cad / fx) if (fx and fx > 0) else None
    else:  # USD
        val_usd = px * qty
        val_cad = (val_usd * fx) if (fx and fx > 0) else None

    return (val_cad, val_usd, fx)

def insert_trade(row: dict) -> int:
    """
    Insert into Neon `trades`. Only include columns that actually exist in the table.
    Also computes trade_value_cad, trade_value_usd, trade_usdcad at insert time.
    """
    cols_in_db = _table_columns("trades")
    payload: Dict[str, object] = {}

    # Compute trade values once at insert
    tval_cad, tval_usd, t_usdcad = _calc_trade_values(
        row.get("currency"), row.get("price"), row.get("quantity")
    )

    mapping = {
        "portfolio_id": row.get("portfolio_id"),
        "portfolio_name": row.get("portfolio_name"),
        "ticker": row.get("ticker"),
        "currency": row.get("currency"),
        "action": row.get("action"),
        "quantity": float(row.get("quantity", 0.0)),
        "price": float(row.get("price", 0.0)),
        "commission": float(row.get("commission", 0.0)),
        "yahoo_symbol": row.get("yahoo_symbol") or "",
        "trade_date": row.get("trade_date"),
        # optional legacy columns:
        "account_number": row.get("account_number"),
        "created_at": row.get("created_at"),
        # persisted values:
        "trade_value_cad": tval_cad,
        "trade_value_usd": tval_usd,
        "trade_usdcad": t_usdcad,
    }

    for k, v in mapping.items():
        if k in cols_in_db:
            payload[k] = v

    engine = get_engine()
    with engine.begin() as conn:
        cols = ", ".join(payload.keys())
        vals = ", ".join([f":{k}" for k in payload.keys()])
        sql = text(f"INSERT INTO trades ({cols}) VALUES ({vals}) RETURNING trade_id;")
        new_id = conn.execute(sql, payload).scalar_one()
    return int(new_id)

def delete_trade(trade_id: int) -> bool:
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM trades WHERE trade_id = :id"), {"id": int(trade_id)})
        return True
    except Exception:
        return False

# -----------------------------
# Loaders (unchanged)
# -----------------------------

def _load_recent(limit: int = 200) -> pd.DataFrame:
    engine = get_engine()
    try:
        return pd.read_sql(
            "SELECT trade_id, portfolio_name, ticker, currency, action, "
            "quantity, price, commission, yahoo_symbol, trade_date, "
            "trade_value_cad, trade_value_usd, trade_usdcad "
            "FROM trades ORDER BY trade_id DESC LIMIT %(lim)s;",
            engine, params={"lim": int(limit)}
        )
    except Exception:
        return pd.DataFrame()

# -----------------------------
# UI (same core functionality, backfill removed)
# -----------------------------

def _validate_inputs(portfolio_name, ticker, currency, action, quantity, price) -> list[str]:
    errs = []
    if not portfolio_name or not str(portfolio_name).strip():
        errs.append("Portfolio is required.")
    if not ticker or not str(ticker).strip():
        errs.append("Ticker is required.")
    if currency not in ("CAD", "USD"):
        errs.append("Currency must be CAD or USD.")
    if action not in ("BUY", "SELL"):
        errs.append("Action must be BUY or SELL.")
    try:
        q = float(quantity)
        if q <= 0:
            errs.append("Quantity must be greater than 0.")
    except Exception:
        errs.append("Quantity is invalid.")
    try:
        p = float(price)
        if p <= 0:
            errs.append("Price must be greater than 0.")
    except Exception:
        errs.append("Price is invalid.")
    return errs

def _normalize_str(s: Optional[str]) -> str:
    return "" if s is None else str(s).strip().upper()

def _dedupe_key(portfolio_name, ticker, currency, action, quantity, price, trade_date) -> str:
    return f"{_normalize_str(portfolio_name)}|{_normalize_str(ticker)}|{_normalize_str(currency)}|" \
           f"{_normalize_str(action)}|{float(quantity):.6f}|{float(price):.6f}|{str(trade_date)}"

def trades_tab():
    st.subheader("📄 Trade Blotter")

    pf = fetch_portfolios()
    if pf.empty:
        st.warning("No portfolios found. Add rows to the 'portfolios' table first (Settings tab).")
        return

    portfolio_names = sorted(pf["portfolio_name"].astype(str).str.strip().unique().tolist())

    # ---- Add Trade ----
    with st.form("add_trade_form", clear_on_submit=False):
        colA, colB = st.columns(2)
        with colA:
            selected_portfolio_name = st.selectbox("Portfolio", options=portfolio_names, index=0)
            action = st.selectbox("Action", options=["BUY", "SELL"], index=0)
            ticker = st.text_input("Ticker", placeholder="e.g., RY").upper()
            currency = st.selectbox("Currency", options=["CAD", "USD"], index=0)
        with colB:
            quantity = st.number_input("Quantity", min_value=0.0, step=1.0, format="%.4f")
            price = st.number_input("Price", min_value=0.0, step=0.01, format="%.6f")
            commission = st.number_input("Commission", min_value=0.0, step=0.01, format="%.2f")
            trade_date = st.date_input("Trade Date")

        submitted = st.form_submit_button("Add Trade", use_container_width=True)

    if submitted:
        errors = _validate_inputs(selected_portfolio_name, ticker, currency, action, quantity, price)
        if errors:
            for e in errors:
                st.error(e)
            return

        dedupe_key = _dedupe_key(selected_portfolio_name, ticker, currency, action, quantity, price, trade_date)
        if st.session_state.get("last_trade_submit_key") == dedupe_key:
            st.info("This trade was just submitted. Ignoring duplicate submit.")
            return

        p = get_portfolio_by_name(selected_portfolio_name)
        if not p:
            st.error("Selected portfolio not found.")
            return

        yahoo_symbol = resolve_yahoo_symbol(ticker, currency)

        row = {
            "portfolio_id": int(p["portfolio_id"]),
            "portfolio_name": p["portfolio_name"],
            "account_number": p.get("account_number"),
            "ticker": ticker.strip().upper(),
            "currency": currency.strip().upper(),
            "action": action.strip().upper(),
            "quantity": float(quantity),
            "price": float(price),
            "commission": float(commission or 0.0),
            "yahoo_symbol": yahoo_symbol or "",
            "trade_date": str(trade_date),
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

        try:
            trade_id = insert_trade(row)
        except Exception as ex:
            st.error(f"Failed to add trade: {ex}")
            return

        st.cache_data.clear()
        st.session_state["portfolio_refresh_token"] = datetime.utcnow().isoformat(timespec="seconds")
        st.session_state["last_trade_submit_key"] = dedupe_key

        st.success(f"Trade added (trade_id={trade_id}) to {row['portfolio_name']}")
        # Show computed values for transparency
        try:
            engine = get_engine()
            with engine.connect() as conn:
                rec = conn.execute(text("""
                    SELECT trade_value_cad, trade_value_usd, trade_usdcad
                    FROM trades WHERE trade_id = :id
                """), {"id": trade_id}).fetchone()
            if rec:
                st.caption(f"Calculated at insert: CAD {rec[0]!s} | USD {rec[1]!s} | USDCAD {rec[2]!s}")
        except Exception:
            pass

    # ---- Recent trades (includes persisted value columns) ----
    st.markdown("#### Recent Trades")
    try:
        recent = _load_recent(25)
        if not recent.empty:
            for c in ["quantity","price","commission","trade_value_cad","trade_value_usd","trade_usdcad"]:
                if c in recent.columns:
                    recent[c] = pd.to_numeric(recent[c], errors="coerce")
            st.dataframe(recent, use_container_width=True, hide_index=True)
        else:
            st.info("No trades recorded yet.")
    except Exception as ex:
        st.warning(f"Could not load recent trades: {ex}")

    # (Backfill/recalculate section has been removed as requested)
