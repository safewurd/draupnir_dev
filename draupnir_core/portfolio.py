import math
from functools import lru_cache
from typing import Dict, Optional, List, Iterable, Tuple
from io import BytesIO
from datetime import datetime

import pandas as pd
import yfinance as yf
import streamlit as st
from sqlalchemy import text, inspect

from draupnir_core.db_config import get_engine

# =========================
# Quiet Yahoo helpers
# =========================

CAD_SUFFIX_CANDIDATES: List[str] = [".TO", ".V", ".NE", ".CN"]  # TSX, TSXV, NEO, CSE

def _has_price_data(sym: str) -> bool:
    if not sym:
        return False
    try:
        hist = yf.Ticker(sym).history(period="1d", auto_adjust=False, actions=False, raise_errors=False)
        return (not hist.empty) and ("Close" in hist.columns) and pd.notna(hist["Close"]).any()
    except Exception:
        return False

@lru_cache(maxsize=4096)
def resolve_yahoo_symbol(ticker: str, currency: str) -> Optional[str]:
    if not ticker:
        return None
    t = ticker.strip().upper()
    cur = (currency or "").upper()

    if cur == "CAD":
        primary = f"{t}.TO"
        if _has_price_data(primary):
            return primary
        for suff in CAD_SUFFIX_CANDIDATES[1:]:
            sym = f"{t}{suff}"
            if _has_price_data(sym):
                return sym
        return primary

    if _has_price_data(t):
        return t
    return t

# =========================
# DB helpers (Neon via SQLAlchemy)
# =========================

def _table_columns(table_name: str) -> List[str]:
    engine = get_engine()
    insp = inspect(engine)
    try:
        return [c["name"] for c in insp.get_columns(table_name)]
    except Exception:
        return []

def _read_table(name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    engine = get_engine()
    cols_sql = ", ".join(columns) if columns else "*"
    try:
        return pd.read_sql_query(f"SELECT {cols_sql} FROM {name}", engine)
    except Exception:
        return pd.DataFrame()

# =========================
# Base currency + FX helpers
# =========================

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

@st.cache_data(show_spinner=False, ttl=300)
def _fetch_fx_pair(symbol: str) -> Optional[float]:
    try:
        hist = yf.Ticker(symbol).history(period="1d", auto_adjust=False, actions=False, raise_errors=False)
        if hist.empty or "Close" not in hist.columns:
            return None
        close = hist["Close"].dropna()
        return float(close.iloc[-1]) if not close.empty else None
    except Exception:
        return None

def _fx_to_base_rates(base: str) -> Dict[str, float]:
    """
    Return a mapping { 'CAD': rate_to_base, 'USD': rate_to_base }.
    If base == CAD: USD value → *USDCAD*, CAD → *1.0*
    If base == USD: CAD value → *CADUSD*, USD → *1.0*
    """
    base = (base or "").upper()
    rates = {"CAD": 1.0, "USD": 1.0}
    if base == "CAD":
        r = _fetch_fx_pair("USDCAD=X")
        rates["USD"] = r if r and r > 0 else 1.35
    elif base == "USD":
        r = _fetch_fx_pair("CADUSD=X")
        rates["CAD"] = r if r and r > 0 else 0.74
    return rates

# =========================
# Loads
# =========================

def load_trades(_: Optional[str] = None) -> pd.DataFrame:
    cols = _table_columns("trades")
    wanted = [
        "trade_id","portfolio_id","portfolio_name","ticker","currency",
        "action","quantity","price","commission","yahoo_symbol","trade_date",
    ]
    for opt in ["account_number","created_at","trade_value_cad","trade_value_usd","trade_usdcad"]:
        if opt in cols:
            wanted.append(opt)

    df = _read_table("trades", [c for c in wanted if c in cols])
    if df.empty:
        # keep the same shape
        for c in wanted:
            if c not in df.columns:
                df[c] = pd.Series(dtype="object")
        return df[wanted]

    def norm_str(series):
        return series.astype(str).str.strip().str.upper()

    for col in ["portfolio_name","ticker","currency","action","yahoo_symbol"]:
        if col in df.columns:
            df[col] = norm_str(df[col])
        else:
            df[col] = ""

    for col in ["quantity","price","commission","trade_value_cad","trade_value_usd","trade_usdcad"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "portfolio_id" in df.columns:
        df["portfolio_id"] = pd.to_numeric(df["portfolio_id"], errors="coerce").astype("Int64")
    if "trade_id" in df.columns:
        df["trade_id"] = pd.to_numeric(df["trade_id"], errors="coerce").astype("Int64")

    return df[[c for c in wanted if c in df.columns]]

def load_portfolios(_: Optional[str] = None) -> pd.DataFrame:
    cols = _table_columns("portfolios")
    wanted = [
        "portfolio_id", "portfolio_name", "account_number",
        "portfolio_owner", "institution", "tax_treatment",
        "interest_yield", "div_eligible_yield", "div_noneligible_yield",
        "reinvest_interest", "reinvest_dividends"
    ]
    wanted = [c for c in wanted if c in cols]
    return _read_table("portfolios", wanted)

def backfill_yahoo_symbols(db_path_unused: Optional[str], trades_df: pd.DataFrame) -> None:
    if trades_df.empty or "trade_id" not in trades_df.columns:
        return
    if "yahoo_symbol" not in trades_df.columns:
        return
    missing = (trades_df["yahoo_symbol"].isna()) | (trades_df["yahoo_symbol"].astype(str).str.strip() == "")
    candidates = trades_df.loc[missing, ["trade_id","ticker","currency"]].dropna(subset=["trade_id"])
    if candidates.empty:
        return

    updates: List[dict] = []
    for _, r in candidates.iterrows():
        sym = resolve_yahoo_symbol(str(r["ticker"]), str(r["currency"]))
        if sym and _has_price_data(sym):
            updates.append({"sym": sym, "tid": int(r["trade_id"])})

    if not updates:
        return

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("UPDATE trades SET yahoo_symbol = :sym WHERE trade_id = :tid"), updates)

# =========================
# Aggregation using trade_value_* as Book Value
# =========================

def aggregate_positions(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per (portfolio_key, effective_symbol) with:
      - current_qty
      - book_value_cad, book_value_usd (from signed sums of trade_value_* by action)
      - metadata (ticker, currency, yahoo_symbol, portfolio_id/name)
    """
    if trades.empty:
        return pd.DataFrame()

    # Resolve effective symbol
    sym = trades["yahoo_symbol"].astype(str).str.strip()
    missing = sym.isna() | (sym == "")
    if missing.any():
        sym.loc[missing] = trades.loc[missing].apply(
            lambda r: resolve_yahoo_symbol(str(r["ticker"]), str(r["currency"])), axis=1
        )
    trades = trades.assign(effective_symbol=sym.fillna(""))

    # Signed quantity and signed trade values
    sign = trades["action"].map(lambda a: 1.0 if a == "BUY" else (-1.0 if a == "SELL" else 0.0))
    trades = trades.assign(signed_qty=trades["quantity"] * sign)

    if "trade_value_cad" in trades.columns:
        trades["signed_value_cad"] = pd.to_numeric(trades["trade_value_cad"], errors="coerce").fillna(0.0) * sign
    else:
        trades["signed_value_cad"] = 0.0

    if "trade_value_usd" in trades.columns:
        trades["signed_value_usd"] = pd.to_numeric(trades["trade_value_usd"], errors="coerce").fillna(0.0) * sign
    else:
        trades["signed_value_usd"] = 0.0

    # If portfolio_id missing, fall back to name as the grouping key
    trades["portfolio_key"] = trades["portfolio_id"].where(trades["portfolio_id"].notna(), trades["portfolio_name"])

    # Aggregate quantity and book values
    qty = (
        trades.groupby(["portfolio_key","effective_symbol"], dropna=False)["signed_qty"]
        .sum()
        .reset_index()
        .rename(columns={"signed_qty": "current_qty"})
    )

    book = trades.groupby(["portfolio_key","effective_symbol"], dropna=False).agg(
        book_value_cad=("signed_value_cad","sum"),
        book_value_usd=("signed_value_usd","sum"),
    ).reset_index()

    pos = qty.merge(book, on=["portfolio_key","effective_symbol"], how="left").fillna({
        "book_value_cad": 0.0, "book_value_usd": 0.0
    })

    # Meta for display
    meta = trades.groupby(["portfolio_key","effective_symbol"], dropna=False).agg(
        display_ticker=("ticker","first"),
        currency=("currency","first"),
        yahoo_symbol=("yahoo_symbol","first"),
        portfolio_id=("portfolio_id","first"),
        trade_portfolio_name=("portfolio_name","first"),
    ).reset_index()

    pos = pos.merge(meta, on=["portfolio_key","effective_symbol"], how="left")

    return pos[[
        "portfolio_key","portfolio_id","trade_portfolio_name",
        "display_ticker","currency","yahoo_symbol","effective_symbol",
        "current_qty","book_value_cad","book_value_usd"
    ]]

# =========================
# Pricing & Valuation to Base
# =========================

@st.cache_data(show_spinner=False, ttl=60)
def _fetch_price_single(sym: str, cache_buster: str = "") -> Optional[float]:
    if not sym:
        return None
    try:
        hist = yf.Ticker(sym).history(period="1d", auto_adjust=False, actions=False, raise_errors=False)
        if hist.empty or "Close" not in hist.columns:
            return None
        px = hist["Close"].dropna()
        return float(px.iloc[-1]) if not px.empty else None
    except Exception:
        return None

def fetch_prices(symbols: Iterable[str]) -> Dict[str, Optional[float]]:
    buster = str(st.session_state.get("portfolio_refresh_token", ""))
    out: Dict[str, Optional[float]] = {}
    for s in symbols:
        s = (s or "").strip()
        out[s] = _fetch_price_single(s, cache_buster=buster) if s else None
    return out

def value_positions(pos: pd.DataFrame) -> pd.DataFrame:
    """
    Computes live price, Market Value (base), Book Value (base),
    Gain/Loss (base), Return % (base).
    """
    if pos.empty:
        return pos

    base = _get_base_currency()
    fx = _fx_to_base_rates(base)

    pos = pos.copy()

    # live prices (native)
    price_map = fetch_prices(pos["effective_symbol"].fillna("").tolist())
    pos["live_price_native"] = pos["effective_symbol"].map(price_map)

    # Market Value in base currency
    def _mv_base(row) -> Optional[float]:
        qty = float(row.get("current_qty") or 0.0)
        px  = row.get("live_price_native")
        if pd.isna(px):
            return None
        cur = str(row.get("currency","")).upper()
        rate = fx.get(cur, 1.0)
        return qty * float(px) * float(rate)

    pos["market_value_base"] = pos.apply(_mv_base, axis=1)

    # Book Value in base currency from trade_value_* sums
    if base == "CAD":
        pos["book_value_base"] = pd.to_numeric(pos["book_value_cad"], errors="coerce").fillna(0.0)
    else:
        pos["book_value_base"] = pd.to_numeric(pos["book_value_usd"], errors="coerce").fillna(0.0)

    # Fallback: if book_value_base is all zeros (e.g., legacy table w/o new cols), estimate from buys
    if (pos["book_value_base"].abs().sum() == 0) and ("live_price_native" in pos.columns):
        # crude fallback: treat avg_book_price = last close (not ideal, but preserves UX)
        pos["book_value_base"] = 0.0  # leave zero rather than fabricate

    # Gain/Loss and Return in base
    pos["gain_loss_base"] = pos.apply(
        lambda r: (r["market_value_base"] - r["book_value_base"])
        if (pd.notna(r.get("market_value_base")) and pd.notna(r.get("book_value_base")))
        else None,
        axis=1
    )
    pos["return_pct_base"] = pos.apply(
        lambda r: (100.0 * r["gain_loss_base"] / r["book_value_base"])
        if (pd.notna(r.get("gain_loss_base")) and pd.notna(r.get("book_value_base")) and r["book_value_base"] != 0)
        else None,
        axis=1
    )

    return pos

# =========================
# Formatting
# =========================

def _format_holdings(df: pd.DataFrame):
    if df.empty:
        return df.style
    df = df.copy()
    numeric_cols = [
        "current_qty","book_value_base","market_value_base","gain_loss_base","return_pct_base",
        "live_price_native"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "current_qty" in df.columns:
        df["current_qty"] = df["current_qty"].apply(lambda x: math.floor(x) if pd.notna(x) else x)

    fmt = {}
    if "current_qty" in df.columns:
        fmt["current_qty"] = "{:,.0f}".format
    for c in ["book_value_base","market_value_base","gain_loss_base","live_price_native"]:
        if c in df.columns:
            fmt[c] = "{:,.2f}".format
    if "return_pct_base" in df.columns:
        fmt["return_pct_base"] = (lambda x: "" if pd.isna(x) else f"{float(x):.1f}%")
    return df.style.format(fmt, na_rep="")

def _render_summary_bar(valued: pd.DataFrame):
    if valued.empty:
        return
    book_total = pd.to_numeric(valued["book_value_base"], errors="coerce").sum()
    mkt_total  = pd.to_numeric(valued["market_value_base"], errors="coerce").sum()
    gl_total   = pd.to_numeric(valued["gain_loss_base"], errors="coerce").sum()
    ret_pct = None
    if book_total and book_total != 0:
        ret_pct = (gl_total / book_total) * 100.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Book Value (Base)",   f"{book_total:,.2f}")
    c2.metric("Market Value (Base)", f"{mkt_total:,.2f}")
    c3.metric("Gain / Loss (Base)",  f"{gl_total:,.2f}")
    c4.metric("Return %",            ("" if ret_pct is None else f"{ret_pct:.1f}%"))

# =========================
# Settings helpers (default portfolio)
# =========================

def _read_default_portfolio_id() -> Optional[int]:
    engine = get_engine()
    try:
        with engine.connect() as conn:
            r = conn.execute(text("SELECT value FROM global_settings WHERE key IN ('default_portfolio_id','DEFAULT_PORTFOLIO_ID') LIMIT 1")).fetchone()
            if r and r[0] is not None and str(r[0]).strip():
                return int(str(r[0]).strip())
            r2 = conn.execute(text("SELECT value FROM settings WHERE key IN ('default_portfolio_id','DEFAULT_PORTFOLIO_ID') LIMIT 1")).fetchone()
            if r2 and r2[0] is not None and str(r2[0]).strip():
                return int(str(r2[0]).strip())
    except Exception:
        return None
    return None

# =========================
# Streamlit UI
# =========================

def portfolio_tab():
    st.subheader("📁 Portfolio")

    base = _get_base_currency()
    st.caption(f"Values shown in base currency: **{base}**")

    trades = load_trades(None)
    if trades.empty:
        st.info("No trades found. Add trades in the Trade Blotter.")
        return

    try:
        backfill_yahoo_symbols(None, trades)
    except Exception:
        pass

    pos = aggregate_positions(trades)
    if pos.empty or (pd.to_numeric(pos["current_qty"], errors="coerce").fillna(0).abs().sum() == 0):
        st.info("No open positions (net quantity is zero).")
        return

    pf = load_portfolios(None)
    if not pf.empty and "portfolio_id" in pos.columns:
        pf_small = pf[["portfolio_id","portfolio_name"]].drop_duplicates()
        pos = pos.merge(pf_small, on="portfolio_id", how="left", suffixes=("", "_canon"))
        pos["portfolio_label"] = pos["portfolio_name"].where(pos["portfolio_name"].notna(), pos["trade_portfolio_name"])
    else:
        pos["portfolio_label"] = pos["trade_portfolio_name"]

    # Determine default selection from settings (by id if present)
    labels = (
        pos[["portfolio_id","portfolio_label"]]
        .drop_duplicates()
        .sort_values(by=["portfolio_label"])
        .reset_index(drop=True)
    )
    default_pid = _read_default_portfolio_id()
    default_index = 0
    if default_pid is not None:
        try:
            default_index = labels.index[labels["portfolio_id"] == default_pid].tolist()[0]
        except Exception:
            default_index = 0

    sel_label = st.selectbox(
        "Portfolio",
        options=labels["portfolio_label"].tolist(),
        index=min(default_index, len(labels) - 1) if len(labels) else 0
    )

    # Filter to the selected portfolio only
    pos = pos[pos["portfolio_label"] == sel_label]

    # Value positions (base)
    valued = value_positions(pos)

    # Summary bar
    _render_summary_bar(valued)

    # Holdings table (base)
    st.markdown("#### Holdings")
    holdings_view = valued[[
        "display_ticker","currency","current_qty","live_price_native",
        "book_value_base","market_value_base","gain_loss_base","return_pct_base"
    ]]
    st.dataframe(_format_holdings(holdings_view), use_container_width=True, hide_index=True)

    # Export Holdings
    st.markdown("---")
    st.markdown("### ⬇️ Export Portfolio Holdings")
    colx, coly = st.columns([1,3])
    with colx:
        if st.button("Prepare Excel File", key="btn_prepare_holdings_excel", type="secondary"):
            st.session_state["export_holdings_ready"] = True

    if st.session_state.get("export_holdings_ready"):
        export_df = valued.copy()
        numeric_cols = ["current_qty","book_value_base","market_value_base","gain_loss_base","return_pct_base","live_price_native"]
        for c in numeric_cols:
            if c in export_df.columns:
                export_df[c] = pd.to_numeric(export_df[c], errors="coerce")

        book_total = pd.to_numeric(export_df["book_value_base"], errors="coerce").sum()
        mkt_total  = pd.to_numeric(export_df["market_value_base"], errors="coerce").sum()
        gl_total   = pd.to_numeric(export_df["gain_loss_base"], errors="coerce").sum()
        ret_pct    = (gl_total / book_total * 100.0) if (book_total not in (None, 0) and pd.notna(book_total)) else None

        summary_df = pd.DataFrame([{
            "Portfolio": sel_label,
            "Book Value (Base)": book_total,
            "Market Value (Base)": mkt_total,
            "Gain/Loss (Base)": gl_total,
            "Return %": (None if ret_pct is None else ret_pct)
        }])

        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            cols_order = [
                "display_ticker","currency","current_qty","live_price_native",
                "book_value_base","market_value_base","gain_loss_base","return_pct_base"
            ]
            cols_order = [c for c in cols_order if c in export_df.columns]
            export_df.to_excel(writer, index=False, sheet_name="Holdings", columns=cols_order)
            summary_df.to_excel(writer, index=False, sheet_name="Summary")

            for ws_name in ["Holdings", "Summary"]:
                ws = writer.sheets[ws_name]
                for col_cells in ws.columns:
                    length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                    ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 12), 50)

        bio.seek(0)
        safe_name = "".join(ch if (ch.isalnum() or ch in ("-","_")) else "_" for ch in sel_label)
        fname = f"portfolio_holdings_{safe_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%SZ')}.xlsx"

        st.download_button(
            label="Download Excel",
            data=bio.getvalue(),
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False
        )

# If running directly for testing:
if __name__ == "__main__":
    st.set_page_config(page_title="Portfolio", layout="wide")
    portfolio_tab()
