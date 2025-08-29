from sqlalchemy import text
import pandas as pd
from draupnir_core.db_config import get_engine

engine = get_engine()

# 1) Insert a portfolio
with engine.begin() as conn:
    pid = conn.execute(text("""
        INSERT INTO portfolios (portfolio_owner, institution, tax_treatment, account_number, portfolio_name)
        VALUES ('Alice','RBC','TFSA','12345','Alice - RBC - TFSA - 12345')
        RETURNING portfolio_id
    """)).scalar_one()
print("New portfolio_id:", pid)

# 2) Insert a trade via pandas
trade = pd.DataFrame([{
    "portfolio_id": pid,
    "portfolio_name": "Alice - RBC - TFSA - 12345",
    "ticker": "RY.TO",
    "currency": "CAD",
    "action": "BUY",
    "quantity": 10,
    "price": 120.50,
    "commission": 0.0,
    "yahoo_symbol": "RY.TO",
    "trade_date": pd.Timestamp.utcnow()
}])
trade.to_sql("trades", engine, if_exists="append", index=False, method="multi")
print("Inserted 1 trade.")

# 3) Read back
dfp = pd.read_sql_query("SELECT portfolio_id, portfolio_name FROM portfolios ORDER BY portfolio_id DESC LIMIT 3", engine)
dft = pd.read_sql_query("SELECT ticker, quantity FROM trades ORDER BY trade_id DESC LIMIT 3", engine)
print("Portfolios:\n", dfp)
print("Trades:\n", dft)
