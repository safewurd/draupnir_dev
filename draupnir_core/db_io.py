# draupnir_core/db_io.py
import pandas as pd
from sqlalchemy import text
from draupnir_core.db_config import get_engine

def read_sql(sql: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql_query(sql, engine)

def exec_many(sql: str, rows: list[dict]) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql), rows)

def insert_trades(df: pd.DataFrame) -> None:
    engine = get_engine()
    df.to_sql("trades", engine, if_exists="append", index=False, method="multi")
