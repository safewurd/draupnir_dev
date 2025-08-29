-- === Key/value settings (two tables for compatibility) ===
CREATE TABLE IF NOT EXISTS global_settings (
  key   TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS settings (
  key   TEXT PRIMARY KEY,
  value TEXT
);

-- === Dropdown metadata ===
CREATE TABLE IF NOT EXISTS institutions (
  id   SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS tax_treatments (
  id   SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL
);

-- Seed the common options
INSERT INTO institutions (name) VALUES
  ('RBC'), ('RBCDI'), ('Sunlife'), ('RBC Insurance')
ON CONFLICT (name) DO NOTHING;

INSERT INTO tax_treatments (name) VALUES
  ('Taxable'), ('TFSA'), ('RRSP'), ('RESP'), ('RRIF')
ON CONFLICT (name) DO NOTHING;

-- === Portfolios (with yield + reinvest fields) ===
CREATE TABLE IF NOT EXISTS portfolios (
  portfolio_id          SERIAL PRIMARY KEY,
  portfolio_owner       TEXT,
  institution           TEXT,
  tax_treatment         TEXT,
  account_number        TEXT,
  portfolio_name        TEXT,
  interest_yield        DOUBLE PRECISION,
  div_eligible_yield    DOUBLE PRECISION,
  div_noneligible_yield DOUBLE PRECISION,
  reinvest_interest     INTEGER DEFAULT 1,
  reinvest_dividends    INTEGER DEFAULT 1
);

-- === Trades (columns your app reads today) ===
CREATE TABLE IF NOT EXISTS trades (
  trade_id        SERIAL PRIMARY KEY,
  portfolio_id    INTEGER,
  portfolio_name  TEXT,
  ticker          TEXT,
  currency        TEXT,
  action          TEXT,          -- 'BUY'/'SELL'
  quantity        DOUBLE PRECISION,
  price           DOUBLE PRECISION,
  commission      DOUBLE PRECISION,
  yahoo_symbol    TEXT,
  trade_date      TIMESTAMPTZ
);
