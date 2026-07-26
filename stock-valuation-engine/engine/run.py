"""
Entry point run by GitHub Actions (and locally: `python -m engine.run`).

For each ticker in config/tickers.yaml:
  1. fetch raw data (fetch.py)
  2. compute derived valuation metrics (metrics.py)
  3. append a dated snapshot row to data/history.db
  4. overwrite data/latest.csv with the most recent snapshot for every ticker

Using SQLite for history + a CSV for "latest" gives you:
  - a queryable time series for free (data/history.db) once you're tracking
    50+ names daily
  - a plain CSV that's easy to diff in git and trivially readable by Streamlit
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from engine.fetch import fetch_watchlist
from engine.metrics import compute_valuation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("engine.run")

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "tickers.yaml"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "history.db"
LATEST_CSV_PATH = DATA_DIR / "latest.csv"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def flatten_watchlist(config: dict) -> list[dict]:
    """
    Turns the nested section -> layer -> tickers config into a flat list of
    {symbol, section, section_target_pct, layer, layer_weight, portfolio_weight, thesis}.

    portfolio_weight = section_target_pct * layer_weight * (ticker's share within the layer).
    A layer's tickers can be a plain list (weight split evenly across them) or a list of
    {ticker, weight} dicts (explicit split, e.g. the Business & Futuristic Overlay layer).
    """
    rows = []
    for section in config["sections"]:
        section_name = section["name"]
        section_target = section["target_pct"]
        for layer in section["layers"]:
            layer_name = layer["name"]
            layer_weight = layer["weight"]
            tickers = layer["tickers"]

            if tickers and isinstance(tickers[0], dict):
                entries = [(t["ticker"], t["weight"]) for t in tickers]
            else:
                even_share = 1 / len(tickers) if tickers else 0
                entries = [(t, even_share) for t in tickers]

            for symbol, in_layer_weight in entries:
                rows.append({
                    "symbol": symbol,
                    "section": section_name,
                    "section_target_pct": section_target,
                    "layer": layer_name,
                    "layer_weight": layer_weight,
                    "portfolio_weight": section_target * layer_weight * in_layer_weight,
                    "thesis": layer.get("thesis", "").strip(),
                })
    return rows


def write_to_history_db(rows: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("valuation_snapshots", conn, if_exists="append", index=False)


def write_latest_csv(rows: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(LATEST_CSV_PATH, index=False)


def main() -> None:
    config = load_config()
    watchlist_rows = flatten_watchlist(config)
    history_years = config.get("history_years", 5)

    # A ticker could in principle appear in more than one layer -- fetch each
    # unique symbol only once regardless.
    unique_symbols = list(dict.fromkeys(row["symbol"] for row in watchlist_rows))

    log.info("Running valuation engine for %d tickers (%d watchlist entries)",
             len(unique_symbols), len(watchlist_rows))

    fetched: dict = {}
    failures: list[str] = []
    for symbol, outcome in fetch_watchlist(unique_symbols, history_years=history_years):
        if isinstance(outcome, Exception):
            failures.append(symbol)
        else:
            fetched[symbol] = outcome

    results: list[dict] = []
    as_of = datetime.now(timezone.utc)
    for row in watchlist_rows:
        outcome = fetched.get(row["symbol"])
        if outcome is None:
            continue
        try:
            result = compute_valuation(
                ticker=row["symbol"],
                current_revenue_ttm=outcome.current_revenue_ttm,
                current_market_cap=outcome.current_market_cap,
                historical_ps_series=outcome.historical_ps_series,
                forward_revenue_estimate=outcome.forward_revenue_estimate,
                revenue_cadence=outcome.revenue_cadence,
                as_of=as_of,
                gross_margin=outcome.gross_margin,
                operating_margin=outcome.operating_margin,
                net_margin=outcome.net_margin,
                net_debt_to_ebitda=outcome.net_debt_to_ebitda,
                interest_coverage=outcome.interest_coverage,
                free_cash_flow_ttm=outcome.free_cash_flow_ttm,
                fcf_margin=outcome.fcf_margin,
                cash_conversion=outcome.cash_conversion,
                revenue_currency=outcome.revenue_currency,
                price_currency=outcome.price_currency,
                fx_rate_applied=outcome.fx_rate_applied,
                currency_note=outcome.currency_note,
            )
            record = result.to_dict()
            record.update({
                "section": row["section"],
                "section_target_pct": row["section_target_pct"],
                "layer": row["layer"],
                "portfolio_weight": row["portfolio_weight"],
                "thesis": row["thesis"],
            })
            results.append(record)
            log.info("%s: P/S %.2f vs median %.2f -- %s",
                      row["symbol"], result.current_ps, result.hist_median_ps, result.expectations_classification)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to compute metrics for %s: %s", row["symbol"], exc)
            failures.append(row["symbol"])

    if results:
        write_to_history_db(results)
        write_latest_csv(results)

    log.info("Done. %d succeeded, %d failed.", len(results), len(failures))
    if failures:
        log.info("Failed tickers: %s", ", ".join(sorted(set(failures))))


if __name__ == "__main__":
    main()
