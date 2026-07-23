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
    tickers = config["tickers"]
    history_years = config.get("history_years", 5)

    log.info("Running valuation engine for %d tickers", len(tickers))
    results: list[dict] = []
    failures: list[str] = []

    for symbol, outcome in fetch_watchlist(tickers, history_years=history_years):
        if isinstance(outcome, Exception):
            failures.append(symbol)
            continue
        try:
            result = compute_valuation(
                ticker=symbol,
                current_revenue_ttm=outcome.current_revenue_ttm,
                current_market_cap=outcome.current_market_cap,
                historical_ps_series=outcome.historical_ps_series,
                forward_revenue_estimate=outcome.forward_revenue_estimate,
                revenue_cadence=outcome.revenue_cadence,
                as_of=datetime.now(timezone.utc),
            )
            results.append(result.to_dict())
            log.info("%s: P/S %.2f vs median %.2f -- %s",
                      symbol, result.current_ps, result.hist_median_ps, result.expectations_classification)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to compute metrics for %s: %s", symbol, exc)
            failures.append(symbol)

    if results:
        write_to_history_db(results)
        write_latest_csv(results)

    log.info("Done. %d succeeded, %d failed.", len(results), len(failures))
    if failures:
        log.info("Failed tickers: %s", ", ".join(failures))


if __name__ == "__main__":
    main()
