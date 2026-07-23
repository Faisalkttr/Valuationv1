# Stock valuation engine

Pulls fundamentals from Yahoo Finance, computes a valuation-vs-history
model (current P/S against its own historical percentile distribution,
plus a forward growth "gap" analysis) for a 50+ ticker watchlist, and
serves the results through a Streamlit dashboard. Fully automated via
GitHub Actions -- no server to run yourself.

## How it works

1. **`config/tickers.yaml`** -- your watchlist. Edit this to add/remove tickers.
2. **`engine/fetch.py`** -- pulls raw price, revenue, and analyst estimate data via `yfinance`.
3. **`engine/metrics.py`** -- pure computation: historical P/S percentiles, forward P/S,
   required growth to "normalise" the valuation, and the growth gap between
   what's required and what analysts expect.
4. **`engine/run.py`** -- orchestrates 1-3 across the whole watchlist, writes:
   - `data/latest.csv` -- one row per ticker, most recent snapshot
   - `data/history.db` -- SQLite table (`valuation_snapshots`) with every run ever, so you get trend charts over time
5. **`.github/workflows/refresh.yml`** -- runs `engine/run.py` on a schedule
   (default: weekdays 18:00 UTC) and commits the refreshed data back to the repo.
6. **`app.py`** -- Streamlit dashboard reading purely from `data/`, so it loads
   instantly with no live API calls at view time.

## Local setup

```bash
pip install -r requirements.txt
python -m engine.run          # populates data/latest.csv and data/history.db
streamlit run app.py
```

## Deploying

1. Push this repo to GitHub.
2. In the repo's Settings > Actions > General, make sure "Read and write permissions"
   is enabled for the `GITHUB_TOKEN` (needed for the workflow to commit data back).
3. Trigger the workflow once manually (Actions tab > Refresh valuation data > Run workflow)
   to populate `data/` for the first time.
4. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, point it at
   `app.py`. It redeploys automatically on every push -- including the daily
   automated data-refresh commits.

## Notes and known approximations

- **Historical P/S** is built from quarterly revenue x historical close price x current
  shares outstanding. Share count drift over the lookback window is ignored --
  fine for a 5-year window on large caps, less accurate for names with heavy
  buybacks/dilution.
- **Forward revenue estimate** comes from `yfinance`'s analyst estimate table, which
  isn't available for every ticker (thinly covered small caps especially).
  When missing, forward-looking fields (`forward_ps`, `growth_gap`, etc.) are left null
  and the dashboard shows "n/a" rather than guessing.
- **Rate limiting**: `engine/fetch.py` pauses briefly between tickers and retries
  transient failures. For 50+ tickers a full run can take a few minutes --
  that's expected and fine for a scheduled job.
- Yahoo Finance's public endpoints are unofficial and can change shape without
  notice; if a run starts failing broadly, check the `yfinance` changelog first.
