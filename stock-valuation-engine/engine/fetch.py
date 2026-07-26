"""
Data fetch layer -- the only module that talks to the network.

Keeping all yfinance calls in one place means:
  - metrics.py stays pure/testable
  - retry + rate-limit handling lives in exactly one spot, which matters
    once you're pulling 50+ tickers in a single GitHub Actions run
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger("engine.fetch")


@dataclass
class RawTickerData:
    ticker: str
    current_revenue_ttm: float
    current_market_cap: float
    historical_ps_series: pd.Series
    forward_revenue_estimate: float | None
    revenue_cadence: str = "quarterly"
    # Fundamentals overlay -- all optional, since coverage varies a lot by
    # ticker/exchange and a single missing field shouldn't fail the fetch.
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    net_debt_to_ebitda: float | None = None
    interest_coverage: float | None = None
    free_cash_flow_ttm: float | None = None
    fcf_margin: float | None = None
    cash_conversion: float | None = None
    # Currency normalization metadata -- see _resolve_fx() below. Present so
    # a P/S computed for a foreign filer (e.g. an ADR reporting revenue in
    # its home currency while trading in USD) can be trusted or flagged.
    revenue_currency: str | None = None
    price_currency: str | None = None
    fx_rate_applied: float | None = None
    currency_note: str = ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _get_ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol)


def _trailing_revenue_series(tkr: yf.Ticker) -> pd.Series:
    """
    Quarterly total revenue, most recent first, as reported.
    Returns a Series indexed by quarter-end date.
    """
    qf = tkr.quarterly_financials
    if qf is None or qf.empty or "Total Revenue" not in qf.index:
        raise ValueError("No quarterly revenue data available")
    rev = qf.loc["Total Revenue"].dropna().sort_index()
    return rev


def _ttm(rev_series: pd.Series) -> float:
    """Trailing-twelve-month revenue = sum of the last 4 quarters."""
    return float(rev_series.iloc[-4:].sum()) if len(rev_series) >= 4 else float(rev_series.sum())


def _historical_ps_series(tkr: yf.Ticker, rev_series: pd.Series, years: int) -> pd.Series:
    """
    Builds a historical trailing-P/S series by combining:
      - rolling 4-quarter revenue at each quarter-end
      - shares outstanding (approximated as constant -- yfinance doesn't
        expose a clean historical shares-outstanding series for most tickers)
      - historical close price on/near that quarter-end date

    This is an approximation (shares outstanding drift is ignored) but is
    good enough to place the CURRENT P/S in its historical distribution,
    which is all the model needs.
    """
    shares = tkr.fast_info.get("shares_outstanding") or tkr.info.get("sharesOutstanding")
    if not shares:
        raise ValueError("No shares outstanding data available")

    rolling_ttm = rev_series.rolling(4).sum().dropna()
    cutoff = rolling_ttm.index.max() - pd.DateOffset(years=years)
    rolling_ttm = rolling_ttm[rolling_ttm.index >= cutoff]

    start = rolling_ttm.index.min() - pd.Timedelta(days=10)
    end = rolling_ttm.index.max() + pd.Timedelta(days=10)
    prices = tkr.history(start=start, end=end, interval="1d")["Close"]
    if prices.empty:
        raise ValueError("No price history available")
    prices.index = prices.index.tz_localize(None)

    ps_values = []
    for q_end, ttm_rev in rolling_ttm.items():
        window = prices[(prices.index >= q_end - pd.Timedelta(days=5)) & (prices.index <= q_end + pd.Timedelta(days=5))]
        if window.empty or ttm_rev <= 0:
            continue
        price = window.iloc[-1]
        market_cap_at_time = price * shares
        ps_values.append(market_cap_at_time / ttm_rev)

    return pd.Series(ps_values)


def _forward_revenue_estimate(tkr: yf.Ticker) -> float | None:
    """
    Pulls the analyst +1y forward revenue estimate if available.
    yfinance's revenue estimate table has changed shape across versions,
    so this is defensive about column/index naming.
    """
    try:
        est = tkr.get_revenue_estimate()
    except Exception:
        return None
    if est is None or est.empty:
        return None
    for row_label in ("+1y", "1y", "0y"):
        if row_label in est.index:
            row = est.loc[row_label]
            for col in ("avg", "Avg. Estimate", "average"):
                if col in row.index and pd.notna(row[col]):
                    return float(row[col])
    return None


def _row_ttm(statement: pd.DataFrame, *row_names: str) -> float | None:
    """
    Sums the trailing 4 quarters of the first matching row name found in a
    quarterly statement (income statement or cash flow). Row labels vary
    across yfinance versions/tickers, so several aliases can be tried.
    """
    if statement is None or statement.empty:
        return None
    for name in row_names:
        if name in statement.index:
            series = statement.loc[name].dropna().sort_index()
            if series.empty:
                continue
            return float(series.iloc[-4:].sum()) if len(series) >= 4 else float(series.sum())
    return None


def _row_latest(statement: pd.DataFrame, *row_names: str) -> float | None:
    """Most recent single-quarter value of the first matching row (for balance-sheet items,
    which are point-in-time, not summed like income/cash-flow items)."""
    if statement is None or statement.empty:
        return None
    for name in row_names:
        if name in statement.index:
            series = statement.loc[name].dropna().sort_index()
            if not series.empty:
                return float(series.iloc[-1])
    return None


def _fundamentals(tkr: yf.Ticker, revenue_ttm: float) -> dict:
    """
    Best-effort margin, leverage, and cash-flow overlay on top of the
    revenue-only valuation model. Every field is optional -- a ticker
    missing one statement (common for foreign listings) still gets the
    rest, rather than the whole fetch failing.
    """
    out: dict = {}
    ebitda: float | None = None
    net_income: float | None = None

    try:
        income = tkr.quarterly_financials
        gross_profit = _row_ttm(income, "Gross Profit")
        operating_income = _row_ttm(income, "Operating Income", "EBIT")
        net_income = _row_ttm(income, "Net Income", "Net Income Common Stockholders")
        interest_expense = _row_ttm(income, "Interest Expense", "Interest Expense Non Operating")
        ebitda = _row_ttm(income, "EBITDA", "Normalized EBITDA")

        if revenue_ttm:
            if gross_profit is not None:
                out["gross_margin"] = gross_profit / revenue_ttm
            if operating_income is not None:
                out["operating_margin"] = operating_income / revenue_ttm
            if net_income is not None:
                out["net_margin"] = net_income / revenue_ttm

        if operating_income is not None and interest_expense:
            out["interest_coverage"] = operating_income / abs(interest_expense)
    except Exception:
        log.debug("%s: income statement fundamentals unavailable", getattr(tkr, "ticker", "?"))

    try:
        balance = tkr.quarterly_balance_sheet
        total_debt = _row_latest(balance, "Total Debt")
        cash = _row_latest(balance, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
        if total_debt is not None and cash is not None and ebitda:
            out["net_debt_to_ebitda"] = (total_debt - cash) / ebitda
    except Exception:
        pass

    try:
        cashflow = tkr.quarterly_cashflow
        fcf = _row_ttm(cashflow, "Free Cash Flow")
        if fcf is None:
            op_cf = _row_ttm(cashflow, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
            capex = _row_ttm(cashflow, "Capital Expenditure")
            if op_cf is not None and capex is not None:
                fcf = op_cf - abs(capex)
        if fcf is not None:
            out["free_cash_flow_ttm"] = fcf
            if revenue_ttm:
                out["fcf_margin"] = fcf / revenue_ttm
            if net_income:
                out["cash_conversion"] = fcf / net_income
    except Exception:
        pass

    return out



def _fx_rate(from_ccy: str, to_ccy: str) -> float | None:
    """
    Units of to_ccy per 1 unit of from_ccy. Tries the direct Yahoo FX pair
    first (e.g. "TWDUSD=X"), then the inverse pair if the direct one isn't
    quoted. Returns None if neither resolves -- caller must treat that as
    "can't verify, don't silently guess".
    """
    if from_ccy == to_ccy:
        return 1.0
    try:
        direct = yf.Ticker(f"{from_ccy}{to_ccy}=X").fast_info.get("last_price")
        if direct:
            return float(direct)
    except Exception:
        pass
    try:
        inverse = yf.Ticker(f"{to_ccy}{from_ccy}=X").fast_info.get("last_price")
        if inverse:
            return 1.0 / float(inverse)
    except Exception:
        pass
    return None


def _resolve_fx(tkr: yf.Ticker) -> tuple[str | None, str | None, float | None, str]:
    """
    Detects when a ticker's financial statements are reported in a
    different currency than its market cap/share price (common for ADRs
    and dual-listed foreign filers -- e.g. TSM reports revenue in TWD but
    trades in USD). Returns (revenue_currency, price_currency, fx_rate,
    note) where fx_rate converts an amount FROM revenue_currency TO
    price_currency; multiply revenue figures by it before comparing them
    to market cap / share price.
    """
    try:
        revenue_ccy = tkr.info.get("financialCurrency")
        price_ccy = tkr.fast_info.get("currency")
    except Exception:
        return None, None, None, "Currency metadata unavailable."

    if not revenue_ccy or not price_ccy:
        return revenue_ccy, price_ccy, None, "Currency metadata incomplete -- unable to verify consistency."

    if revenue_ccy == price_ccy:
        return revenue_ccy, price_ccy, 1.0, ""

    rate = _fx_rate(revenue_ccy, price_ccy)
    if rate is None:
        return revenue_ccy, price_ccy, None, (
            f"Revenue reported in {revenue_ccy} but priced in {price_ccy}, and no FX rate could be "
            f"resolved -- P/S and margin figures for this ticker are UNRELIABLE, treat with caution."
        )
    return revenue_ccy, price_ccy, rate, (
        f"Revenue converted from {revenue_ccy} to {price_ccy} at {rate:.4g} to match the share price currency."
    )


def fetch_ticker(symbol: str, history_years: int = 5) -> RawTickerData:
    tkr = _get_ticker(symbol)

    rev_series = _trailing_revenue_series(tkr)
    native_revenue_ttm = _ttm(rev_series)  # in financial_currency, used for fundamentals ratios

    revenue_ccy, price_ccy, fx_rate, currency_note = _resolve_fx(tkr)

    # P/S needs revenue in the SAME currency as market cap/share price.
    # Margins/ratios (gross margin, FCF margin, etc.) don't -- numerator
    # and denominator are both in financial_currency already, so a ratio
    # is currency-agnostic and is computed separately, below, on the
    # native (unconverted) figures.
    ps_rev_series = rev_series * fx_rate if fx_rate else rev_series
    current_revenue_ttm = _ttm(ps_rev_series) if fx_rate else native_revenue_ttm

    market_cap = tkr.fast_info.get("market_cap") or tkr.info.get("marketCap")
    if not market_cap:
        raise ValueError(f"{symbol}: no market cap available")

    hist_ps = _historical_ps_series(tkr, ps_rev_series, years=history_years)
    forward_rev = _forward_revenue_estimate(tkr)
    if forward_rev and fx_rate:
        forward_rev = forward_rev * fx_rate
    fundamentals = _fundamentals(tkr, native_revenue_ttm)

    return RawTickerData(
        ticker=symbol,
        current_revenue_ttm=current_revenue_ttm,
        current_market_cap=float(market_cap),
        historical_ps_series=hist_ps,
        forward_revenue_estimate=forward_rev,
        revenue_currency=revenue_ccy,
        price_currency=price_ccy,
        fx_rate_applied=fx_rate,
        currency_note=currency_note,
        **fundamentals,
    )


def fetch_watchlist(tickers: list[str], history_years: int = 5, pause_seconds: float = 1.0):
    """
    Fetches each ticker in turn, yielding (symbol, RawTickerData | Exception).
    A small pause between calls keeps a 50+ ticker run polite to Yahoo's
    unofficial endpoint and avoids tripping rate limits.
    """
    for symbol in tickers:
        try:
            yield symbol, fetch_ticker(symbol, history_years=history_years)
        except Exception as exc:  # noqa: BLE001 -- we want to keep going on any single-ticker failure
            log.warning("Failed to fetch %s: %s", symbol, exc)
            yield symbol, exc
        time.sleep(pause_seconds)
