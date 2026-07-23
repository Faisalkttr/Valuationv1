"""
Valuation metrics engine.

Given a ticker's price/revenue/shares history and a forward revenue
estimate, this module derives the same fields as the CG Power export:

  Current Revenue TTM, Current Market Cap, Current P/S
  Historical Median / 75th / 90th percentile P/S
  Forward Revenue Estimate, Forward Revenue Growth, Forward P/S
  Required Revenue to Normalise Valuation, Required Revenue Growth
  Growth Gap, Years to Normalise Multiple, Expectations Burden Score
  Expectations Classification, Target Multiple Value/Label
  Valuation Anchor Confidence / Observation Count

All functions here are pure (no network calls) so they're easy to unit
test -- fetch.py is responsible for getting real data into this shape.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class ValuationResult:
    ticker: str
    as_of: str
    current_revenue_ttm: float
    current_market_cap: float
    current_ps: float
    hist_median_ps: float
    hist_p75_ps: float
    hist_p90_ps: float
    forward_revenue_estimate: float | None
    forward_revenue_growth: float | None
    forward_ps: float | None
    required_revenue: float
    required_growth: float
    growth_gap: float | None
    years_to_normalise: float | None
    expectations_burden_score: float | None
    expectations_classification: str
    target_multiple_value: float
    target_multiple_label: str
    valuation_anchor_confidence: str
    valuation_anchor_observation_count: int
    revenue_data_cadence: str

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_confidence(n_obs: int) -> str:
    if n_obs >= 40:
        return "high"
    if n_obs >= 15:
        return "medium"
    return "low"


def _classify_expectations(burden_score: float | None) -> str:
    if burden_score is None:
        return "Insufficient data"
    if burden_score < 25:
        return "Forward Expectations Manageable"
    if burden_score < 60:
        return "Forward Expectations Elevated"
    return "Forward Expectations Stretched"


def compute_valuation(
    ticker: str,
    current_revenue_ttm: float,
    current_market_cap: float,
    historical_ps_series: pd.Series,
    forward_revenue_estimate: float | None,
    revenue_cadence: str = "quarterly",
    as_of: datetime | None = None,
) -> ValuationResult:
    """
    historical_ps_series: a pandas Series of historical trailing P/S
    observations (one per quarter, ideally), used to build the
    percentile distribution. Index doesn't matter, only values.
    """
    as_of = as_of or datetime.utcnow()

    current_ps = current_market_cap / current_revenue_ttm

    clean = historical_ps_series.dropna()
    clean = clean[clean > 0]
    n_obs = len(clean)

    hist_median = float(clean.median()) if n_obs else float("nan")
    hist_p75 = float(clean.quantile(0.75)) if n_obs else float("nan")
    hist_p90 = float(clean.quantile(0.90)) if n_obs else float("nan")

    # Target multiple: historical median is the default "tactical anchor" --
    # it represents where the market has typically priced this stock.
    target_multiple_value = hist_median
    target_multiple_label = "Historical Median Tactical Anchor"

    # Revenue required, at the CURRENT price, for P/S to fall back to the
    # target multiple -- i.e. how much the business needs to grow into the
    # price already being paid for it.
    required_revenue = current_market_cap / target_multiple_value if target_multiple_value else float("nan")
    required_growth = (required_revenue / current_revenue_ttm) - 1 if current_revenue_ttm else float("nan")

    forward_growth = None
    forward_ps = None
    growth_gap = None
    years_to_normalise = None
    burden_score = None

    if forward_revenue_estimate and forward_revenue_estimate > 0:
        forward_growth = (forward_revenue_estimate / current_revenue_ttm) - 1
        forward_ps = current_market_cap / forward_revenue_estimate

        # Negative growth_gap = analysts expect MORE growth than is needed
        # to justify today's price (favourable). Positive = the market is
        # pricing in more growth than analysts currently forecast (a stretch).
        growth_gap = required_growth - forward_growth

        # Years to normalise: how long, at the forecast growth rate,
        # until revenue reaches the "required" level.
        if forward_growth > 0:
            years_to_normalise = np.log(required_revenue / current_revenue_ttm) / np.log(1 + forward_growth)
            years_to_normalise = max(years_to_normalise, 0)

        # Burden score: 0-100 scale, higher = more growth is being
        # demanded by the current price relative to what analysts expect.
        # A simple, explainable formulation: how much of the required
        # growth is NOT already covered by the forward estimate.
        if forward_growth != 0:
            coverage = forward_growth / required_growth if required_growth else 1
            burden_score = float(np.clip(100 * (1 - min(coverage, 1)), 0, 100))
        else:
            burden_score = 100.0

    return ValuationResult(
        ticker=ticker,
        as_of=as_of.isoformat(),
        current_revenue_ttm=current_revenue_ttm,
        current_market_cap=current_market_cap,
        current_ps=current_ps,
        hist_median_ps=hist_median,
        hist_p75_ps=hist_p75,
        hist_p90_ps=hist_p90,
        forward_revenue_estimate=forward_revenue_estimate,
        forward_revenue_growth=forward_growth,
        forward_ps=forward_ps,
        required_revenue=required_revenue,
        required_growth=required_growth,
        growth_gap=growth_gap,
        years_to_normalise=years_to_normalise,
        expectations_burden_score=burden_score,
        expectations_classification=_classify_expectations(burden_score),
        target_multiple_value=target_multiple_value,
        target_multiple_label=target_multiple_label,
        valuation_anchor_confidence=_classify_confidence(n_obs),
        valuation_anchor_observation_count=n_obs,
        revenue_data_cadence=revenue_cadence,
    )
