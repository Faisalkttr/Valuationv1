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
    plain_explanation: str
    target_multiple_value: float
    target_multiple_label: str
    valuation_anchor_confidence: str
    valuation_anchor_observation_count: int
    revenue_data_cadence: str
    # Fundamentals overlay -- quality/risk context alongside the pure
    # revenue multiple, so "cheap on P/S" can be checked against
    # "actually profitable and not over-levered".
    gross_margin: float | None
    operating_margin: float | None
    net_margin: float | None
    net_debt_to_ebitda: float | None
    interest_coverage: float | None
    free_cash_flow_ttm: float | None
    fcf_margin: float | None
    cash_conversion: float | None
    quality_flag: str
    # Currency normalization metadata -- see fetch.py's _resolve_fx().
    revenue_currency: str | None
    price_currency: str | None
    fx_rate_applied: float | None
    currency_note: str

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


def _burden_score(required_growth: float, forward_growth: float) -> float:
    """
    0-100: how much of the growth "debt" implied by the current price is
    NOT already covered by the forward estimate. 0 = fully covered (or no
    growth was needed in the first place), 100 = forecast falls entirely
    short of what's required.

    Edge case this fixes: when required_growth <= 0, the stock already
    trades AT OR BELOW its historical median -- no growth is needed to
    look "normal", so there's no growth debt regardless of what's
    forecast. The naive coverage ratio (forward/required) inverts sign
    in this case and previously produced a bogus 100 ("Stretched") for
    what is actually the most favorable case the model can describe.
    """
    if required_growth <= 0:
        return 0.0
    coverage = forward_growth / required_growth
    return float(np.clip(100 * (1 - min(coverage, 1)), 0, 100))


def _plain_explanation(required_growth: float, forward_growth: float, classification: str) -> str:
    """A one-line, non-technical readout of what the numbers above actually mean."""
    req_pct = f"{required_growth * 100:.0f}%"
    fwd_pct = f"{forward_growth * 100:.0f}%"

    if required_growth <= 0:
        fwd_pct = f"{forward_growth * 100:.0f}%"
        return (
            f"Already trades at or below its historical valuation norm (needs 0% growth) -- "
            f"{fwd_pct} forecast growth would be upside on top of an already-fair price."
        )
    if classification == "Forward Expectations Manageable":
        return (
            f"Needs {req_pct} growth to look fairly valued by its own history; analysts expect "
            f"{fwd_pct} -- the bar is comfortably cleared if the forecast is anywhere close to right."
        )
    if classification == "Forward Expectations Elevated":
        return (
            f"Needs {req_pct} growth to look fairly valued; analysts expect {fwd_pct} -- covers part "
            f"of what's required, but the price still leans on the forecast coming through."
        )
    return (
        f"Needs {req_pct} growth to look fairly valued by its own history, but analysts only expect "
        f"{fwd_pct} -- the price is asking for more growth than is currently forecast."
    )


def _classify_quality(
    operating_margin: float | None,
    net_debt_to_ebitda: float | None,
    fcf_margin: float | None,
) -> str:
    """
    A simple, explainable read on whether the growth story sits on solid
    fundamentals -- deliberately conservative: any missing input just
    means "Insufficient data" rather than guessing.
    """
    if operating_margin is None and net_debt_to_ebitda is None and fcf_margin is None:
        return "Insufficient data"

    flags = []
    if operating_margin is not None and operating_margin < 0:
        flags.append("unprofitable")
    if net_debt_to_ebitda is not None and net_debt_to_ebitda > 3:
        flags.append("high leverage")
    if fcf_margin is not None and fcf_margin < 0:
        flags.append("cash burning")

    if not flags:
        return "Solid"
    if len(flags) == 1:
        return f"Watch: {flags[0]}"
    return f"Caution: {', '.join(flags)}"


def compute_valuation(
    ticker: str,
    current_revenue_ttm: float,
    current_market_cap: float,
    historical_ps_series: pd.Series,
    forward_revenue_estimate: float | None,
    revenue_cadence: str = "quarterly",
    as_of: datetime | None = None,
    gross_margin: float | None = None,
    operating_margin: float | None = None,
    net_margin: float | None = None,
    net_debt_to_ebitda: float | None = None,
    interest_coverage: float | None = None,
    free_cash_flow_ttm: float | None = None,
    fcf_margin: float | None = None,
    cash_conversion: float | None = None,
    revenue_currency: str | None = None,
    price_currency: str | None = None,
    fx_rate_applied: float | None = None,
    currency_note: str = "",
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
        # See _burden_score() docstring for the near-zero/negative
        # required_growth edge case this handles correctly.
        burden_score = _burden_score(required_growth, forward_growth)

    classification = _classify_expectations(burden_score)
    if forward_growth is None:
        explanation = (
            f"No analyst forward revenue estimate available -- would need {required_growth*100:.0f}% "
            f"growth to look fairly valued by its own history, but there's no forecast to check that against."
        )
    else:
        explanation = _plain_explanation(required_growth, forward_growth, classification)

    if currency_note and fx_rate_applied is None and revenue_currency and price_currency:
        explanation = f"⚠ {currency_note} {explanation}"

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
        expectations_classification=classification,
        plain_explanation=explanation,
        target_multiple_value=target_multiple_value,
        target_multiple_label=target_multiple_label,
        valuation_anchor_confidence=_classify_confidence(n_obs),
        valuation_anchor_observation_count=n_obs,
        revenue_data_cadence=revenue_cadence,
        gross_margin=gross_margin,
        operating_margin=operating_margin,
        net_margin=net_margin,
        net_debt_to_ebitda=net_debt_to_ebitda,
        interest_coverage=interest_coverage,
        free_cash_flow_ttm=free_cash_flow_ttm,
        fcf_margin=fcf_margin,
        cash_conversion=cash_conversion,
        quality_flag=_classify_quality(operating_margin, net_debt_to_ebitda, fcf_margin),
        revenue_currency=revenue_currency,
        price_currency=price_currency,
        fx_rate_applied=fx_rate_applied,
        currency_note=currency_note,
    )
