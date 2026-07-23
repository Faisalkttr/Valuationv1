"""
Streamlit dashboard for the valuation engine.

Reads data/latest.csv (written by engine/run.py, refreshed daily by
GitHub Actions) -- no live network calls happen here, so the dashboard
loads instantly even with 50+ tickers.

Run locally with:  streamlit run app.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
LATEST_CSV = ROOT / "data" / "latest.csv"
HISTORY_DB = ROOT / "data" / "history.db"

st.set_page_config(page_title="Valuation watchlist", layout="wide")


@st.cache_data(ttl=3600)
def load_latest() -> pd.DataFrame:
    if not LATEST_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(LATEST_CSV)


@st.cache_data(ttl=3600)
def load_history(ticker: str) -> pd.DataFrame:
    if not HISTORY_DB.exists():
        return pd.DataFrame()
    with sqlite3.connect(HISTORY_DB) as conn:
        df = pd.read_sql(
            "SELECT * FROM valuation_snapshots WHERE ticker = ? ORDER BY as_of",
            conn, params=(ticker,),
        )
    if not df.empty:
        df["as_of"] = pd.to_datetime(df["as_of"])
    return df


df = load_latest()

st.title("Valuation watchlist")

if df.empty:
    st.info(
        "No data yet. Run `python -m engine.run` locally, or trigger the "
        "'Refresh valuation data' workflow from the GitHub Actions tab."
    )
    st.stop()

tab_overview, tab_detail = st.tabs(["Watchlist overview", "Ticker detail"])

with tab_overview:
    st.caption(f"{len(df)} tickers · last refreshed {df['as_of'].max()}")

    view = df[[
        "ticker", "current_ps", "hist_median_ps", "forward_ps",
        "forward_revenue_growth", "growth_gap", "expectations_classification",
    ]].copy()
    view["forward_revenue_growth"] = (view["forward_revenue_growth"] * 100).round(1)
    view["growth_gap"] = (view["growth_gap"] * 100).round(1)
    view = view.rename(columns={
        "current_ps": "Current P/S",
        "hist_median_ps": "Median P/S",
        "forward_ps": "Forward P/S",
        "forward_revenue_growth": "Fwd growth %",
        "growth_gap": "Growth gap %",
        "expectations_classification": "Classification",
    })

    sort_col = st.selectbox("Sort by", view.columns[1:], index=5)
    view = view.sort_values(sort_col, ascending=False)
    st.dataframe(view, use_container_width=True, hide_index=True)

with tab_detail:
    ticker = st.selectbox("Ticker", sorted(df["ticker"].unique()))
    row = df[df["ticker"] == ticker].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current P/S", f"{row['current_ps']:.2f}x")
    col2.metric("Historical median", f"{row['hist_median_ps']:.2f}x")
    col3.metric("Forward P/S", f"{row['forward_ps']:.2f}x" if pd.notna(row["forward_ps"]) else "n/a")
    col4.metric("Classification", row["expectations_classification"])

    fig = go.Figure(go.Bar(
        x=["Median", "75th pct", "Current", "90th pct", "Forward"],
        y=[row["hist_median_ps"], row["hist_p75_ps"], row["current_ps"],
           row["hist_p90_ps"], row["forward_ps"]],
        marker_color=["#898781", "#898781", "#2a78d6", "#898781", "#1baf7a"],
    ))
    fig.update_layout(yaxis_title="Price / Sales (x)", height=350, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    hist = load_history(ticker)
    if len(hist) > 1:
        st.subheader("P/S over time")
        line = go.Figure(go.Scatter(x=hist["as_of"], y=hist["current_ps"], mode="lines+markers"))
        line.update_layout(yaxis_title="Current P/S (x)", height=300, margin=dict(t=20))
        st.plotly_chart(line, use_container_width=True)
    else:
        st.caption("Trend chart appears once the engine has run more than once.")
