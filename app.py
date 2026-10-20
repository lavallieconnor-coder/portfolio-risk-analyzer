"""Portfolio Risk Analyzer — Streamlit app.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from portfolio_risk import analytics as an
from portfolio_risk import data as dt
from portfolio_risk.validation import InputError, clean_positions

load_dotenv()

st.set_page_config(page_title="Portfolio Risk Analyzer", page_icon="📊", layout="wide")

DEFAULT_PORTFOLIO = pd.DataFrame(
    {"Ticker": ["AAPL", "MSFT", "NVDA", "JPM", "TLT"], "Amount": [3000.0, 2500.0, 1500.0, 1500.0, 1500.0]}
)
PERIODS = {"1 year": 365, "3 years": 365 * 3, "5 years": 365 * 5, "10 years": 365 * 10}
BENCHMARKS = {
    "SPY": "S&P 500 (SPY)",
    "QQQ": "Nasdaq-100 (QQQ)",
    "VTI": "US Total Market (VTI)",
    "ACWI": "Global Equity (ACWI)",
    "AOR": "60/40 Growth Allocation (AOR)",
}


# ---------------------------------------------------------------- cached I/O
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers: tuple[str, ...], start: date, end: date):
    return dt.download_prices(list(tickers), pd.Timestamp(start), pd.Timestamp(end))


@st.cache_data(ttl=3600, show_spinner=False)
def load_rf():
    return dt.risk_free_rate()


@st.cache_data(ttl=300, show_spinner=False)
def load_finnhub(symbol: str, api_key: str) -> dict:
    """Quote + profile + metrics for one symbol; errors are captured per field."""
    out: dict = {"symbol": symbol}
    for name, fn in (("quote", dt.finnhub_quote), ("profile", dt.finnhub_profile), ("metrics", dt.finnhub_metrics)):
        try:
            out[name] = fn(symbol, api_key)
        except dt.DataError as exc:
            out[name] = None
            out.setdefault("errors", []).append(str(exc))
    return out


# ---------------------------------------------------------------- formatting
def pct(x: float, digits: int = 2) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x * 100:.{digits}f}%"


def num(x: float, digits: int = 2) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x:.{digits}f}"


PCT_METRICS = {"Total return", "Annualized return", "Annualized volatility", "Max drawdown", "1-day VaR (95%)", "Tracking error"}


def fmt_metric(name: str, value: float) -> str:
    return pct(value) if name in PCT_METRICS else num(value)


# ---------------------------------------------------------------- sidebar
st.sidebar.header("1 · Build your portfolio")
input_mode = st.sidebar.radio("Enter positions as", ["Dollar amounts", "Weights (%)"], horizontal=True)
edited = st.sidebar.data_editor(
    DEFAULT_PORTFOLIO if input_mode == "Dollar amounts"
    else DEFAULT_PORTFOLIO.assign(Amount=DEFAULT_PORTFOLIO.Amount / DEFAULT_PORTFOLIO.Amount.sum() * 100),
    num_rows="dynamic",
    width="stretch",
    key=f"positions_{input_mode}",
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", help="Stock or ETF symbol, e.g. AAPL or VOO"),
        "Amount": st.column_config.NumberColumn(
            "$ Amount" if input_mode == "Dollar amounts" else "Weight %", min_value=0.0, step=100.0 if input_mode == "Dollar amounts" else 1.0
        ),
    },
)

st.sidebar.header("2 · Analysis settings")
bench = st.sidebar.selectbox("Benchmark", list(BENCHMARKS), format_func=BENCHMARKS.get)
period_label = st.sidebar.selectbox("Look-back period", list(PERIODS) + ["Custom"], index=1)
today = date.today()
if period_label == "Custom":
    start_date = st.sidebar.date_input("Start date", today - timedelta(days=365 * 3), max_value=today - timedelta(days=60))
    end_date = st.sidebar.date_input("End date", today, max_value=today)
else:
    start_date, end_date = today - timedelta(days=PERIODS[period_label]), today
rebalance = st.sidebar.radio(
    "Rebalancing", ["Buy & hold", "Daily rebalance to target"], help="Buy & hold lets weights drift with prices; rebalancing keeps them constant."
) == "Daily rebalance to target"

rf_auto, rf_source = load_rf()
rf = st.sidebar.number_input(
    "Risk-free rate (annual %)", value=round(rf_auto * 100, 2), step=0.25, format="%.2f",
    help=f"Pre-filled from the {rf_source}. Used for Sharpe and Sortino.",
) / 100

api_key = dt.get_finnhub_key()
if not api_key:
    api_key = st.sidebar.text_input("Finnhub API key (optional)", type="password",
                                    help="Needed for live quotes and company data. Get a free key at finnhub.io.") or None

# ---------------------------------------------------------------- header
st.title("📊 Portfolio Risk Analyzer")
st.caption(
    "Measure how a portfolio of stocks and ETFs has performed, how risky it has been, and where that risk "
    "comes from, compared with a market benchmark. **Educational tool — not investment advice. Past performance "
    "does not predict future results.**"
)

# ---------------------------------------------------------------- validation
try:
    rows, notices = clean_positions(edited, weights_mode=input_mode == "Weights (%)")
except InputError as exc:
    st.error(str(exc))
    st.stop()
for n in notices:
    st.info(n)
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

tickers = rows["Ticker"].tolist()
all_symbols = tuple(dict.fromkeys(tickers + [bench]))

# ---------------------------------------------------------------- data
with st.spinner("Downloading price history…"):
    try:
        prices, missing = load_prices(all_symbols, start_date, end_date)
    except dt.DataError as exc:
        st.error(f"Could not load prices: {exc}")
        st.stop()

if bench in missing:
    st.error(f"No data for the benchmark {bench}. Try another benchmark.")
    st.stop()
if missing:
    st.warning(f"No price data for **{', '.join(missing)}**; they were removed from the analysis. Check the spelling.")
    rows = rows[~rows["Ticker"].isin(missing)]
    tickers = rows["Ticker"].tolist()
if len(tickers) < 3:
    st.error("Fewer than three valid tickers remain. Please fix or add positions.")
    st.stop()
if len(prices) < 60:
    st.error(f"Only {len(prices)} overlapping trading days were found. Choose a longer period, or check for recently listed tickers.")
    st.stop()

weights = an.normalize_weights(dict(zip(rows["Ticker"], rows["Amount"])))
rets = an.daily_returns(prices)
asset_rets = rets[tickers]
bench_rets = rets[bench]
port_rets = an.portfolio_returns(asset_rets, weights, rebalance=rebalance)

port_stats = an.summarize(port_rets, rf, bench_rets)
bench_stats = an.summarize(bench_rets, rf, bench_rets)
first_day, last_day = prices.index[0].date(), prices.index[-1].date()
st.caption(f"Analysis window: **{first_day} → {last_day}** · {len(rets)} trading days · daily adjusted closes · "
           f"risk-free rate {pct(rf)} · benchmark {bench}")

# ---------------------------------------------------------------- KPI row
k = st.columns(5)
k[0].metric("Annualized return", pct(port_stats["Annualized return"]),
            f"{(port_stats['Annualized return'] - bench_stats['Annualized return']) * 100:+.2f} pts vs {bench}")
k[1].metric("Annualized volatility", pct(port_stats["Annualized volatility"]),
            f"{(port_stats['Annualized volatility'] - bench_stats['Annualized volatility']) * 100:+.2f} pts vs {bench}",
            delta_color="inverse")
k[2].metric("Sharpe ratio", num(port_stats["Sharpe ratio"]),
            f"{port_stats['Sharpe ratio'] - bench_stats['Sharpe ratio']:+.2f} vs {bench}")
k[3].metric("Max drawdown", pct(port_stats["Max drawdown"]),
            f"{(port_stats['Max drawdown'] - bench_stats['Max drawdown']) * 100:+.2f} pts vs {bench}")
k[4].metric("Beta vs. benchmark", num(port_stats["Beta vs. benchmark"]))

tab_perf, tab_risk, tab_hold, tab_insight, tab_method = st.tabs(
    ["Performance", "Risk breakdown", "Holdings (live)", "Assessment", "Methodology"]
)

# ---------------------------------------------------------------- performance
with tab_perf:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Growth of $10,000")
        growth = pd.DataFrame({"Portfolio": an.growth_of(port_rets), bench: an.growth_of(bench_rets)})
        fig = px.line(growth, labels={"value": "Value ($)", "index": "", "variable": ""})
        fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.08), margin=dict(t=30))
        fig.update_yaxes(tickprefix="$", separatethousands=True)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Allocation")
        alloc = pd.DataFrame({"Ticker": list(weights), "Weight": list(weights.values())})
        fig = px.pie(alloc, names="Ticker", values="Weight", hole=0.45)
        fig.update_traces(textinfo="label+percent")
        fig.update_layout(showlegend=False, margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Portfolio vs. benchmark")
    table = pd.DataFrame({"Portfolio": port_stats, bench: bench_stats})
    st.dataframe(
        table.apply(lambda col: [fmt_metric(m, v) for m, v in col.items()]),
        width="stretch",
    )

    st.subheader("Individual asset statistics")
    per_asset = pd.DataFrame({t: an.summarize(asset_rets[t], rf, bench_rets) for t in tickers}).T
    per_asset.insert(0, "Weight", pd.Series(weights))
    st.dataframe(
        per_asset.apply(lambda col: [pct(v) if col.name in PCT_METRICS | {"Weight"} else num(v) for v in col]),
        width="stretch",
    )

# ---------------------------------------------------------------- risk
with tab_risk:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Drawdown from peak")
        dd = pd.DataFrame({"Portfolio": an.drawdown_series(port_rets), bench: an.drawdown_series(bench_rets)})
        fig = px.area(dd, labels={"value": "Drawdown", "index": "", "variable": ""})
        fig.update_traces(stackgroup=None, fill="tozeroy")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.08), margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Rolling 63-day volatility (annualized)")
        roll = pd.DataFrame({
            "Portfolio": port_rets.rolling(63).std() * np.sqrt(an.TRADING_DAYS),
            bench: bench_rets.rolling(63).std() * np.sqrt(an.TRADING_DAYS),
        }).dropna()
        fig = px.line(roll, labels={"value": "Volatility", "index": "", "variable": ""})
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.08), margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    rc = an.risk_contributions(asset_rets, weights)
    with c1:
        st.subheader("Capital weight vs. risk contribution")
        rc_df = pd.DataFrame({"Capital weight": pd.Series(weights), "Share of portfolio risk": rc}).reset_index(names="Ticker")
        fig = px.bar(rc_df.melt(id_vars="Ticker", var_name="Measure", value_name="Share"),
                     x="Ticker", y="Share", color="Measure", barmode="group")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(legend=dict(orientation="h", y=1.1, title=None), margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")
        st.caption("When an asset's risk share is well above its capital weight, it is driving more of the "
                   "portfolio's ups and downs than its dollar size suggests.")
    with c2:
        st.subheader("Correlation of daily returns")
        corr = asset_rets.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Risk vs. return by asset")
    rr = pd.DataFrame({
        "Annualized volatility": [an.annualized_volatility(asset_rets[t]) for t in tickers] +
                                 [port_stats["Annualized volatility"], bench_stats["Annualized volatility"]],
        "Annualized return": [an.annualized_return(asset_rets[t]) for t in tickers] +
                             [port_stats["Annualized return"], bench_stats["Annualized return"]],
        "Name": tickers + ["Portfolio", bench],
        "Type": ["Asset"] * len(tickers) + ["Portfolio", "Benchmark"],
    })
    fig = px.scatter(rr, x="Annualized volatility", y="Annualized return", text="Name", color="Type", size_max=14)
    fig.update_traces(textposition="top center", marker=dict(size=13))
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(margin=dict(t=30), legend=dict(orientation="h", y=1.08, title=None))
    st.plotly_chart(fig, width="stretch")
    st.caption("The portfolio usually sits to the left of a straight weighted average of its assets: imperfect "
               "correlation means diversification lowers volatility without giving up the average return.")

# ---------------------------------------------------------------- holdings (Finnhub)
with tab_hold:
    st.subheader("Live quotes and company data")
    if not api_key:
        st.info("Add a Finnhub API key (in `.env` or the sidebar) to see live quotes, company profiles and "
                "Finnhub's fundamental metrics. All other analysis works without it.")
    else:
        recs, errors = [], []
        with st.spinner("Fetching Finnhub data…"):
            for t in tickers + [bench]:
                info = load_finnhub(t, api_key)
                q, p, m = info.get("quote"), info.get("profile") or {}, info.get("metrics") or {}
                errors += info.get("errors", [])
                recs.append({
                    "Ticker": t,
                    "Name": p.get("name") or "—",
                    "Industry": p.get("finnhubIndustry") or ("ETF / fund" if not p else "—"),
                    "Price": q.price if q else np.nan,
                    "Change %": q.change_pct / 100 if q else np.nan,
                    "Market cap ($B)": p.get("marketCapitalization", np.nan) / 1000 if p.get("marketCapitalization") else np.nan,
                    "Finnhub beta": m.get("beta", np.nan),
                    "P/E (TTM)": m.get("peTTM", np.nan),
                    "52-wk high": m.get("52WeekHigh", np.nan),
                    "52-wk low": m.get("52WeekLow", np.nan),
                    "Weight": weights.get(t, np.nan),
                })
        hold = pd.DataFrame(recs).set_index("Ticker")
        st.dataframe(
            hold, width="stretch",
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Change %": st.column_config.NumberColumn(format="percent"),
                "Market cap ($B)": st.column_config.NumberColumn(format="%.1f"),
                "Finnhub beta": st.column_config.NumberColumn(format="%.2f"),
                "P/E (TTM)": st.column_config.NumberColumn(format="%.1f"),
                "52-wk high": st.column_config.NumberColumn(format="$%.2f"),
                "52-wk low": st.column_config.NumberColumn(format="$%.2f"),
                "Weight": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.caption("Source: Finnhub /quote, /stock/profile2 and /stock/metric. Quotes may be delayed. "
                   "Company profile and fundamentals are not available for ETFs on the free tier.")
        if errors:
            with st.expander(f"{len(errors)} Finnhub request(s) returned no data"):
                for e in dict.fromkeys(errors):
                    st.write("•", e)

        # Position values at today's prices, when amounts were entered in dollars.
        if input_mode == "Dollar amounts":
            st.subheader("Position value if bought at the start of the window")
            start_px = prices[tickers].iloc[0]
            invested = rows.set_index("Ticker")["Amount"]
            shares = invested / start_px
            now_px = hold.loc[tickers, "Price"].fillna(prices[tickers].iloc[-1])
            pos = pd.DataFrame({"Invested": invested, "Shares": shares, "Current price": now_px,
                                "Current value": shares * now_px})
            pos["Gain / loss"] = pos["Current value"] - pos["Invested"]
            pos.loc["Total"] = [pos["Invested"].sum(), np.nan, np.nan, pos["Current value"].sum(), pos["Gain / loss"].sum()]
            st.dataframe(pos, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="$%.2f") for c in ["Invested", "Current price", "Current value", "Gain / loss"]
            } | {"Shares": st.column_config.NumberColumn(format="%.3f")})


# ---------------------------------------------------------------- assessment
def build_assessment() -> list[str]:
    notes = []
    pv, bv = port_stats["Annualized volatility"], bench_stats["Annualized volatility"]
    ps, bs = port_stats["Sharpe ratio"], bench_stats["Sharpe ratio"]
    pm, bm = port_stats["Max drawdown"], bench_stats["Max drawdown"]
    pr, br = port_stats["Annualized return"], bench_stats["Annualized return"]

    notes.append(
        f"**Return.** The portfolio compounded at {pct(pr)} per year vs. {pct(br)} for {bench}, "
        f"{'outperforming' if pr > br else 'underperforming'} by {abs(pr - br) * 100:.2f} percentage points."
    )
    rel = pv / bv - 1
    notes.append(
        f"**Volatility.** Annualized volatility was {pct(pv)}, which is {abs(rel) * 100:.0f}% "
        f"{'higher' if rel > 0 else 'lower'} than the benchmark's {pct(bv)}."
    )
    notes.append(
        f"**Risk-adjusted return.** The Sharpe ratio of {num(ps)} vs. {num(bs)} means the portfolio earned "
        f"{'more' if ps > bs else 'less'} excess return per unit of risk than {bench}."
    )
    notes.append(
        f"**Worst decline.** The largest peak-to-trough loss was {pct(pm)} (benchmark: {pct(bm)}). "
        f"An investor with \\$10,000 at the peak would have seen it fall to about \\${10_000 * (1 + pm):,.0f}."
    )
    b = port_stats["Beta vs. benchmark"]
    notes.append(
        f"**Market sensitivity.** A beta of {num(b)} implies the portfolio tended to move about "
        f"{b:.2f}× as much as {bench} on a given day."
    )

    excess_risk = rc - pd.Series(weights)
    top = excess_risk.idxmax()
    if excess_risk[top] > 0.05:
        notes.append(
            f"**Concentration of risk.** {top} is {pct(weights[top], 0)} of the capital but drives "
            f"{pct(rc[top], 0)} of portfolio variance, more than any other holding relative to its size."
        )
    hhi = an.herfindahl_index(weights)
    notes.append(
        f"**Diversification.** The Herfindahl index of weights is {hhi:.2f} (1/N = {1 / len(weights):.2f} for equal weights), "
        f"which is equivalent to about {1 / hhi:.1f} equally sized positions."
    )
    corr_vals = asset_rets.corr().where(np.triu(np.ones((len(tickers),) * 2, dtype=bool), k=1)).stack()
    high = corr_vals[corr_vals > 0.7]
    if not high.empty:
        pairs = ", ".join(f"{a}/{b_} ({v:.2f})" for (a, b_), v in high.sort_values(ascending=False).items())
        notes.append(f"**Overlapping exposures.** Highly correlated pairs: {pairs}. These positions offer limited diversification against each other.")
    elif not corr_vals.empty:
        notes.append(f"**Overlapping exposures.** No pair of holdings has a correlation above 0.70 (highest: {corr_vals.max():.2f}).")
    notes.append(
        f"**Daily loss risk.** On 95% of days the portfolio lost no more than {pct(port_stats['1-day VaR (95%)'])} "
        f"(historical 1-day VaR), so the worst 5% of days were larger losses than that."
    )
    return notes


with tab_insight:
    st.subheader("Data-supported assessment")
    for line in build_assessment():
        st.markdown(f"- {line}")
    st.caption("These sentences are generated from the numbers above with fixed rules; they describe the past and "
               "are not a forecast or a recommendation. Try changing weights in the sidebar to see how they change.")

# ---------------------------------------------------------------- methodology
with tab_method:
    st.markdown(
        f"""
**Data**
- Daily adjusted closing prices (split- and dividend-adjusted) from Yahoo Finance via `yfinance`.
  Finnhub's historical candle endpoint is not part of its free tier.
- Live quotes, company profiles and fundamental metrics from the Finnhub REST API.
- Risk-free rate: latest 13-week US T-bill yield (`^IRX`), editable in the sidebar.
- Only dates on which **all** selected assets traded are used, so returns line up.

**Calculations** (r = daily simple return, 252 trading days per year)
| Metric | Formula |
|---|---|
| Portfolio return | buy & hold: value-weighted with drifting weights; rebalanced: Σ wᵢ·rᵢ each day |
| Total return | Π(1 + r) − 1 |
| Annualized return (CAGR) | (1 + total)^(252/N) − 1 |
| Annualized volatility | stdev(r) × √252 |
| Sharpe ratio | mean(r − r_f) / stdev(r − r_f) × √252 |
| Sortino ratio | mean(r − r_f) / downside deviation × √252 |
| Max drawdown | min over t of (Wₜ / max₍ₛ≤ₜ₎ Wₛ − 1) |
| Beta | cov(r_p, r_m) / var(r_m) |
| Tracking error | stdev(r_p − r_m) × √252 |
| Historical VaR (95%) | −5th percentile of daily returns |
| Risk contribution | wᵢ·(Σw)ᵢ / (wᵀΣw) |
| Herfindahl index | Σ wᵢ² |

**Limitations**
- Historical statistics are backward-looking and sensitive to the chosen window.
- Volatility and VaR assume past return distributions are representative; tail events can be larger.
- No transaction costs, taxes, or cash drag are modeled. Short positions and leverage are not supported.
- Yahoo Finance is an unofficial data source and may occasionally be unavailable or rate-limited.
        """
    )
