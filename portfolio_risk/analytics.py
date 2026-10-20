"""Portfolio return and risk calculations.

All functions are pure (no network calls) so they can be unit-tested with
synthetic data. Prices are expected as a DataFrame indexed by date with one
column per ticker, containing adjusted closing prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns: r_t = P_t / P_{t-1} - 1."""
    return prices.pct_change().dropna(how="all")


def normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    """Scale non-negative weights (or dollar amounts) so they sum to 1."""
    total = sum(v for v in raw.values() if v > 0)
    if total <= 0:
        raise ValueError("At least one position must have a positive weight or amount.")
    return {k: max(v, 0.0) / total for k, v in raw.items()}


def portfolio_returns(
    returns: pd.DataFrame, weights: dict[str, float], rebalance: bool = True
) -> pd.Series:
    """Daily portfolio returns.

    rebalance=True  -> weights are reset to target every day (constant mix).
    rebalance=False -> buy-and-hold: initial weights drift with prices.
    """
    w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
    if rebalance:
        return returns.mul(w, axis=1).sum(axis=1)
    growth = (1 + returns).cumprod().mul(w, axis=1)
    value = growth.sum(axis=1)
    prev = value.shift(1).fillna(1.0)
    return value / prev - 1


def growth_of(returns: pd.Series, start_value: float = 10_000.0) -> pd.Series:
    """Value path of an investment that compounds the given returns."""
    return start_value * (1 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series) -> float:
    """Compound annual growth rate (CAGR) based on trading days elapsed."""
    n = len(returns)
    if n == 0:
        return float("nan")
    return float((1 + total_return(returns)) ** (TRADING_DAYS / n) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    """Sample standard deviation of daily returns scaled by sqrt(252)."""
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_annual: float) -> float:
    """Annualized Sharpe ratio using daily excess returns.

    Sharpe = mean(r - rf_daily) / std(r - rf_daily) * sqrt(252)
    """
    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    sd = excess.std(ddof=1)
    if np.isnan(sd) or sd < 1e-12:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free_annual: float) -> float:
    """Like Sharpe, but only penalizes downside deviation."""
    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    if np.isnan(downside) or downside < 1e-12:
        return float("nan")
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Percentage decline from the running peak of the wealth index."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline (a negative number)."""
    if returns.empty:
        return float("nan")
    return float(drawdown_series(returns).min())


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Sensitivity of the portfolio to the benchmark: cov(p, m) / var(m)."""
    joined = pd.concat([returns, benchmark], axis=1).dropna()
    if len(joined) < 2:
        return float("nan")
    cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] else float("nan")


def tracking_error(returns: pd.Series, benchmark: pd.Series) -> float:
    """Annualized volatility of the return difference vs. the benchmark."""
    diff = (returns - benchmark).dropna()
    return float(diff.std(ddof=1) * np.sqrt(TRADING_DAYS))


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """One-day historical Value at Risk, returned as a positive loss fraction."""
    if returns.empty:
        return float("nan")
    return float(-np.percentile(returns, (1 - confidence) * 100))


def risk_contributions(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Share of total portfolio variance contributed by each asset.

    RC_i = w_i * (Cov @ w)_i / (w' Cov w). Contributions sum to 1.
    """
    w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
    cov = returns.cov()
    port_var = float(w @ cov @ w)
    if port_var == 0:
        return pd.Series(0.0, index=w.index)
    return w * (cov @ w) / port_var


def herfindahl_index(weights: dict[str, float]) -> float:
    """Concentration measure: sum of squared weights (1/N = fully diversified)."""
    return float(sum(v ** 2 for v in weights.values()))


def summarize(returns: pd.Series, risk_free_annual: float, benchmark: pd.Series | None = None) -> dict:
    """Bundle the headline metrics for one return stream."""
    out = {
        "Total return": total_return(returns),
        "Annualized return": annualized_return(returns),
        "Annualized volatility": annualized_volatility(returns),
        "Sharpe ratio": sharpe_ratio(returns, risk_free_annual),
        "Sortino ratio": sortino_ratio(returns, risk_free_annual),
        "Max drawdown": max_drawdown(returns),
        "1-day VaR (95%)": historical_var(returns, 0.95),
    }
    if benchmark is not None:
        out["Beta vs. benchmark"] = beta(returns, benchmark)
        out["Tracking error"] = tracking_error(returns, benchmark)
    return out
