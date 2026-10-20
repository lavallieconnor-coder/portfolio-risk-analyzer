"""Checks the financial calculations against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import analytics as an


@pytest.fixture
def prices():
    idx = pd.bdate_range("2024-01-01", periods=5)
    return pd.DataFrame({"A": [100, 110, 99, 108.9, 119.79], "B": [50, 50, 55, 55, 49.5]}, index=idx)


def test_daily_returns(prices):
    r = an.daily_returns(prices)
    assert r["A"].tolist() == pytest.approx([0.10, -0.10, 0.10, 0.10])
    assert r["B"].tolist() == pytest.approx([0.0, 0.10, 0.0, -0.10])


def test_normalize_weights():
    assert an.normalize_weights({"A": 3000, "B": 1000}) == pytest.approx({"A": 0.75, "B": 0.25})
    with pytest.raises(ValueError):
        an.normalize_weights({"A": 0, "B": 0})


def test_rebalanced_portfolio_is_weighted_average(prices):
    r = an.daily_returns(prices)
    p = an.portfolio_returns(r, {"A": 0.5, "B": 0.5}, rebalance=True)
    assert p.tolist() == pytest.approx([0.05, 0.0, 0.05, 0.0])


def test_buy_and_hold_matches_final_value(prices):
    r = an.daily_returns(prices)
    p = an.portfolio_returns(r, {"A": 0.5, "B": 0.5}, rebalance=False)
    # $0.5 in A grows to 0.5*119.79/100, $0.5 in B to 0.5*49.5/50
    expected_final = 0.5 * 1.1979 + 0.5 * 0.99
    assert (1 + p).prod() == pytest.approx(expected_final)


def test_total_and_annualized_return():
    r = pd.Series([0.01] * 252)
    assert an.total_return(r) == pytest.approx(1.01 ** 252 - 1)
    assert an.annualized_return(r) == pytest.approx(1.01 ** 252 - 1)
    # Half a year of data annualizes by squaring the growth factor.
    half = pd.Series([0.01] * 126)
    assert an.annualized_return(half) == pytest.approx(1.01 ** 252 - 1)


def test_volatility_scaling():
    r = pd.Series([0.01, -0.01] * 50)
    assert an.annualized_volatility(r) == pytest.approx(r.std(ddof=1) * np.sqrt(252))


def test_sharpe_zero_rf():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    assert an.sharpe_ratio(r, 0.0) == pytest.approx(r.mean() / r.std() * np.sqrt(252))


def test_sharpe_constant_returns_is_nan():
    assert np.isnan(an.sharpe_ratio(pd.Series([0.001] * 10), 0.0))


def test_max_drawdown():
    # 100 -> 120 -> 90 -> 130: worst decline is 120 -> 90 = -25%
    wealth = pd.Series([100, 120, 90, 130])
    r = wealth.pct_change().dropna()
    assert an.max_drawdown(r) == pytest.approx(-0.25)


def test_beta_of_scaled_series():
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(0, 0.01, 500))
    assert an.beta(2 * m, m) == pytest.approx(2.0)
    assert an.beta(m, m) == pytest.approx(1.0)


def test_risk_contributions_sum_to_one():
    rng = np.random.default_rng(2)
    r = pd.DataFrame(rng.normal(0, 0.01, (500, 3)), columns=list("XYZ"))
    rc = an.risk_contributions(r, {"X": 0.5, "Y": 0.3, "Z": 0.2})
    assert rc.sum() == pytest.approx(1.0)


def test_equal_weight_uncorrelated_risk_parity():
    # Identical independent volatility + equal weights -> roughly equal risk shares.
    rng = np.random.default_rng(3)
    r = pd.DataFrame(rng.normal(0, 0.01, (20000, 4)), columns=list("ABCD"))
    rc = an.risk_contributions(r, {c: 0.25 for c in "ABCD"})
    assert rc.values == pytest.approx([0.25] * 4, abs=0.02)


def test_historical_var():
    r = pd.Series(np.linspace(-0.05, 0.05, 101))
    assert an.historical_var(r, 0.95) == pytest.approx(0.045)


def test_herfindahl():
    assert an.herfindahl_index({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}) == pytest.approx(0.25)
    assert an.herfindahl_index({"A": 1.0}) == pytest.approx(1.0)
