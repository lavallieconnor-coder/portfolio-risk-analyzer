"""Data access: Finnhub (quotes, profiles, metrics) and Yahoo Finance (history).

Finnhub's free tier does not include historical candles (/stock/candle returns
"You don't have access to this resource."), so daily price history comes from
Yahoo Finance via the yfinance package. Every function here returns data or
raises DataError with a readable message; the UI decides how to display it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
import requests

FINNHUB_URL = "https://finnhub.io/api/v1"
TIMEOUT = 10


class DataError(Exception):
    """Raised when data cannot be retrieved or is unusable."""


def get_finnhub_key() -> str | None:
    """Read the key from the environment (.env is loaded by app.py)."""
    key = os.getenv("FINNHUB_API_KEY", "").strip()
    return key or None


def _finnhub_get(path: str, params: dict, api_key: str) -> dict:
    try:
        resp = requests.get(
            f"{FINNHUB_URL}{path}", params={**params, "token": api_key}, timeout=TIMEOUT
        )
    except requests.RequestException as exc:
        raise DataError(f"Could not reach Finnhub: {exc}") from exc
    if resp.status_code == 401:
        raise DataError("Finnhub rejected the API key (401). Check FINNHUB_API_KEY.")
    if resp.status_code == 403:
        raise DataError(f"Finnhub endpoint {path} is not available on your plan (403).")
    if resp.status_code == 429:
        raise DataError("Finnhub rate limit reached (60 calls/min on free tier). Try again shortly.")
    if not resp.ok:
        raise DataError(f"Finnhub returned HTTP {resp.status_code} for {path}.")
    try:
        data = resp.json()
    except ValueError as exc:
        raise DataError("Finnhub returned a non-JSON response.") from exc
    if isinstance(data, dict) and "error" in data:
        raise DataError(f"Finnhub error: {data['error']}")
    return data


@dataclass
class Quote:
    price: float
    change: float
    change_pct: float
    prev_close: float


def finnhub_quote(symbol: str, api_key: str) -> Quote:
    d = _finnhub_get("/quote", {"symbol": symbol}, api_key)
    # Finnhub returns zeros (not an error) for unknown symbols.
    if not d or not d.get("c"):
        raise DataError(f"No Finnhub quote for {symbol}.")
    return Quote(price=d["c"], change=d.get("d") or 0.0, change_pct=d.get("dp") or 0.0, prev_close=d.get("pc") or 0.0)


def finnhub_profile(symbol: str, api_key: str) -> dict:
    """Company profile. Returns {} for ETFs, which the free endpoint doesn't cover."""
    return _finnhub_get("/stock/profile2", {"symbol": symbol}, api_key) or {}


def finnhub_metrics(symbol: str, api_key: str) -> dict:
    d = _finnhub_get("/stock/metric", {"symbol": symbol, "metric": "all"}, api_key)
    return d.get("metric", {}) if isinstance(d, dict) else {}


def download_prices(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    """Adjusted daily closes from Yahoo Finance.

    Returns (prices, missing) where `missing` lists tickers that returned no
    data (typos, delisted symbols, or Yahoo outages).
    """
    import yfinance as yf

    try:
        raw = yf.download(
            tickers, start=start, end=end + pd.Timedelta(days=1),
            auto_adjust=True, progress=False, group_by="column", threads=True,
        )
    except Exception as exc:  # yfinance raises a variety of exception types
        raise DataError(f"Price download failed: {exc}") from exc

    if raw is None or raw.empty:
        raise DataError("No price data returned. Check the tickers and your internet connection.")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(close, pd.DataFrame):
        close = close.to_frame()
    if list(close.columns) == ["Close"]:
        close.columns = tickers[:1]

    missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    close = close.drop(columns=[t for t in missing if t in close.columns])
    # Keep only dates where every remaining asset traded, so returns align.
    close = close.dropna(how="any")
    return close, missing


def risk_free_rate(default: float = 0.04) -> tuple[float, str]:
    """Latest 13-week T-bill yield (^IRX) as a decimal, with a fallback."""
    import yfinance as yf

    try:
        hist = yf.Ticker("^IRX").history(period="10d")
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1]) / 100, "13-week T-bill (^IRX, Yahoo Finance)"
    except Exception:
        pass
    return default, "default assumption (T-bill data unavailable)"
