"""Cleaning and validating the positions the user types into the sidebar."""

from __future__ import annotations

import re

import pandas as pd

TICKER_RE = re.compile(r"^[A-Z0-9.\-\^=]{1,12}$")
MIN_POSITIONS = 3


class InputError(ValueError):
    """User input that the analysis cannot proceed with."""


def clean_positions(table: pd.DataFrame, weights_mode: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Return (positions, notices).

    positions has columns Ticker (upper-case, unique) and Amount (> 0).
    notices are non-fatal messages to show the user. Raises InputError for
    problems the user must fix.
    """
    notices: list[str] = []
    rows = table.dropna(subset=["Ticker"]).copy()
    rows["Ticker"] = rows["Ticker"].astype(str).str.strip().str.upper()
    rows["Amount"] = pd.to_numeric(rows["Amount"], errors="coerce").fillna(0.0)
    rows = rows[rows["Ticker"] != ""]

    bad = [t for t in rows["Ticker"] if not TICKER_RE.match(t)]
    if bad:
        raise InputError(f"These don't look like valid ticker symbols: {', '.join(bad)}")
    if (rows["Amount"] < 0).any():
        raise InputError("Amounts and weights must be zero or positive (short positions are not supported).")
    if rows["Ticker"].duplicated().any():
        notices.append("Duplicate tickers were combined.")
        rows = rows.groupby("Ticker", as_index=False, sort=False)["Amount"].sum()

    rows = rows[rows["Amount"] > 0].reset_index(drop=True)
    if len(rows) < MIN_POSITIONS:
        raise InputError(f"Please enter at least **{MIN_POSITIONS}** positions with a positive amount or weight.")
    if weights_mode and abs(rows["Amount"].sum() - 100) > 0.5:
        notices.append(f"Weights add up to {rows['Amount'].sum():.1f}%, so they were rescaled to 100%.")
    return rows[["Ticker", "Amount"]], notices
