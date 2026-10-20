"""Input validation and API error handling.

Tests marked `network` call Finnhub / Yahoo Finance; skip them offline with
    pytest -m "not network"
"""

import os

import pandas as pd
import pytest
from dotenv import load_dotenv

from portfolio_risk import data as dt
from portfolio_risk.validation import InputError, clean_positions

load_dotenv()


def table(tickers, amounts):
    return pd.DataFrame({"Ticker": tickers, "Amount": amounts})


def test_clean_positions_normalizes_case_and_whitespace():
    rows, notes = clean_positions(table([" aapl", "msft ", "spy"], [1, 2, 3]))
    assert rows["Ticker"].tolist() == ["AAPL", "MSFT", "SPY"]
    assert notes == []


def test_fewer_than_three_positions_rejected():
    with pytest.raises(InputError, match="at least"):
        clean_positions(table(["AAPL", "MSFT"], [1, 1]))


def test_zero_amount_rows_are_dropped_before_count():
    with pytest.raises(InputError):
        clean_positions(table(["AAPL", "MSFT", "SPY"], [1, 1, 0]))


def test_negative_amount_rejected():
    with pytest.raises(InputError, match="positive"):
        clean_positions(table(["AAPL", "MSFT", "SPY"], [1, -1, 1]))


def test_invalid_symbol_characters_rejected():
    with pytest.raises(InputError, match="valid ticker"):
        clean_positions(table(["AAPL", "MS FT", "SPY"], [1, 1, 1]))


def test_duplicates_combined():
    rows, notes = clean_positions(table(["AAPL", "aapl", "MSFT", "SPY"], [1, 2, 1, 1]))
    assert rows.set_index("Ticker").loc["AAPL", "Amount"] == 3
    assert any("Duplicate" in n for n in notes)


def test_blank_and_non_numeric_cells_handled():
    rows, _ = clean_positions(table(["AAPL", None, "MSFT", "SPY", "", "QQQ"], [1, 5, "abc", 1, 1, 1]))
    # Blank tickers are skipped; MSFT's non-numeric amount becomes 0 and is dropped.
    assert rows["Ticker"].tolist() == ["AAPL", "SPY", "QQQ"]


def test_weights_not_summing_to_100_produce_notice():
    _, notes = clean_positions(table(["A", "B", "C"], [20, 20, 20]), weights_mode=True)
    assert any("rescaled" in n for n in notes)


# ------------------------------------------------------------------ network
network = pytest.mark.network


@network
def test_finnhub_bad_key_raises_readable_error():
    with pytest.raises(dt.DataError, match="(?i)api key|401"):
        dt.finnhub_quote("AAPL", "not-a-real-key")


@network
@pytest.mark.skipif(not os.getenv("FINNHUB_API_KEY"), reason="no FINNHUB_API_KEY")
def test_finnhub_unknown_symbol_raises():
    with pytest.raises(dt.DataError):
        dt.finnhub_quote("ZZZZQQ", os.environ["FINNHUB_API_KEY"])


@network
def test_download_reports_missing_ticker():
    prices, missing = dt.download_prices(["AAPL", "ZZZZQQ", "SPY"], pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01"))
    assert "ZZZZQQ" in missing
    assert {"AAPL", "SPY"} <= set(prices.columns)
    assert not prices.isna().any().any()
