"""Static pre-2021 CBOT corn/soybean settlement history, bundled with the app
to backfill contract years older than Massive's ~2021-09-02 data floor.

Extracted once from the user's own JSA reference workbook ("Futures
History.xlsx") via scripts/extract_legacy_history.py into
legacy_futures_history.csv — corn (ZC) and soybeans (ZS) only, 2008-2021,
each contract's last ~300 trading sessions before expiration (settlement
price only, no OHLC — so this can backfill the seasonal overlay/spread/
harmonic charts but not the Contract History tab's high/low-by-month, which
needs real intraday highs and lows). No wheat sheets exist in the source
file, so ZW/KE get nothing from here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CSV_PATH = HERE / "legacy_futures_history.csv"

LEGACY_COLUMNS = ["product_code", "month_letter", "contract_year", "date", "settle"]


def load_legacy_history() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=LEGACY_COLUMNS)
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df


def get_legacy_series(history: pd.DataFrame, product_code: str, month_letter: str, year: int) -> pd.Series:
    """Settle series for one specific contract (e.g. ZC, 'Z', 2016), indexed
    by date. Empty if this file doesn't cover that product/month/year."""
    if history.empty:
        return pd.Series(dtype=float)
    sub = history[
        (history["product_code"] == product_code)
        & (history["month_letter"] == month_letter)
        & (history["contract_year"] == year)
    ]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.set_index("date")["settle"].sort_index()
