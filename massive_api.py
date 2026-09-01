"""Client for the Massive (api.massive.com) futures REST API."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd
import requests

BASE_URL = "https://api.massive.com/futures/v1"

_MONTH_CODES = "FGHJKMNQUVXZ"
_TICKER_RE = re.compile(r"^([A-Z]{1,3})([FGHJKMNQUVXZ])(\d)$")


class MassiveApiError(RuntimeError):
    pass


def _get(path: str, api_key: str, params: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(f"{BASE_URL}{path}", headers=headers, params=params or {}, timeout=20)
    if resp.status_code != 200:
        raise MassiveApiError(f"Massive API {path} returned {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    if data.get("status") not in ("OK", None):
        raise MassiveApiError(f"Massive API {path} error: {data}")
    return data


def _is_outright_ticker(ticker: str, product_code: str) -> bool:
    """Plain single-month contracts only (no spreads/butterflies/combos)."""
    if not ticker.startswith(product_code):
        return False
    return bool(_TICKER_RE.match(ticker))


def get_active_contract_tickers(product_code: str, api_key: str, as_of: date, limit: int = 400) -> list[dict]:
    """Return outright contract tickers + settlement dates for a product, nearest first."""
    data = _get(
        "/contracts",
        api_key,
        params={
            "product_code": product_code,
            "active": "true",
            "date": as_of.isoformat(),
            "limit": limit,
        },
    )
    seen = {}
    for r in data.get("results", []):
        ticker = r.get("ticker", "")
        if not _is_outright_ticker(ticker, product_code):
            continue
        settlement = r.get("settlement_date") or r.get("last_trade_date")
        if not settlement:
            continue
        seen[ticker] = settlement
    rows = [{"ticker": t, "expiration": d} for t, d in seen.items()]
    rows.sort(key=lambda r: r["expiration"])
    return rows


def get_snapshots(tickers: list[str], api_key: str) -> dict[str, dict]:
    """Return {ticker: snapshot} for the given outright tickers."""
    if not tickers:
        return {}
    out: dict[str, dict] = {}
    # keep query strings reasonable
    for i in range(0, len(tickers), 25):
        chunk = tickers[i : i + 25]
        data = _get("/snapshot", api_key, params={"ticker.any_of": ",".join(chunk), "limit": len(chunk)})
        for r in data.get("results", []):
            t = r.get("details", {}).get("ticker")
            if t:
                out[t] = r
    return out


def _extract_price(snapshot: dict) -> float | None:
    for path in (
        ("last_trade", "price"),
        ("session", "settlement_price"),
        ("session", "close"),
        ("last_quote", "bid"),
    ):
        node = snapshot
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, (int, float)) and node:
            return float(node)
    return None


def get_futures_curve(product_code: str, api_key: str, as_of: date, n_contracts: int = 8) -> pd.DataFrame:
    """Live futures curve for a product: ticker, expiration, last price — nearest N months."""
    contracts = get_active_contract_tickers(product_code, api_key, as_of)[:n_contracts]
    tickers = [c["ticker"] for c in contracts]
    snaps = get_snapshots(tickers, api_key)

    rows = []
    for c in contracts:
        snap = snaps.get(c["ticker"], {})
        price = _extract_price(snap)
        if price is None:
            continue
        rows.append(
            {
                "ticker": c["ticker"],
                "expiration": pd.to_datetime(c["expiration"]).date(),
                "price": price,
            }
        )
    return pd.DataFrame(rows)


BAR_COLUMNS = ["open", "high", "low", "close", "settle", "volume"]


def get_daily_bars(ticker: str, api_key: str) -> pd.DataFrame:
    """Full daily OHLC + settlement + volume history for one contract, indexed by
    trading date (ascending). 'settle' falls back to the session close when
    settlement is missing or zero — the current session's bar carries
    settlement_price 0 until it settles.
    """
    data = _get(f"/aggs/{ticker}", api_key, params={"resolution": "1session", "limit": 50000})
    rows = []
    for bar in data.get("results", []):
        day = bar.get("session_end_date")
        o, h, l, c = bar.get("open"), bar.get("high"), bar.get("low"), bar.get("close")
        settle = bar.get("settlement_price") or c
        if not day or not settle or o is None or h is None or l is None or c is None:
            continue
        rows.append({
            "date": pd.to_datetime(day).date(),
            "open": float(o), "high": float(h), "low": float(l), "close": float(c),
            "settle": float(settle), "volume": bar.get("volume"),
        })
    if not rows:
        return pd.DataFrame(columns=BAR_COLUMNS)
    return pd.DataFrame(rows).set_index("date").sort_index()


def get_daily_bars_many(tickers: list[str], api_key: str, max_workers: int = 8) -> dict[str, pd.DataFrame]:
    """Daily OHLC bars for several contracts at once. Fetched in parallel — each
    call is ~5s of round trip, so a curve's worth in sequence is painfully slow."""
    out: dict[str, pd.DataFrame] = {}
    if not tickers:
        return out
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tickers))) as pool:
        futures = {pool.submit(get_daily_bars, t, api_key): t for t in tickers}
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                out[ticker] = fut.result()
            except (MassiveApiError, requests.RequestException):
                out[ticker] = pd.DataFrame(columns=BAR_COLUMNS)
    return out


FED_FUNDS_PRODUCT = "ZQ"


def get_fed_funds_rate(api_key: str, as_of: date) -> dict:
    """Front-month CME 30-Day Federal Funds futures (ZQ) as an implied rate.

    ZQ settles against the average daily effective fed funds rate for its contract
    month, quoted as 100 - rate, so the front contract is the market's read on the
    current fed funds rate. Needs a high contract limit: ZQ's /contracts response is
    dominated by butterfly/spread combos, and a small limit truncates the outrights
    before the nearby months appear.
    """
    contracts = get_active_contract_tickers(FED_FUNDS_PRODUCT, api_key, as_of, limit=1000)
    if not contracts:
        raise MassiveApiError("No active ZQ (fed funds) contracts returned.")
    front = contracts[0]
    snaps = get_snapshots([front["ticker"]], api_key)
    price = _extract_price(snaps.get(front["ticker"], {}))
    if price is None:
        raise MassiveApiError(f"No price available for fed funds contract {front['ticker']}.")
    return {
        "ticker": front["ticker"],
        "expiration": pd.to_datetime(front["expiration"]).date(),
        "price": float(price),
        "rate_pct": round(100.0 - float(price), 3),
    }
