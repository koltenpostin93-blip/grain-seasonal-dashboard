"""WASDE and NASS report release-date lookups.

WASDE dates come from USDA's ESMIS report archive (esmis.nal.usda.gov) — a
free, unauthenticated API that is the authoritative release record, since
WASDE does get rescheduled (e.g. Oct 2025's report was cancelled outright by
the government shutdown and folded into a delayed Nov 14 release alongside
Crop Production). A fixed "2nd Friday of the month" formula would have missed
that.

NASS dates come from NASS's own published .ics release calendar, which starts
in 2022 — Sep-Dec 2021 (the tail our futures history needs) is backfilled
from the well-documented release cadence instead of scraping the old
PDF/Newsroom archive.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd
import requests

WASDE_URL = "https://esmis.nal.usda.gov/api/v1/release/findByIdentifier/wasde"
NASS_ICS_URL = "https://www.nass.usda.gov/Publications/Calendar/{year}/NassReleases{year}.ics"

# Grain-relevant NASS reports, matched case-insensitively as substrings
# against each .ics VEVENT's SUMMARY line.
NASS_REPORT_KEYWORDS = {
    "Crop Production": ["crop production"],
    "Grain Stocks": ["grain stocks"],
    "Acreage": ["acreage"],
    "Prospective Plantings": ["prospective plantings"],
    "Winter Wheat & Canola Seedings": ["winter wheat"],
    "Small Grains Summary": ["small grains summary"],
    "Crop Progress": ["crop progress"],
}

# NASS's cadence is consistent enough to hardcode this short a backfill with
# high confidence (see report_dates.py's module docstring).
NASS_2021_FALLBACK: list[tuple[date, str]] = [
    (date(2021, 9, 10), "Crop Production"),
    (date(2021, 9, 30), "Grain Stocks"),
    (date(2021, 9, 30), "Small Grains Summary"),
    (date(2021, 10, 12), "Crop Production"),
    (date(2021, 11, 9), "Crop Production"),
    (date(2021, 12, 9), "Crop Production"),
]


class ReportDatesError(RuntimeError):
    pass


def _fetch_wasde_page(page: int) -> dict:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(WASDE_URL, params={"page": page}, timeout=8)
        except requests.RequestException as e:
            last_error = e
            time.sleep(0.3 * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp.json()
        last_error = ReportDatesError(f"ESMIS API returned {resp.status_code} for page {page}")
        time.sleep(0.3 * (attempt + 1))
    raise last_error or ReportDatesError(f"ESMIS API request failed for page {page}")


def get_wasde_dates(max_pages: int = 10) -> list[date]:
    """WASDE release dates, most recent first. ESMIS's archive runs back to
    1995 (~700 records across ~30 pages) but our futures data only needs
    2021 onward — `max_pages` (25 records/page, page 0 = most recent) caps
    the fetch to a multi-decade safety margin instead of the full archive,
    and pages are fetched in parallel on top of that."""
    first = _fetch_wasde_page(0)
    dates: list[date] = [
        pd.to_datetime(rec["release_datetime"]).date()
        for rec in first.get("results", []) if rec.get("release_datetime")
    ]
    total_pages = min(first.get("pager", {}).get("total_pages", 1), max_pages)

    remaining = range(1, total_pages)
    if remaining:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch_wasde_page, p): p for p in remaining}
            for fut in as_completed(futures):
                data = fut.result()
                for rec in data.get("results", []):
                    dt = rec.get("release_datetime")
                    if dt:
                        dates.append(pd.to_datetime(dt).date())
    return sorted(set(dates))


_ICS_EVENT_RE = re.compile(r"BEGIN:VEVENT(.*?)END:VEVENT", re.S)
_ICS_SUMMARY_RE = re.compile(r"^SUMMARY:(.*)$", re.M)
_ICS_DTSTART_RE = re.compile(r"^DTSTART[^:]*:(\d{8})", re.M)


def _parse_ics_events(text: str) -> list[tuple[date, str]]:
    events = []
    for block in _ICS_EVENT_RE.findall(text):
        summary_m = _ICS_SUMMARY_RE.search(block)
        dtstart_m = _ICS_DTSTART_RE.search(block)
        if not summary_m or not dtstart_m:
            continue
        d = dtstart_m.group(1)
        events.append((date(int(d[:4]), int(d[4:6]), int(d[6:8])), summary_m.group(1).strip()))
    return events


def _decode_ics(raw: bytes) -> str:
    """NASS serves most years' .ics as UTF-8 but at least one (2022) as
    UTF-16 with no charset header to tell them apart — sniff by checking
    which decode actually produces a real calendar."""
    for encoding in ("utf-8", "utf-16"):
        try:
            text = raw.decode(encoding)
        except UnicodeError:
            continue
        if "BEGIN:VCALENDAR" in text:
            return text
    return raw.decode("utf-8", errors="replace")


def _fetch_nass_year(year: int) -> str | None:
    """nass.usda.gov's TLS front end drops the handshake for Python's
    ssl/urllib3 stack every time (SSLEOFError, confirmed 100% reproducible,
    not flaky) while curl against the identical URL succeeds every time —
    a TLS-fingerprint quirk on their end, not a real outage. Shell out to
    curl for this one endpoint; fall back to requests (best effort, in case
    curl isn't on PATH in some deploy environment) rather than hard-failing."""
    url = NASS_ICS_URL.format(year=year)
    if shutil.which("curl"):
        try:
            result = subprocess.run(
                ["curl", "-sf", "--max-time", "8", url],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout:
                return _decode_ics(result.stdout)
        except (subprocess.SubprocessError, OSError):
            pass
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                return _decode_ics(resp.content)
        except requests.RequestException:
            pass
        time.sleep(0.3 * (attempt + 1))
    return None


def get_nass_dates(years: list[int]) -> list[tuple[date, str]]:
    """(date, report label) pairs for the grain-relevant reports in
    NASS_REPORT_KEYWORDS, across the given calendar years. Years before 2022
    return nothing here — see NASS_2021_FALLBACK. Fetched in parallel — one
    request per year, each ~100KB, adds up fast run sequentially."""
    out: list[tuple[date, str]] = []
    with ThreadPoolExecutor(max_workers=min(8, len(years))) as pool:
        futures = {pool.submit(_fetch_nass_year, y): y for y in years}
        for fut in as_completed(futures):
            text = fut.result()
            if not text:
                continue
            for day, summary in _parse_ics_events(text):
                low = summary.lower()
                for label, keywords in NASS_REPORT_KEYWORDS.items():
                    if any(kw in low for kw in keywords):
                        out.append((day, label))
                        break
    return sorted(set(out))
