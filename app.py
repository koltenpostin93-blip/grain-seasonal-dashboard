import base64
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_chat import run_chat
from massive_api import MassiveApiError, get_daily_bars_many, get_futures_curve
from report_dates import NASS_2021_FALLBACK, ReportDatesError, get_nass_dates, get_wasde_dates

HERE = Path(__file__).parent

LOGO_FILE = "logo-50yr.png"
FAVICON_FILE = "jsa_favicon.png"
WATERMARK_FILE = "logo-50yr.png"
WATERMARK_OPACITY = 0.10


def asset(name: str) -> str:
    return str(HERE / "assets" / name)


def watermark_path() -> str | None:
    for name in (WATERMARK_FILE, LOGO_FILE):
        candidate = asset(name)
        if Path(candidate).exists():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def watermark_uri(path: str) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


st.set_page_config(
    page_title="JSA - Grain Seasonal Futures & Spreads",
    page_icon=asset(FAVICON_FILE),
    layout="wide",
)

COMMODITIES = [
    {"key": "corn", "label": "Corn", "sublabel": "CBOT · ZC", "product_code": "ZC", "unit": "¢/bu"},
    {"key": "soybeans", "label": "Soybeans", "sublabel": "CBOT · ZS", "product_code": "ZS", "unit": "¢/bu"},
    {"key": "chi_wheat", "label": "Chicago wheat (SRW)", "sublabel": "CBOT · ZW", "product_code": "ZW", "unit": "¢/bu"},
    {"key": "kc_wheat", "label": "KC wheat (HRW)", "sublabel": "CBOT · KE", "product_code": "KE", "unit": "¢/bu"},
]

MONTH_LETTERS = {
    "F": "Jan", "G": "Feb", "H": "Mar", "J": "Apr", "K": "May", "M": "Jun",
    "N": "Jul", "Q": "Aug", "U": "Sep", "V": "Oct", "X": "Nov", "Z": "Dec",
}

YEAR_COLORS = ["#0693e3", "#e8833a", "#5aa469", "#b05fb0", "#9aa5b1", "#c0392b"]
AVG_COLOR = "#111111"
EXP_COLOR = "#c62828"
FND_COLOR = "#8e24aa"
MAX_YEARS_BACK = 5
DATA_START_NOTE = (
    "Massive's daily settlement history starts 2021-09-02, so seasonal overlays "
    "cover roughly the last 4 full contract years — older analogs are skipped, not wrong."
)
FND_NOTE = (
    "FND marks the CME grain rule's First Notice Day — the last business day of the "
    "month preceding the delivery month (weekends only, exchange holidays not applied)."
)

GROUP_BAND = "background-color:#EAF7EA;"


def grain_fnd(expiration: date) -> date:
    """CME grain rule: First Notice Day is the last business day of the month
    preceding the delivery month. All four CBOT grain contracts here are
    physically delivered, unlike CME livestock (cash-settled, no FND)."""
    first_of_delivery_month = pd.Timestamp(year=expiration.year, month=expiration.month, day=1)
    return (first_of_delivery_month - pd.offsets.BDay(1)).date()


def get_api_key() -> str:
    try:
        key = st.secrets.get("MASSIVE_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("MASSIVE_API_KEY", "")


def get_anthropic_key() -> str:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("ANTHROPIC_API_KEY", "")


def friendly_contract(ticker: str, product_code: str) -> str:
    suffix = ticker[len(product_code):]
    if len(suffix) == 2 and suffix[0] in MONTH_LETTERS:
        return f"{MONTH_LETTERS[suffix[0]]} '2{suffix[1]}"
    return ticker


def shift_ticker_year(ticker: str, product_code: str, delta: int) -> str | None:
    """ZCU6 -> ZCU5 at delta=-1. Massive quotes outrights with a single-digit year."""
    suffix = ticker[len(product_code):]
    if len(suffix) != 2 or suffix[0] not in MONTH_LETTERS:
        return None
    month, year = suffix[0], int(suffix[1])
    shifted = year + delta
    if shifted < 0:
        return None
    return f"{product_code}{month}{shifted % 10}"


@st.cache_data(ttl="5m", show_spinner=False)
def load_curve(product_code: str, api_key: str, as_of: str, n_contracts: int) -> pd.DataFrame:
    """Massive's contract list can lag a day right at midnight UTC rollover — the
    server's local date ticks over before the feed has published that date's active
    set, and /contracts comes back empty. Retry against yesterday rather than show
    a false "no live contracts" warning for what is really just feed lag."""
    d = date.fromisoformat(as_of)
    curve = get_futures_curve(product_code, api_key, d, n_contracts=n_contracts)
    if curve.empty:
        curve = get_futures_curve(product_code, api_key, d - timedelta(days=1), n_contracts=n_contracts)
    return curve


@st.cache_data(ttl="6h", show_spinner="Loading settlement history…")
def load_bars(tickers: tuple[str, ...], api_key: str) -> dict[str, pd.DataFrame]:
    return get_daily_bars_many(list(tickers), api_key)


def load_histories(tickers: tuple[str, ...], api_key: str) -> dict[str, pd.Series]:
    """Settlement-price view over load_bars, for the existing line-chart call sites."""
    bars_map = load_bars(tickers, api_key)
    return {t: b["settle"] for t, b in bars_map.items() if not b.empty}


@st.cache_data(ttl="24h", show_spinner="Loading WASDE report dates…")
def load_wasde_dates() -> list[date]:
    try:
        return get_wasde_dates()
    except ReportDatesError:
        return []


@st.cache_data(ttl="24h", show_spinner="Loading NASS report calendar…")
def load_nass_dates(years: tuple[int, ...]) -> list[tuple[date, str]]:
    events = get_nass_dates(list(years))
    if min(years) <= 2021:
        events = events + NASS_2021_FALLBACK
    return sorted(set(events))


def plotly_config(filename: str) -> dict:
    return {
        "displayModeBar": True,
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": filename,
                                 "height": 700, "width": 1400, "scale": 2},
    }


@st.cache_data(show_spinner=False, max_entries=32)
def figure_png(fig_json: str) -> bytes | None:
    try:
        import plotly.io as pio
        return pio.from_json(fig_json).to_image(format="png", width=1400, height=700, scale=2)
    except Exception:
        return None


def export_row(frame: pd.DataFrame, filename: str, key: str, fig=None):
    row = st.container(horizontal=True, vertical_alignment="center")
    with row:
        tsv = frame.to_csv(sep="\t", index=False)
        with st.popover("Copy", width=90):
            st.caption("Tab-separated — use the copy icon, then paste into Excel.")
            st.code(tsv, language=None, height=260)
        st.download_button("CSV", frame.to_csv(index=False).encode(), f"{filename}.csv",
                           "text/csv", key=f"csv_{key}", width=90)
        if fig is not None:
            if st.button("PNG", key=f"png_btn_{key}", width=90,
                         help="Render this chart as a PNG for download."):
                st.session_state[f"png_ready_{key}"] = True
            if st.session_state.get(f"png_ready_{key}"):
                data = figure_png(fig.to_json())
                if data:
                    st.download_button("Save PNG", data, f"{filename}.png", "image/png",
                                       key=f"png_dl_{key}", width=120)
                else:
                    st.caption("PNG export unavailable — use the camera icon on the chart.")


def _add_vline(fig, x, text, color):
    x = pd.Timestamp(x) if isinstance(x, date) else x
    fig.add_shape(type="line", xref="x", yref="paper", x0=x, x1=x, y0=0, y1=1,
                  line=dict(color=color, dash="dash", width=1.5))
    fig.add_annotation(x=x, xref="x", y=1.0, yref="paper", text=text, showarrow=False,
                       yanchor="bottom", font=dict(size=10, color=color))


REPORT_COLORS = {"WASDE": "#e8833a", "NASS": "#5aa469"}


def _add_report_markers(fig, report_dates: dict[str, list[date]], start: date, end: date):
    """Thin, unlabeled vertical lines for report-release dates inside [start, end],
    one per date, with a single legend entry per source (shapes don't otherwise
    appear in the legend, and per-line text annotations would clutter a chart
    that can have a dozen+ release dates in view)."""
    for name, dates in report_dates.items():
        color = REPORT_COLORS.get(name, "#888888")
        visible = [d for d in dates if start <= d <= end]
        for d in visible:
            x = pd.Timestamp(d)
            fig.add_shape(type="line", xref="x", yref="paper", x0=x, x1=x, y0=0, y1=1,
                          line=dict(color=color, dash="dot", width=1), opacity=0.55, layer="below")
        if visible:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                     line=dict(color=color, dash="dot", width=1.5),
                                     name=f"{name} release", showlegend=True))


def _selected_report_dates(wasde_dates: list[date], nass_dates: list[tuple[date, str]],
                           show_wasde: bool, show_nass_major: bool,
                           show_crop_progress: bool) -> dict[str, list[date]]:
    out: dict[str, list[date]] = {}
    if show_wasde and wasde_dates:
        out["WASDE"] = wasde_dates
    nass_selected = []
    for d, label in nass_dates:
        if label == "Crop Progress":
            if show_crop_progress:
                nass_selected.append(d)
        elif show_nass_major:
            nass_selected.append(d)
    if nass_selected:
        out["NASS"] = sorted(set(nass_selected))
    return out


def _style_axes(fig, y_title, x_title, height=420):
    wm = watermark_path()
    if wm:
        fig.add_layout_image(dict(
            source=watermark_uri(wm), xref="paper", yref="paper",
            x=0.5, y=0.5, sizex=0.5, sizey=0.5,
            xanchor="center", yanchor="middle", sizing="contain",
            opacity=WATERMARK_OPACITY, layer="below",
        ))
    fig.update_layout(
        height=height, margin=dict(l=10, r=20, t=30, b=10),
        yaxis_title=y_title, xaxis_title=x_title,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=10)),
    )
    fig.update_yaxes(gridcolor="#eceff1", zeroline=True, zerolinecolor="#cfd8dc", automargin=True)
    fig.update_xaxes(gridcolor="#eceff1")


WINDOW_CHOICES = {"6M": 183, "1Y": 365, "18M": 548}


def _year_grid_align(by_dte: dict[str, pd.Series], window_days: int) -> pd.DataFrame:
    """Put every year on a common daily grid so they can be combined pointwise —
    shared by both the mean and the median/percentile-band summaries below."""
    grid = pd.RangeIndex(-window_days, 1)
    aligned = {}
    for name, s in by_dte.items():
        clean = s[~s.index.duplicated(keep="last")].sort_index()
        aligned[name] = clean.reindex(grid).interpolate(limit_area="inside")
    return pd.DataFrame(aligned)


def _year_grid_average(by_dte: dict[str, pd.Series], window_days: int) -> pd.Series:
    """Mean across years at each grid point, restricted to points where most
    years are present — avoids the mean lurching between 'all years' and 'one
    lonely year' at the edges of the window."""
    frame = _year_grid_align(by_dte, window_days)
    required = max(2, (len(by_dte) + 1) // 2)
    return frame.mean(axis=1, skipna=True)[frame.count(axis=1) >= required]


def _year_grid_median_band(by_dte: dict[str, pd.Series], window_days: int,
                           band: tuple[float, float] = (0.25, 0.75)) -> pd.DataFrame:
    """Median plus a lower/upper percentile band across years at each grid
    point — resists distortion from any single outlier year (a drought spike,
    a trade-war shock) the way a plain mean can't."""
    frame = _year_grid_align(by_dte, window_days)
    required = max(2, (len(by_dte) + 1) // 2)
    enough = frame.count(axis=1) >= required
    stats = pd.DataFrame({
        "median": frame.median(axis=1, skipna=True),
        "lower": frame.quantile(band[0], axis=1),
        "upper": frame.quantile(band[1], axis=1),
    })
    return stats[enough]


HARMONIC_PERIOD_DAYS = 365.25
HARMONIC_TERMS = 2  # fundamental annual cycle + one overtone; keeps the fit smooth rather than wiggly


def _harmonic_seasonal_curve(by_dte: dict[str, pd.Series], window_days: int,
                             n_harmonics: int = HARMONIC_TERMS) -> pd.Series:
    """Fits one smooth seasonal curve by pooling every overlaid year's (dte,
    price) points into a single regression, rather than aggregating them
    pointwise the way Average/Median do.

    Different years sit at completely different price levels (2022's drought
    corn vs. 2024's), so pooling raw prices into a plain sin/cos regression
    would just fit cross-year drift, not within-year seasonality. The fix is
    a per-year fixed effect (one dummy variable per contract year) plus
    harmonic terms shared across all years: the dummies absorb each year's
    price level, leaving the sin/cos coefficients to capture only the common
    seasonal shape. The returned curve is that shared shape added back onto
    the current (back=0) year's own fitted level, so it reads in the same
    price terms as the other traces — "where this year's contract would sit
    if it were tracking the typical seasonal path."
    """
    years = list(by_dte.keys())
    if len(years) < 2:
        return pd.Series(dtype=float)

    xs, ys, year_idx = [], [], []
    for i, s in enumerate(by_dte.values()):
        clean = s[~s.index.duplicated(keep="last")].dropna()
        xs.extend(clean.index.to_numpy(dtype=float))
        ys.extend(clean.to_numpy(dtype=float))
        year_idx.extend([i] * len(clean))
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    year_idx = np.asarray(year_idx)
    if len(xs) < 4 * n_harmonics + len(years):
        return pd.Series(dtype=float)  # not enough points to fit this many parameters

    n_years = len(years)
    year_dummies = np.zeros((len(xs), n_years))
    year_dummies[np.arange(len(xs)), year_idx] = 1
    harmonic_cols = []
    for k in range(1, n_harmonics + 1):
        harmonic_cols.append(np.sin(2 * np.pi * k * xs / HARMONIC_PERIOD_DAYS))
        harmonic_cols.append(np.cos(2 * np.pi * k * xs / HARMONIC_PERIOD_DAYS))
    design = np.column_stack([year_dummies, *harmonic_cols])
    coefs, *_ = np.linalg.lstsq(design, ys, rcond=None)

    current_level = coefs[0]  # by_dte's insertion order is back=0 (current year) first
    harmonic_coefs = coefs[n_years:]
    grid = np.arange(-window_days, 1)
    grid_cols = []
    for k in range(1, n_harmonics + 1):
        grid_cols.append(np.sin(2 * np.pi * k * grid / HARMONIC_PERIOD_DAYS))
        grid_cols.append(np.cos(2 * np.pi * k * grid / HARMONIC_PERIOD_DAYS))
    fitted = current_level + np.column_stack(grid_cols) @ harmonic_coefs
    return pd.Series(fitted, index=pd.Index(grid, name="dte"))


# "Off" skips any cross-year summary line. "Average" is the original plain
# mean. "Median + bands" resists distortion from a single outlier year.
# "Harmonic" fits one smooth seasonal shape (Fourier regression on day-of-
# year, pooled across years with per-year fixed effects) rather than
# aggregating years pointwise.
SEASONAL_LINE_CHOICES = ["Off", "Average", "Median + bands", "Harmonic"]


def _add_seasonal_overlay(fig, by_dte: dict[str, pd.Series], window_days: int,
                          seasonal_line: str, anchor_expiry: date, fmt: str):
    """Draws the chosen cross-year summary on top of the individual year
    traces already added to fig."""
    if seasonal_line == "Off" or len(by_dte) <= 1:
        return
    if seasonal_line == "Median + bands":
        stats = _year_grid_median_band(by_dte, window_days)
        if not len(stats):
            return
        xs = [anchor_expiry + timedelta(days=int(d)) for d in stats.index]
        fig.add_trace(go.Scatter(x=xs, y=list(stats["upper"]), mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=list(stats["lower"]), mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(17,17,17,0.12)",
                                 name=f"25–75th pct ({len(by_dte)}yr)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=list(stats["median"]), mode="lines",
                                 name=f"Median ({len(by_dte)}yr)",
                                 line=dict(color=AVG_COLOR, width=2.2, dash="dot"),
                                 hovertemplate=f"Median<br>%{{y:{fmt}}}<extra></extra>"))
        return
    if seasonal_line == "Harmonic":
        curve = _harmonic_seasonal_curve(by_dte, window_days)
        if not len(curve):
            return
        fig.add_trace(go.Scatter(
            x=[anchor_expiry + timedelta(days=int(d)) for d in curve.index],
            y=list(curve.values), mode="lines", name=f"Harmonic fit ({len(by_dte)}yr)",
            line=dict(color=AVG_COLOR, width=2.2, dash="dot"),
            hovertemplate=f"Harmonic<br>%{{y:{fmt}}}<extra></extra>",
        ))
        return
    # default / "Average"
    avg = _year_grid_average(by_dte, window_days)
    if len(avg):
        fig.add_trace(go.Scatter(
            x=[anchor_expiry + timedelta(days=int(d)) for d in avg.index],
            y=list(avg.values), mode="lines", name=f"Avg ({len(by_dte)}yr)",
            line=dict(color=AVG_COLOR, width=2.2, dash="dot"),
            hovertemplate=f"Avg<br>%{{y:{fmt}}}<extra></extra>",
        ))


def render_seasonal_futures(commodity: dict, api_key: str, as_of: date, report_dates: dict | None = None):
    code = commodity["product_code"]
    key = commodity["key"]
    unit = commodity["unit"]

    try:
        # n_contracts matches the other sub-tabs (10) so they share one cached
        # curve fetch per commodity instead of paying for the live-quote round
        # trip twice.
        curve = load_curve(code, api_key, as_of.isoformat(), 10)
    except MassiveApiError as e:
        st.error(f"Couldn't load {commodity['label']} quotes: {e}")
        return
    if curve.empty:
        st.warning(f"{commodity['label']}: no live contracts.")
        return

    tickers = list(curve["ticker"])
    expiries = dict(zip(curve["ticker"], curve["expiration"]))

    row = st.container(horizontal=True, vertical_alignment="bottom")
    with row:
        ticker = st.selectbox(
            "Contract", tickers, key=f"fut_ticker_{key}", width=150,
            format_func=lambda t: friendly_contract(t, code),
        )
        years_back = st.slider("Prior years", 1, MAX_YEARS_BACK, 4, key=f"fut_years_{key}", width=170)
        indexed = st.toggle("Indexed (start = 100)", value=False, key=f"fut_idx_{key}",
                            help="Rebase each year to 100 at the start of the window, so years with "
                                 "very different price levels can be compared on shape alone.")
        seasonal_line = st.segmented_control("Seasonal line", SEASONAL_LINE_CHOICES,
                                             default="Average", key=f"fut_avgmode_{key}")
        window_label = st.segmented_control("Window", list(WINDOW_CHOICES), default="1Y", key=f"fut_win_{key}")

    window_days = WINDOW_CHOICES.get(window_label or "1Y", 365)
    anchor_expiry = expiries[ticker]
    label = friendly_contract(ticker, code)
    y_title = "Index (start = 100)" if indexed else f"Price ({unit})"
    fmt = ".1f" if indexed else ".2f"

    shifted = [shift_ticker_year(ticker, code, -b) for b in range(years_back + 1)]
    shifted = [t for t in shifted if t]
    hist = load_histories(tuple(shifted), api_key)

    st.caption(f"**{label}** — recent history")
    current = hist.get(ticker)
    if current is None or not len(current):
        st.info("No settlement history for this contract yet.")
    else:
        cutoff = as_of - timedelta(days=window_days)
        shown = current[current.index >= cutoff]
        if not len(shown):
            st.info(f"No sessions inside the {window_label} window.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(shown.index), y=list(shown.values), mode="lines", name=label,
                line=dict(color=YEAR_COLORS[0], width=2), showlegend=False,
                hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}<extra></extra>",
            ))
            if as_of <= anchor_expiry:
                _add_vline(fig, anchor_expiry, "expiration", EXP_COLOR)
                _add_vline(fig, grain_fnd(anchor_expiry), "FND", FND_COLOR)
            if report_dates:
                _add_report_markers(fig, report_dates, shown.index.min(), shown.index.max())
            _style_axes(fig, f"Price ({unit})", None)
            st.plotly_chart(fig, width="stretch", key=f"fut_hist_{key}",
                            config=plotly_config(f"{key}_{ticker}_history"))
            export_row(shown.rename("price").reset_index().rename(columns={"index": "date"}),
                       f"{key}_{ticker}_history", key=f"futhist_{key}")

    st.caption(f"**{label}** — seasonal, aligned on expiration")
    fig = go.Figure()
    by_dte: dict[str, pd.Series] = {}
    skipped: list[str] = []

    for back in range(years_back + 1):
        t = shift_ticker_year(ticker, code, -back)
        if not t:
            continue
        series = hist.get(t)
        if series is None or not len(series):
            skipped.append(t)
            continue
        year_expiry = expiries.get(t, series.index.max())
        dte = [-(year_expiry - d).days for d in series.index]
        keep = [i for i, d in enumerate(dte) if d >= -window_days]
        if not keep:
            skipped.append(t)
            continue

        ys = [series.values[i] for i in keep]
        if indexed:
            base = ys[0]
            if not base:
                skipped.append(t)
                continue
            ys = [y / base * 100 for y in ys]

        xs_dte = [dte[i] for i in keep]
        xs = [anchor_expiry + timedelta(days=d) for d in xs_dte]
        name = t + (" (current)" if back == 0 else "")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=name,
            line=dict(color=YEAR_COLORS[back % len(YEAR_COLORS)], width=3 if back == 0 else 1.5),
            opacity=1.0 if back == 0 else 0.75,
            hovertemplate=f"{name}<br>%{{y:{fmt}}}<extra></extra>",
        ))
        by_dte[t] = pd.Series(ys, index=pd.Index(xs_dte, name="dte"))

    if not by_dte:
        st.info("No settlement history available for this contract's prior-year analogs.")
    else:
        _add_seasonal_overlay(fig, by_dte, window_days, seasonal_line or "Average", anchor_expiry, fmt)
        if as_of <= anchor_expiry:
            _add_vline(fig, anchor_expiry, "expiration", EXP_COLOR)
            _add_vline(fig, grain_fnd(anchor_expiry), "FND", FND_COLOR)
        _style_axes(fig, y_title, None)
        st.plotly_chart(fig, width="stretch", key=f"fut_seas_{key}",
                        config=plotly_config(f"{key}_{ticker}_seasonal"))
        export_row(pd.DataFrame(by_dte).sort_index().reset_index(),
                   f"{key}_{ticker}_seasonal", key=f"futseas_{key}")
        note = (
            f"{len(by_dte)} contract year{'s' if len(by_dte) != 1 else ''} overlaid · x = 0 is "
            f"{label}'s expiration, so every year lines up at the same point in its life."
        )
        if skipped:
            note += f" No usable history for {', '.join(skipped)}."
        st.caption(note)


MAX_SPREAD_LEGS = 6


def _leg_label(code: str, ticker: str, sign: int) -> str:
    return f"{'+' if sign > 0 else '−'}{friendly_contract(ticker, code)}"


def _spread_label(legs: list[tuple[str, str, int]]) -> str:
    return " / ".join(_leg_label(c, t, s) for c, t, s in legs)


def _combine_series(hist: dict[str, pd.Series], legs: list[tuple[str, str, int]]) -> pd.Series | None:
    """legs: list of (product_code, ticker, sign). Combines each leg's settle
    series on their overlapping sessions; None if any leg has no history."""
    series_list = []
    for _, ticker, _ in legs:
        s = hist.get(ticker)
        if s is None or not len(s):
            return None
        series_list.append(s)
    combined = pd.concat({i: s for i, s in enumerate(series_list)}, axis=1).dropna()
    if combined.empty:
        return None
    return sum(sign * combined[i] for i, (_, _, sign) in enumerate(legs))


def render_leg_picker(curves: dict[str, pd.DataFrame], state_key: str,
                      cross_commodity: bool) -> list[tuple[str, str, int]] | None:
    """Renders one row per leg (commodity picker only if cross_commodity, always a
    contract picker + Buy/Sell side except leg 1 which is the fixed +1 anchor),
    plus +/− buttons that grow or shrink the leg count. Returns the picked legs
    as (product_code, ticker, sign), or None if a row can't be completed yet
    (e.g. a commodity has no live contracts). For a fixed (non-cross-commodity)
    spread, `curves` holds exactly one product_code -> curve entry."""
    n_legs_key = f"{state_key}_nlegs"
    n_legs = st.session_state.setdefault(n_legs_key, 2)
    fixed_code = None if cross_commodity else next(iter(curves))

    legs: list[tuple[str, str, int]] = []
    complete = True
    for i in range(n_legs):
        row = st.container(horizontal=True, vertical_alignment="bottom")
        with row:
            if cross_commodity:
                names = [c["label"] for c in COMMODITIES]
                pick = st.selectbox(f"Leg {i + 1}", names, index=min(i, len(COMMODITIES) - 1),
                                    key=f"{state_key}_c{i}", width=170)
                code = COMMODITIES[names.index(pick)]["product_code"]
            else:
                code = fixed_code
            curve = curves.get(code, pd.DataFrame())
            tickers = list(curve["ticker"]) if not curve.empty else []
            if not tickers:
                st.warning(f"No live contracts for {code}.")
                complete = False
                continue
            ticker = st.selectbox(
                "Contract" if cross_commodity else f"Leg {i + 1}", tickers,
                index=min(i, len(tickers) - 1), key=f"{state_key}_t{i}", width=150,
                format_func=lambda t, code=code: friendly_contract(t, code),
            )
            if i == 0:
                sign = 1
                st.caption("Buy (anchor)")
            else:
                side = st.segmented_control("Side", ["Buy", "Sell"], default="Sell",
                                            key=f"{state_key}_side{i}")
                sign = 1 if (side or "Sell") == "Buy" else -1
        legs.append((code, ticker, sign))

    btns = st.container(horizontal=True)
    with btns:
        if st.button("+ Add leg", key=f"{state_key}_add", disabled=n_legs >= MAX_SPREAD_LEGS):
            st.session_state[n_legs_key] = n_legs + 1
            st.rerun()
        if st.button("− Remove leg", key=f"{state_key}_rm", disabled=n_legs <= 2):
            st.session_state[n_legs_key] = n_legs - 1
            st.rerun()

    return legs if complete else None


def render_spread_charts(legs: list[tuple[str, str, int]], api_key: str, as_of: date,
                         expiries_by_product: dict[str, dict[str, date]], years_back: int,
                         seasonal_line: str, window_label: str, unit: str,
                         report_dates: dict | None, key: str):
    """Shared recent-history + seasonal-overlay rendering for both the
    per-commodity calendar spread (fixed product_code across legs) and the
    cross-commodity spread builder (each leg its own product_code)."""
    window_days = WINDOW_CHOICES.get(window_label or "1Y", 365)
    label = _spread_label(legs)
    y_title = f"Spread ({unit})"
    anchor_code, anchor_ticker, _ = legs[0]
    anchor_expiry = expiries_by_product.get(anchor_code, {}).get(anchor_ticker)
    if anchor_expiry is None:
        st.warning("Couldn't resolve the first leg's expiration.")
        return

    year_legsets: list[list[tuple[str, str, int]]] = []
    for back in range(years_back + 1):
        shifted = []
        ok = True
        for code, ticker, sign in legs:
            t = shift_ticker_year(ticker, code, -back)
            if not t:
                ok = False
                break
            shifted.append((code, t, sign))
        if ok:
            year_legsets.append(shifted)

    all_tickers = tuple(sorted({t for legset in year_legsets for _, t, _ in legset}))
    hist = load_histories(all_tickers, api_key)

    st.caption(f"**{label}** — recent history")
    current_series = _combine_series(hist, year_legsets[0]) if year_legsets else None
    if current_series is None:
        st.info("No overlapping settlement history for this spread.")
    else:
        cutoff = as_of - timedelta(days=window_days)
        shown = current_series[current_series.index >= cutoff]
        if not len(shown):
            st.info(f"No sessions inside the {window_label} window.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(shown.index), y=list(shown.values), mode="lines", name=label,
                line=dict(color=YEAR_COLORS[0], width=2), showlegend=False,
                hovertemplate="%{x|%b %d, %Y}<br>%{y:+.2f}<extra></extra>",
            ))
            if report_dates:
                _add_report_markers(fig, report_dates, shown.index.min(), shown.index.max())
            _style_axes(fig, y_title, None)
            st.plotly_chart(fig, width="stretch", key=f"sp_hist_{key}",
                            config=plotly_config(f"{key}_history"))
            export_row(shown.rename("spread").reset_index().rename(columns={"index": "date"}),
                       f"{key}_history", key=f"sphist_{key}")

    st.caption(f"**{label}** — seasonal, aligned on leg 1 expiration")
    fig = go.Figure()
    by_dte: dict[str, pd.Series] = {}
    skipped: list[str] = []

    for back, legset in enumerate(year_legsets):
        spread = _combine_series(hist, legset)
        if spread is None:
            skipped.append(_spread_label(legset))
            continue
        leg0_code, leg0_ticker, _ = legset[0]
        leg0_expiry = expiries_by_product.get(leg0_code, {}).get(leg0_ticker, spread.index.max())
        dte = [-(leg0_expiry - d).days for d in spread.index]
        keep = [i for i, d in enumerate(dte) if d >= -window_days]
        if not keep:
            skipped.append(_spread_label(legset))
            continue

        xs_dte = [dte[i] for i in keep]
        xs = [anchor_expiry + timedelta(days=d) for d in xs_dte]
        ys = [spread.values[i] for i in keep]
        name = _spread_label(legset) + (" (current)" if back == 0 else "")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=name,
            line=dict(color=YEAR_COLORS[back % len(YEAR_COLORS)], width=3 if back == 0 else 1.5),
            opacity=1.0 if back == 0 else 0.75,
            hovertemplate=f"{name}<br>%{{y:+.2f}}<extra></extra>",
        ))
        by_dte[_spread_label(legset)] = pd.Series(ys, index=pd.Index(xs_dte, name="dte"))

    if not by_dte:
        st.info("No overlapping settlement history for this spread's prior-year analogs.")
    else:
        _add_seasonal_overlay(fig, by_dte, window_days, seasonal_line or "Average", anchor_expiry, "+.2f")
        if as_of <= anchor_expiry:
            _add_vline(fig, anchor_expiry, "leg 1 expiration", EXP_COLOR)
            _add_vline(fig, grain_fnd(anchor_expiry), "leg 1 FND", FND_COLOR)
        _style_axes(fig, y_title, None)
        st.plotly_chart(fig, width="stretch", key=f"sp_seas_{key}",
                        config=plotly_config(f"{key}_seasonal"))
        export_row(pd.DataFrame(by_dte).sort_index().reset_index(),
                   f"{key}_seasonal", key=f"spseas_{key}")
        note = (
            f"{len(by_dte)} contract year{'s' if len(by_dte) != 1 else ''} overlaid · x = 0 is "
            f"leg 1's expiration, so every year lines up at the same point in its life."
        )
        if skipped:
            note += f" No usable history for {', '.join(skipped)}."
        st.caption(note)


def render_seasonal_spread(commodity: dict, api_key: str, as_of: date, report_dates: dict | None = None):
    code = commodity["product_code"]
    key = commodity["key"]
    unit = commodity["unit"]

    try:
        curve = load_curve(code, api_key, as_of.isoformat(), 10)
    except MassiveApiError as e:
        st.error(f"Couldn't load {commodity['label']} quotes: {e}")
        return
    if curve.empty or len(curve) < 2:
        st.warning(f"{commodity['label']}: not enough live contracts to build a spread.")
        return

    legs = render_leg_picker({code: curve}, f"sp_{key}", cross_commodity=False)

    controls = st.container(horizontal=True, vertical_alignment="bottom")
    with controls:
        years_back = st.slider("Prior years", 1, MAX_YEARS_BACK, 4, key=f"sp_years_{key}", width=170)
        seasonal_line = st.segmented_control("Seasonal line", SEASONAL_LINE_CHOICES,
                                             default="Average", key=f"sp_avgmode_{key}")
        window_label = st.segmented_control("Window", list(WINDOW_CHOICES), default="1Y", key=f"sp_win_{key}")

    if not legs:
        return

    expiries = dict(zip(curve["ticker"], curve["expiration"]))
    render_spread_charts(legs, api_key, as_of, {code: expiries}, years_back, seasonal_line or "Average",
                         window_label, unit, report_dates, key=f"{key}_{'_'.join(t for _, t, _ in legs)}")


def build_spread_matrix(curve: pd.DataFrame, code: str) -> tuple[pd.DataFrame, list[str]]:
    """Near contracts down the rows, deferred contracts across the columns — every
    near/far combination's current nominal spread (near price - far price) at once."""
    legs = list(curve.itertuples(index=False))
    labels = [friendly_contract(r.ticker, code) for r in legs]
    rows = []
    for i, near in enumerate(legs):
        row = {"Contract": labels[i], "Price": near.price}
        for j, far in enumerate(legs):
            col = labels[j]
            row[col] = near.price - far.price if j > i else None
        rows.append(row)
    return pd.DataFrame(rows), labels


def render_spread_matrix(commodity: dict, api_key: str, as_of: date):
    code = commodity["product_code"]
    key = commodity["key"]
    unit = commodity["unit"]

    st.caption(
        f"Every near month against every deferred month at once, in {unit} "
        "(near price − far price). Positive = near trading over far (inverted); "
        "negative = near trading under far (normal carry-market shape)."
    )
    n_load = st.slider("Contract months", 3, 10, 10, key=f"mx_months_{key}", width=200)

    try:
        curve = load_curve(code, api_key, as_of.isoformat(), n_load)
    except MassiveApiError as e:
        st.error(f"Couldn't load {commodity['label']} quotes: {e}")
        return
    if curve.empty or len(curve) < 2:
        st.warning("Not enough live contracts to build a matrix.")
        return

    frame, labels = build_spread_matrix(curve, code)
    display = frame.copy()
    display["Price"] = [f"{v:.2f}" for v in frame["Price"]]
    for col in labels:
        display[col] = [f"{v:+.2f}" if v is not None and pd.notna(v) else "" for v in frame[col]]

    def zebra(row: pd.Series):
        return [GROUP_BAND if row.name % 2 == 0 else ""] * len(row)

    styler = display.style.apply(zebra, axis=1)
    with st.container(key=f"tablewrap_mx_{key}"):
        st.dataframe(styler, hide_index=True, width="stretch",
                     height=min(35 * (len(display) + 1) + 3, 500))
    export_row(display, f"spread_matrix_{key}", key=f"mx_{key}")


TIMEFRAME_RULES = {"Daily": None, "Weekly": "W-FRI", "Monthly": "ME"}


def resample_ohlc(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Daily OHLC -> weekly/monthly OHLC (open=first, high=max, low=min, close=last)."""
    b = bars.copy()
    b.index = pd.to_datetime(b.index)
    o = b["open"].resample(rule).first()
    h = b["high"].resample(rule).max()
    l = b["low"].resample(rule).min()
    c = b["close"].resample(rule).last()
    return pd.concat({"open": o, "high": h, "low": l, "close": c}, axis=1).dropna(how="any")


def monthly_high_low(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar month the contract traded: the session high/low made
    that month and the date each occurred on."""
    if bars.empty:
        return pd.DataFrame()
    b = bars.copy()
    b.index = pd.to_datetime(b.index)
    rows = []
    for period, group in b.groupby(b.index.to_period("M")):
        hi_idx = group["high"].idxmax()
        lo_idx = group["low"].idxmin()
        rows.append({
            "Month": period.strftime("%b %Y"),
            "High": group.loc[hi_idx, "high"],
            "High date": hi_idx.date(),
            "Low": group.loc[lo_idx, "low"],
            "Low date": lo_idx.date(),
        })
    return pd.DataFrame(rows)


def render_contract_history(commodity: dict, api_key: str, as_of: date, report_dates: dict | None = None):
    code = commodity["product_code"]
    key = commodity["key"]
    unit = commodity["unit"]

    try:
        curve = load_curve(code, api_key, as_of.isoformat(), 10)
    except MassiveApiError as e:
        st.error(f"Couldn't load {commodity['label']} quotes: {e}")
        return
    if curve.empty:
        st.warning(f"{commodity['label']}: no live contracts.")
        return

    tickers = list(curve["ticker"])

    row = st.container(horizontal=True, vertical_alignment="bottom")
    with row:
        base_ticker = st.selectbox(
            "Contract", tickers, key=f"hist_ticker_{key}", width=150,
            format_func=lambda t: friendly_contract(t, code),
        )
        years_back = st.slider(
            "Contract year", 0, MAX_YEARS_BACK, 0, key=f"hist_yearsback_{key}", width=190,
            help="0 = the currently listed contract; higher values step back to that "
                 "same contract month in a prior year (e.g. last year's Sep corn).",
        )
        timeframe = st.segmented_control(
            "Timeframe", list(TIMEFRAME_RULES), default="Weekly", key=f"hist_tf_{key}",
        )

    target = shift_ticker_year(base_ticker, code, -years_back)
    if not target:
        st.info("No contract available that far back.")
        return

    bars = load_bars((target,), api_key).get(target)
    if bars is None or bars.empty:
        st.info(f"No settlement history for {target} yet.")
        return

    label = friendly_contract(target, code)
    rule = TIMEFRAME_RULES.get(timeframe or "Weekly")
    plot_bars = resample_ohlc(bars, rule) if rule else bars

    hi_idx, lo_idx = bars["high"].idxmax(), bars["low"].idxmin()
    with st.container(horizontal=True):
        st.metric("All-time high", f"{bars.loc[hi_idx, 'high']:.2f}",
                  f"{pd.Timestamp(hi_idx):%b %d, %Y}", delta_color="off", border=True)
        st.metric("All-time low", f"{bars.loc[lo_idx, 'low']:.2f}",
                  f"{pd.Timestamp(lo_idx):%b %d, %Y}", delta_color="off", border=True)
        st.metric("Sessions on record", f"{len(bars):,}",
                  f"{pd.Timestamp(bars.index.min()):%b %Y} → {pd.Timestamp(bars.index.max()):%b %Y}",
                  delta_color="off", border=True)

    st.caption(f"**{label}** — {timeframe or 'Weekly'} ({unit})")
    fig = go.Figure(go.Candlestick(
        x=list(plot_bars.index), open=plot_bars["open"], high=plot_bars["high"],
        low=plot_bars["low"], close=plot_bars["close"], showlegend=False,
        increasing_line_color=YEAR_COLORS[2], decreasing_line_color=EXP_COLOR, name=label,
    ))
    if report_dates:
        _add_report_markers(fig, report_dates, pd.Timestamp(plot_bars.index.min()).date(),
                            pd.Timestamp(plot_bars.index.max()).date())
    _style_axes(fig, f"Price ({unit})", None, height=380)
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch", key=f"hist_candle_{key}",
                    config=plotly_config(f"{key}_{target}_{timeframe}"))
    export_row(plot_bars.reset_index().rename(columns={"index": "date"}),
               f"{key}_{target}_{timeframe}", key=f"histchart_{key}")

    st.caption(f"**{label}** — monthly highs & lows")
    stats = monthly_high_low(bars)
    if stats.empty:
        st.info("No monthly stats available.")
    else:
        display = stats.copy()
        display["High"] = display["High"].map(lambda v: f"{v:.2f}")
        display["Low"] = display["Low"].map(lambda v: f"{v:.2f}")
        with st.container(key=f"tablewrap_hilo_{key}"):
            st.dataframe(display, hide_index=True, width="stretch",
                        height=min(35 * (len(display) + 1) + 3, 480))
        export_row(stats, f"{key}_{target}_monthly_hilo", key=f"histstats_{key}")
    st.caption(
        f"{label} traded {len(bars):,} sessions from "
        f"{pd.Timestamp(bars.index.min()):%b %d, %Y} to {pd.Timestamp(bars.index.max()):%b %d, %Y}. "
        "Highs/lows are session intraday extremes, not settlement prices."
    )


def render_commodity(commodity: dict, api_key: str, as_of: date, report_dates: dict | None = None):
    tab_futures, tab_spread, tab_matrix, tab_history = st.tabs(
        ["Seasonal futures", "Seasonal spread", "Spread matrix", "Contract history"]
    )
    with tab_futures:
        render_seasonal_futures(commodity, api_key, as_of, report_dates)
    with tab_spread:
        render_seasonal_spread(commodity, api_key, as_of, report_dates)
    with tab_matrix:
        render_spread_matrix(commodity, api_key, as_of)
    with tab_history:
        render_contract_history(commodity, api_key, as_of, report_dates)


def render_cross_commodity_spread(api_key: str, as_of: date, report_dates: dict | None = None):
    st.markdown("##### Cross-commodity spread")
    st.caption(
        "Spread any two or more commodities' contracts against each other — e.g. KC wheat "
        "over corn, or soybeans over Chicago wheat — same leg builder as each commodity's "
        "own Seasonal spread tab, but each leg picks its own commodity."
    )

    curves: dict[str, pd.DataFrame] = {}
    expiries_by_product: dict[str, dict[str, date]] = {}
    for c in COMMODITIES:
        try:
            curve = load_curve(c["product_code"], api_key, as_of.isoformat(), 10)
        except MassiveApiError:
            curve = pd.DataFrame()
        curves[c["product_code"]] = curve
        expiries_by_product[c["product_code"]] = (
            dict(zip(curve["ticker"], curve["expiration"])) if not curve.empty else {}
        )

    legs = render_leg_picker(curves, "xsp", cross_commodity=True)

    controls = st.container(horizontal=True, vertical_alignment="bottom")
    with controls:
        years_back = st.slider("Prior years", 1, MAX_YEARS_BACK, 4, key="xsp_years", width=170)
        seasonal_line = st.segmented_control("Seasonal line", SEASONAL_LINE_CHOICES,
                                             default="Average", key="xsp_avgmode")
        window_label = st.segmented_control("Window", list(WINDOW_CHOICES), default="1Y", key="xsp_win")

    if not legs:
        return

    # ¢/bu for every commodity here; recompute rather than hardcode in case a
    # future commodity is added in different units.
    units = {c["product_code"]: c["unit"] for c in COMMODITIES}
    unit = units.get(legs[0][0], "¢/bu")

    render_spread_charts(legs, api_key, as_of, expiries_by_product, years_back, seasonal_line or "Average",
                         window_label, unit, report_dates,
                         key="xsp_" + "_".join(t for _, t, _ in legs))


def render_report_calendar(wasde_dates: list[date], nass_dates: list[tuple[date, str]], as_of: date):
    st.markdown("##### Report release calendar")
    st.caption(
        "WASDE dates come from USDA's ESMIS report archive (esmis.nal.usda.gov) — the "
        "authoritative release record, since WASDE does get rescheduled (e.g. Oct 2025's "
        "report was cancelled outright by the government shutdown and folded into a "
        "delayed Nov 14 release). NASS dates come from NASS's published .ics release "
        "calendar (2022 onward); Sep–Dec 2021 is backfilled from the well-documented "
        "release cadence rather than the live calendar, which doesn't reach that far back."
    )

    rows = [{"Date": d, "Report": "WASDE", "Source": "USDA WASDE"} for d in wasde_dates]
    rows += [{"Date": d, "Report": label, "Source": "USDA NASS"} for d, label in nass_dates]
    if not rows:
        st.warning("No report dates loaded.")
        return
    frame = pd.DataFrame(rows).sort_values("Date", ascending=False)

    report_types = sorted(frame["Report"].unique())
    controls = st.container(horizontal=True, vertical_alignment="bottom")
    with controls:
        picked = st.multiselect(
            "Report types", report_types,
            default=[r for r in report_types if r != "Crop Progress"],
            key="cal_report_types",
        )
        year_range = st.slider(
            "Year", 2021, as_of.year, (max(2021, as_of.year - 1), as_of.year), key="cal_years",
        )

    filtered = frame[
        frame["Report"].isin(picked)
        & frame["Date"].apply(lambda d: year_range[0] <= d.year <= year_range[1])
    ]
    if filtered.empty:
        st.info("No report dates match the current filters.")
        return

    display = filtered.copy()
    display["Date"] = display["Date"].map(lambda d: d.strftime("%Y-%m-%d"))
    with st.container(key="tablewrap_calendar"):
        st.dataframe(display, hide_index=True, width="stretch",
                    height=min(35 * (len(display) + 1) + 3, 620))
    export_row(display, "report_calendar", key="calendar")
    st.caption(f"{len(filtered):,} report dates shown, {year_range[0]}–{year_range[1]}.")


def _tool_get_live_curve(args: dict, api_key: str, as_of: date) -> dict:
    code = args["product_code"]
    n = args.get("n_contracts", 8)
    curve = load_curve(code, api_key, as_of.isoformat(), n)
    if curve.empty:
        return {"error": f"No live contracts returned for {code}."}
    return {"contracts": [
        {"ticker": r.ticker, "label": friendly_contract(r.ticker, code),
         "expiration": r.expiration.isoformat(), "price": r.price}
        for r in curve.itertuples(index=False)
    ]}


def _tool_get_contract_summary(args: dict, api_key: str, as_of: date) -> dict:
    code, ticker = args["product_code"], args["ticker"]
    bars = load_bars((ticker,), api_key).get(ticker)
    if bars is None or bars.empty:
        return {"error": f"No settlement history for {ticker}."}
    hi_idx, lo_idx = bars["high"].idxmax(), bars["low"].idxmin()
    return {
        "ticker": ticker, "label": friendly_contract(ticker, code), "sessions": len(bars),
        "first_date": str(bars.index.min()), "last_date": str(bars.index.max()),
        "all_time_high": float(bars.loc[hi_idx, "high"]), "high_date": str(hi_idx),
        "all_time_low": float(bars.loc[lo_idx, "low"]), "low_date": str(lo_idx),
        "latest_settle": float(bars["settle"].iloc[-1]), "latest_date": str(bars.index[-1]),
    }


def _tool_get_monthly_high_low(args: dict, api_key: str, as_of: date) -> dict:
    ticker = args["ticker"]
    bars = load_bars((ticker,), api_key).get(ticker)
    if bars is None or bars.empty:
        return {"error": f"No settlement history for {ticker}."}
    return {"ticker": ticker, "months": monthly_high_low(bars).to_dict(orient="records")}


def _tool_get_price_on_date(args: dict, api_key: str, as_of: date) -> dict:
    ticker = args["ticker"]
    target = pd.Timestamp(args["target_date"]).date()
    bars = load_bars((ticker,), api_key).get(ticker)
    if bars is None or bars.empty:
        return {"error": f"No settlement history for {ticker}."}
    eligible = bars[pd.to_datetime(bars.index).date <= target]
    if eligible.empty:
        return {"error": f"No sessions on or before {target} for {ticker}."}
    return {"ticker": ticker, "date": str(eligible.index[-1]), "settle": float(eligible["settle"].iloc[-1])}


def _tool_get_seasonal_stats(args: dict, api_key: str, as_of: date) -> dict:
    code, ticker = args["product_code"], args["ticker"]
    years_back = min(args.get("years_back", 4), MAX_YEARS_BACK)
    curve = load_curve(code, api_key, as_of.isoformat(), 8)
    expiries = dict(zip(curve["ticker"], curve["expiration"])) if not curve.empty else {}
    checkpoints = [90, 60, 30, 14, 7]  # calendar days before expiration

    # by_dte uses the same sign convention as the chart code (negative = before
    # expiration, 0 = expiration) so it can feed _harmonic_seasonal_curve directly.
    by_cp: dict[int, list[float]] = {cp: [] for cp in checkpoints}
    by_dte: dict[str, pd.Series] = {}
    for back in range(years_back + 1):
        t = shift_ticker_year(ticker, code, -back)
        if not t:
            continue
        bars = load_bars((t,), api_key).get(t)
        if bars is None or bars.empty:
            continue
        expiry = expiries.get(t, bars.index.max())
        dte_days = [-(expiry - d).days for d in bars.index]
        settle = bars["settle"].to_numpy()
        by_dte[t] = pd.Series(settle, index=pd.Index(dte_days, name="dte"))
        for cp in checkpoints:
            matches = [i for i, d in enumerate(dte_days) if abs(d - (-cp)) <= 3]
            if matches:
                by_cp[cp].append(float(settle[matches[-1]]))

    harmonic_curve = _harmonic_seasonal_curve(by_dte, window_days=100) if len(by_dte) > 1 else pd.Series(dtype=float)

    def harmonic_at(cp: int) -> float | None:
        if harmonic_curve.empty:
            return None
        target = -cp
        nearest = min(harmonic_curve.index, key=lambda d: abs(d - target))
        return float(harmonic_curve.loc[nearest])

    return {
        "ticker": ticker,
        "checkpoints_days_to_expiry": {
            str(cp): {"average": (sum(v) / len(v)) if v else None,
                      "min": min(v) if v else None, "max": max(v) if v else None,
                      "years_used": len(v),
                      "harmonic_fit": harmonic_at(cp)}
            for cp, v in by_cp.items()
        },
    }


def _tool_get_wasde_dates(args: dict, wasde_dates: list[date]) -> dict:
    start, end = pd.Timestamp(args["start_date"]).date(), pd.Timestamp(args["end_date"]).date()
    return {"dates": [str(d) for d in wasde_dates if start <= d <= end]}


def _tool_get_nass_dates(args: dict, nass_dates: list[tuple[date, str]]) -> dict:
    start, end = pd.Timestamp(args["start_date"]).date(), pd.Timestamp(args["end_date"]).date()
    report_type = args.get("report_type")
    rows = [{"date": str(d), "report": label} for d, label in nass_dates
            if start <= d <= end and (not report_type or report_type.lower() in label.lower())]
    return {"reports": rows}


def make_tool_dispatch(api_key: str, as_of: date, wasde_dates: list[date],
                       nass_dates: list[tuple[date, str]]):
    def dispatch(name: str, args: dict) -> dict:
        if name == "get_live_curve":
            return _tool_get_live_curve(args, api_key, as_of)
        if name == "get_contract_summary":
            return _tool_get_contract_summary(args, api_key, as_of)
        if name == "get_monthly_high_low":
            return _tool_get_monthly_high_low(args, api_key, as_of)
        if name == "get_price_on_date":
            return _tool_get_price_on_date(args, api_key, as_of)
        if name == "get_seasonal_stats":
            return _tool_get_seasonal_stats(args, api_key, as_of)
        if name == "get_wasde_dates":
            return _tool_get_wasde_dates(args, wasde_dates)
        if name == "get_nass_dates":
            return _tool_get_nass_dates(args, nass_dates)
        return {"error": f"Unknown tool {name}"}
    return dispatch


SYSTEM_PROMPT_TEMPLATE = (
    "You are a grain-markets assistant embedded in a Streamlit dashboard covering CBOT corn (ZC), "
    "soybeans (ZS), Chicago/SRW wheat (ZW), and KC/HRW wheat (KE) futures. Today's date is {today}.\n\n"
    "You have tools to look up live and historical futures prices, monthly high/low stats for "
    "specific contracts, and USDA WASDE/NASS report release dates. Use them whenever a question "
    "needs real data — never guess a price or date from memory. When a question names a contract "
    "loosely (e.g. \"December corn\" or \"the front month\"), resolve it via get_live_curve first, "
    "then use the resulting ticker in other calls. Massive's settlement history starts 2021-09-02, "
    "so anything requesting data before that will come back empty — say so rather than inventing "
    "numbers. Keep answers concise and cite the specific numbers/dates your tools returned."
)


@st.fragment
def render_ask_ai_chat(api_key: str, anthropic_key: str, as_of: date,
                       wasde_dates: list[date], nass_dates: list[tuple[date, str]]):
    """Isolated in a fragment so sending a message only reruns this box, not
    the whole page — a full rerun re-fetches/re-renders all four commodities'
    tabs too, which is far too slow to pay on every chat turn. Defined as a
    real module-level function (not a closure inside render_ask_ai) since a
    fresh closure object on every call is what triggers Streamlit's "fragment
    does not exist anymore" error on some reruns — fragments need a stable
    identity across reruns, which only a top-level function reliably gives."""
    if st.session_state.chat_history and st.button("Clear chat", key="clear_chat"):
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("e.g. What's the average corn price 30 days before December expiration?")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        client = anthropic.Anthropic(api_key=anthropic_key)
        dispatch = make_tool_dispatch(api_key, as_of, wasde_dates, nass_dates)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(today=as_of.isoformat())
        api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = run_chat(client, api_messages, dispatch, system_prompt)
                except Exception as e:
                    answer = f"Error calling Claude: {e}"
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


def render_ask_ai(api_key: str, anthropic_key: str, as_of: date,
                  wasde_dates: list[date], nass_dates: list[tuple[date, str]]):
    st.markdown("##### Ask AI")
    st.caption(
        "Ask about corn/soybean/wheat futures prices, monthly highs & lows, seasonal patterns, or "
        "WASDE/NASS report dates. Answers are grounded in live tool calls, not guesses — Massive's "
        "settlement history starts 2021-09-02."
    )
    if not anthropic_key:
        st.info(
            "Add an `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` "
            '(`ANTHROPIC_API_KEY = "..."`) or as an environment variable to enable this tab.'
        )
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    render_ask_ai_chat(api_key, anthropic_key, as_of, wasde_dates, nass_dates)


def main():
    col_logo, col_title = st.columns([1, 6], vertical_alignment="center")
    with col_logo:
        st.image(asset(LOGO_FILE), width=150)
    with col_title:
        st.title("Grain Seasonal Futures & Spreads")
    st.caption(
        "Live CBOT grain futures, aligned across prior contract years to reveal seasonal "
        "patterns in outright price and in calendar spreads. "
        f"Data as of {datetime.now():%b %d, %Y %I:%M %p} · quotes delayed per Massive API."
    )
    st.caption(DATA_START_NOTE)
    st.caption(FND_NOTE)

    api_key = get_api_key()
    if not api_key:
        st.error(
            "No MASSIVE_API_KEY found. Add it to `.streamlit/secrets.toml` "
            '(`MASSIVE_API_KEY = "..."`) or as an environment variable.'
        )
        st.stop()

    as_of = date.today()
    wasde_dates = load_wasde_dates()
    nass_dates = load_nass_dates(tuple(range(2021, as_of.year + 1)))

    with st.sidebar:
        st.markdown("##### Report markers")
        st.caption("Vertical lines on the price/candlestick charts.")
        show_wasde = st.checkbox("WASDE", value=True, key="show_wasde")
        show_nass_major = st.checkbox(
            "NASS (major reports)", value=True, key="show_nass_major",
            help="Crop Production, Grain Stocks, Acreage, Prospective Plantings, "
                 "Winter Wheat & Canola Seedings, Small Grains Summary.",
        )
        show_crop_progress = st.checkbox(
            "NASS Crop Progress (weekly)", value=False, key="show_crop_progress",
            help="Off by default — weekly releases add a lot of lines.",
        )
    report_dates_visible = _selected_report_dates(
        wasde_dates, nass_dates, show_wasde, show_nass_major, show_crop_progress
    )
    anthropic_key = get_anthropic_key()

    tab_labels = [c["label"] for c in COMMODITIES] + ["Cross-commodity spread", "Report calendar", "Ask AI"]
    tabs = st.tabs(tab_labels)
    for tab, commodity in zip(tabs, COMMODITIES):
        with tab:
            st.caption(commodity["sublabel"])
            render_commodity(commodity, api_key, as_of, report_dates_visible)
    with tabs[-3]:
        render_cross_commodity_spread(api_key, as_of, report_dates_visible)
    with tabs[-2]:
        render_report_calendar(wasde_dates, nass_dates, as_of)
    with tabs[-1]:
        render_ask_ai(api_key, anthropic_key, as_of, wasde_dates, nass_dates)


if __name__ == "__main__":
    main()
