"""
SIF Performance Dashboard (Client Edition)
-------------------------------------------
Presents fund performance against Nifty 50 / Nifty 500 benchmarks over any
selected date range: indexed growth chart, period return summary,
risk & capture-ratio table, and trailing returns.

Data is fetched automatically from the fund NAV API and yfinance; no manual
data-source selection or debug tooling is exposed to the end user.

Run with:  streamlit run fund_dashboard.py
"""

import datetime as dt

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="SIF Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BENCHMARK_NAMES = {"Nifty 50", "Nifty 500", "NIFTY 50", "NIFTY 500"}

# --------------------------------------------------------------------------
# Look & feel — rich, elegant dark theme built for an HNI-client audience.
# --------------------------------------------------------------------------
BG = "#0B1220"           # page background
CARD_BG = "#131B2E"      # panels / tables / chart card
CARD_BG_ALT = "#161F36"  # zebra striping / hover
BORDER = "#242D42"
PRIMARY = "#F1F4F8"      # heading / primary text
BODY_TEXT = "#C9D1DD"    # table body text
MUTED = "#8B95A5"
ACCENT = "#D4AF37"       # gold
ACCENT_SOFT = "rgba(212, 175, 55, 0.10)"
POSITIVE = "#3ECF8E"
NEGATIVE = "#FF6B6B"

PALETTE = [
    "#4FD1C5", "#D4AF37", "#818CF8", "#FF8A65", "#34D399",
    "#60A5FA", "#F472B6", "#FBBF24", "#A78BFA", "#F87171",
]

st.markdown(
    f"""
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        [data-testid="stHeader"] {{visibility: hidden;}}

        html, body, .stApp {{
            background-color: {BG};
            font-family: 'Inter', sans-serif;
        }}

        .block-container {{
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }}

        h1, h2, h3, h4, h5 {{
            color: {PRIMARY};
            font-family: 'Playfair Display', serif;
            font-weight: 650;
            letter-spacing: 0.01em;
        }}

        .dash-title {{
            font-family: 'Playfair Display', serif;
            font-size: 2.1rem;
            font-weight: 700;
            color: {ACCENT};
            margin-bottom: 0.15rem;
            letter-spacing: 0.01em;
        }}
        .dash-subtitle {{
            color: {MUTED};
            font-size: 0.95rem;
            margin-bottom: 1.6rem;
            font-family: 'Inter', sans-serif;
        }}
        .section-divider {{
            border: none;
            border-top: 1px solid {BORDER};
            margin: 2rem 0 1.3rem 0;
        }}
        .section-caption {{
            color: {MUTED};
            font-size: 0.85rem;
            margin-bottom: 0.9rem;
            line-height: 1.5;
        }}

        [data-testid="stMetricValue"] {{
            color: {PRIMARY};
        }}

        /* Plotly chart card */
        [data-testid="stPlotlyChart"] {{
            background-color: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 14px;
            box-shadow: 0 8px 28px rgba(0,0,0,0.4);
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {CARD_BG};
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] h3 {{
            color: {ACCENT};
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            margin-top: 1.2rem;
            margin-bottom: 0.5rem;
        }}

        /* Buttons */
        .stButton>button {{
            background-color: transparent;
            color: {ACCENT};
            border: 1px solid {ACCENT};
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.15s ease;
        }}
        .stButton>button:hover {{
            background-color: {ACCENT_SOFT};
            border-color: {ACCENT};
            color: {ACCENT};
        }}

        /* Inputs */
        div[data-baseweb="select"] > div,
        .stDateInput input {{
            background-color: {CARD_BG_ALT} !important;
            border-color: {BORDER} !important;
            color: {PRIMARY} !important;
            border-radius: 6px !important;
        }}

        /* HNI-styled static tables */
        .hni-table-wrap {{
            background-color: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 6px;
            margin-bottom: 0.6rem;
            box-shadow: 0 8px 28px rgba(0,0,0,0.4);
            overflow-x: auto;
        }}
        .hni-table-wrap table {{
            width: 100%;
            border-collapse: collapse;
            background: transparent;
            font-family: 'Inter', sans-serif;
        }}
        .hni-table-wrap thead th {{
            background-color: {CARD_BG_ALT};
            color: {ACCENT};
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 12px 16px;
            border-bottom: 1px solid {BORDER};
            text-align: right;
            white-space: nowrap;
        }}
        .hni-table-wrap thead th:first-child {{
            text-align: left;
        }}
        .hni-table-wrap tbody th {{
            padding: 10px 16px;
            border-bottom: 1px solid {BORDER};
            color: {PRIMARY};
            font-weight: 500;
            text-align: left;
            white-space: nowrap;
            background: transparent;
        }}
        .hni-table-wrap tbody td {{
            padding: 10px 16px;
            border-bottom: 1px solid {BORDER};
            color: {BODY_TEXT};
            font-size: 0.86rem;
            text-align: right;
            white-space: nowrap;
            font-variant-numeric: tabular-nums;
        }}
        .hni-table-wrap tbody tr:last-child th,
        .hni-table-wrap tbody tr:last-child td {{
            border-bottom: none;
        }}
        .hni-table-wrap tbody tr:hover td,
        .hni-table-wrap tbody tr:hover th {{
            background-color: {ACCENT_SOFT};
        }}
        .hni-table-scroll {{
            max-height: 420px;
            overflow-y: auto;
        }}

        /* Heatmap tables have 13+ columns (Year + 12 months + Annual) — use
           tighter padding/type so everything fits without horizontal scroll. */
        .hni-heatmap-table thead th {{
            padding: 10px 8px;
            font-size: 0.66rem;
        }}
        .hni-heatmap-table tbody th {{
            padding: 8px 10px;
            font-size: 0.82rem;
        }}
        .hni-heatmap-table tbody td {{
            padding: 8px 6px;
            font-size: 0.78rem;
        }}
        .hni-heatmap-table thead th:last-child,
        .hni-heatmap-table tbody td:last-child {{
            border-left: 1px solid {BORDER};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Core calculations
# --------------------------------------------------------------------------
def get_series_names(df: pd.DataFrame):
    all_cols = [c for c in df.columns if c != "Date"]
    benchmarks = [c for c in all_cols if c in BENCHMARK_NAMES]
    funds = [c for c in all_cols if c not in BENCHMARK_NAMES]
    return funds, benchmarks


def snap_to_previous_trading_day(df: pd.DataFrame, target: dt.date) -> pd.Timestamp:
    """If `target` isn't a date present in the sheet (weekend/holiday/gap),
    snap back to the most recent available date on or before it. If target
    is earlier than every date in the sheet, fall back to the earliest
    available date."""
    ts = pd.Timestamp(target)
    valid = df["Date"][df["Date"] <= ts]
    if valid.empty:
        return df["Date"].min()
    return valid.max()


def filter_available_in_window(df: pd.DataFrame, start: dt.date, end: dt.date, cols):
    """A series is 'available' for the selected window only if it has a
    valid (non-NaN) NAV on every trading day the sheet has between start
    and end (i.e. it was already listed for the whole window)."""
    start_ts = snap_to_previous_trading_day(df, start)
    end_ts = snap_to_previous_trading_day(df, end)
    window = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)]
    if window.empty:
        return [], window, start_ts, end_ts
    available = [c for c in cols if window[c].notna().all()]
    return available, window, start_ts, end_ts


def build_indexed(window: pd.DataFrame, cols, base=100.0):
    """Rebase each column so its first value in the window = base."""
    idx_df = window[["Date"] + cols].copy()
    for c in cols:
        first_val = idx_df[c].iloc[0]
        idx_df[c] = idx_df[c] / first_val * base
    return idx_df


def trailing_return(full_df: pd.DataFrame, col: str, as_of: pd.Timestamp, days_back, kind="calendar"):
    """Compute trailing return for a column as of `as_of`, looking back
    `days_back` calendar days (kind='calendar') or using the first row
    (kind='inception'). Returns None if data isn't available that far back."""
    series = full_df[["Date", col]].dropna()
    if series.empty:
        return None

    series = series[series["Date"] <= as_of]
    if series.empty:
        return None

    end_val = series[col].iloc[-1]
    end_date = series["Date"].iloc[-1]

    if kind == "inception":
        start_val = series[col].iloc[0]
        start_date = series["Date"].iloc[0]
        if start_date == end_date:
            return None
        years = (end_date - start_date).days / 365.25
        if years <= 0:
            return None
        if years < 1:
            return (end_val / start_val - 1) * 100
        cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
        return cagr

    target_date = as_of - pd.Timedelta(days=days_back)
    prior = series[series["Date"] <= target_date]
    if prior.empty:
        return None
    start_val = prior[col].iloc[-1]
    start_date = prior["Date"].iloc[-1]

    years = (end_date - start_date).days / 365.25
    if days_back >= 365 and years >= 1:
        # annualize for periods of 1yr or more
        if start_val <= 0:
            return None
        return ((end_val / start_val) ** (1 / years) - 1) * 100
    if start_val == 0:
        return None
    return (end_val / start_val - 1) * 100


PERIODS = [
    ("1D", 1),
    ("1W", 7),
    ("1M", 30),
    ("3M", 91),
    ("6M", 182),
    ("1Y", 365),
    ("3Y", 365 * 3),
    ("5Y", 365 * 5),
]


def build_trailing_table(full_df: pd.DataFrame, cols, as_of: pd.Timestamp):
    rows = []
    for c in cols:
        row = {"Fund": c}
        for label, days in PERIODS:
            row[label] = trailing_return(full_df, c, as_of, days)
        row["Since Inception (CAGR/Abs)"] = trailing_return(full_df, c, as_of, None, kind="inception")
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Fund")
    return table


# --------------------------------------------------------------------------
# Risk & Capture Ratio metrics (selected period)
# --------------------------------------------------------------------------
def max_drawdown(nav_series: pd.Series) -> float:
    """Maximum peak-to-trough decline (%) within the series."""
    running_max = nav_series.cummax()
    drawdown = nav_series / running_max - 1.0
    return drawdown.min() * 100


def annualized_std_dev(returns: pd.Series, periods_per_year=252) -> float:
    """Annualized standard deviation (%) of daily returns."""
    if returns.empty or len(returns) < 2:
        return None
    return returns.std(ddof=1) * np.sqrt(periods_per_year) * 100


def up_down_capture(fund_returns: pd.Series, bench_returns: pd.Series):
    """Compounded up-capture / down-capture ratio (%) of a fund vs a
    benchmark over the same set of trading days.

    Up-capture = cumulative fund return on days benchmark was positive
                 divided by cumulative benchmark return on those same days.
    Down-capture = the same, restricted to days benchmark was negative.
    """
    aligned = pd.concat([fund_returns, bench_returns], axis=1, join="inner").dropna()
    if aligned.empty:
        return None, None
    aligned.columns = ["fund", "bench"]

    up_mask = aligned["bench"] > 0
    down_mask = aligned["bench"] < 0

    def _cum(series):
        return (1 + series).prod() - 1

    up_capture = None
    if up_mask.sum() > 0:
        bench_up = _cum(aligned.loc[up_mask, "bench"])
        fund_up = _cum(aligned.loc[up_mask, "fund"])
        if bench_up != 0:
            up_capture = (fund_up / bench_up) * 100

    down_capture = None
    if down_mask.sum() > 0:
        bench_down = _cum(aligned.loc[down_mask, "bench"])
        fund_down = _cum(aligned.loc[down_mask, "fund"])
        if bench_down != 0:
            down_capture = (fund_down / bench_down) * 100

    return up_capture, down_capture


def build_risk_table(window: pd.DataFrame, cols, reference_benchmark: str):
    """Std dev, up/down capture, capture ratio and max drawdown for each
    column in `cols`, computed on the selected-period window, measured
    against `reference_benchmark`."""
    dated = window.set_index("Date")
    bench_ret = dated[reference_benchmark].pct_change().dropna()

    rows = []
    for c in cols:
        nav = dated[c].dropna()
        ret = nav.pct_change().dropna()

        row = {"Fund": c}
        row["Std Dev (Ann., %)"] = annualized_std_dev(ret)
        row["Max Drawdown (%)"] = max_drawdown(nav) if not nav.empty else None

        if c == reference_benchmark:
            row["Up Capture (%)"] = 100.0
            row["Down Capture (%)"] = 100.0
            row["Capture Ratio"] = 1.0
        else:
            up_cap, down_cap = up_down_capture(ret, bench_ret)
            row["Up Capture (%)"] = up_cap
            row["Down Capture (%)"] = down_cap
            if up_cap is not None and down_cap not in (None, 0):
                row["Capture Ratio"] = up_cap / down_cap
            else:
                row["Capture Ratio"] = None

        rows.append(row)

    table = pd.DataFrame(rows).set_index("Fund")
    return table


# --------------------------------------------------------------------------
# Monthly returns (Year x Month grid) — same data serves as both the
# "monthly return table" and the "monthly heatmap": the heatmap is just
# this table with the cell shading added, not a separate calculation.
# --------------------------------------------------------------------------
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def build_monthly_returns(
    full_df: pd.DataFrame, col: str, as_of: pd.Timestamp = None, since: pd.Timestamp = None
) -> pd.DataFrame:
    """Year x Month grid of monthly returns (%) for `col`, using its full
    available history up to `as_of` (not limited to the sidebar's selected
    period — a multi-year view is the whole point of this table). Each
    monthly return is NAV(month-end) / NAV(previous month-end) - 1. An
    'Annual' column gives the compounded return for each calendar year.

    `since`, if given, drops any data before that date first — used to
    trim a benchmark's heatmap down to the same starting point as a
    shorter-history fund, so the two are directly comparable year-for-year.

    Returns an empty DataFrame if there's under two months of history.
    """
    series = full_df[["Date", col]].dropna().copy()
    if since is not None:
        series = series[series["Date"] >= since]
    if as_of is not None:
        series = series[series["Date"] <= as_of]
    if series.empty:
        return pd.DataFrame()

    series = series.set_index("Date")[col].sort_index()
    monthly_nav = series.resample("ME").last()
    monthly_ret = monthly_nav.pct_change().dropna() * 100

    if monthly_ret.empty:
        return pd.DataFrame()

    result = monthly_ret.reset_index()
    result.columns = ["Date", "Return"]
    result["Year"] = result["Date"].dt.year
    result["Month"] = result["Date"].dt.strftime("%b")

    pivot = result.pivot(index="Year", columns="Month", values="Return")
    pivot = pivot.reindex(columns=[m for m in MONTH_ORDER if m in pivot.columns])

    def _compound_year(row):
        vals = row.dropna()
        if vals.empty:
            return np.nan
        return ((vals / 100 + 1).prod() - 1) * 100

    pivot["Annual"] = pivot.apply(_compound_year, axis=1)
    pivot.index.name = "Year"
    pivot.index = pivot.index.astype(str)
    return pivot



# Rendered as static HTML (via pandas Styler) rather than st.dataframe so
# the full HNI table design (header, banding, colours) is guaranteed to
# render, independent of the Streamlit grid component's own theming.
# --------------------------------------------------------------------------
def _color_signed(val):
    if pd.isna(val):
        return f"color: {MUTED}"
    if val > 0:
        return f"color: {POSITIVE}; font-weight: 600"
    if val < 0:
        return f"color: {NEGATIVE}; font-weight: 600"
    return f"color: {BODY_TEXT}"


def _color_ratio(val, pivot=1.0):
    if pd.isna(val):
        return f"color: {MUTED}"
    if val > pivot:
        return f"color: {POSITIVE}; font-weight: 600"
    if val < pivot:
        return f"color: {NEGATIVE}; font-weight: 600"
    return f"color: {BODY_TEXT}"


def _color_pivot(val, pivot, higher_is_better=True):
    """Colour a value green/red based on which side of `pivot` it falls on
    (rather than which side of zero). Used for capture-ratio-style metrics
    where the meaningful threshold is 100% / 1.0, not 0 — e.g. a Down
    Capture of -20% (fund rose while the benchmark fell) is favourable and
    must show green, even though the raw number is negative; a Down
    Capture of 130% is unfavourable and must show red, even though the raw
    number is positive."""
    if pd.isna(val):
        return f"color: {MUTED}"
    if val == pivot:
        return f"color: {BODY_TEXT}"
    is_better = (val > pivot) if higher_is_better else (val < pivot)
    color = POSITIVE if is_better else NEGATIVE
    return f"color: {color}; font-weight: 600"


def render_table(
    df: pd.DataFrame,
    decimals=2,
    pct_cols=(),
    ratio_cols=(),
    color_pct_cols=(),
    color_ratio_cols=(),
    color_pivot_cols=None,
    scroll=False,
):
    """Render a DataFrame as a styled, static HNI-look table.

    - `pct_cols`: numeric columns formatted with `decimals` places + a % sign.
    - `ratio_cols`: numeric columns formatted with `decimals` places, no %.
    - Any other numeric column defaults to `decimals` places, no %.
    - `color_pct_cols` / `color_ratio_cols`: sign-based green/red colouring
      (pivot at 0 — for genuine returns, where positive is always favourable).
    - `color_pivot_cols`: dict of {column: (pivot, higher_is_better)} for
      metrics where the meaningful threshold isn't zero — e.g. capture
      ratios, where the pivot is 100% / 1.0 and a negative raw number can
      still be the favourable outcome (see Down Capture).
    """
    color_pivot_cols = color_pivot_cols or {}
    fmt = {}
    for c in df.columns:
        if c in pct_cols:
            fmt[c] = f"{{:.{decimals}f}}%"
        elif c in ratio_cols:
            fmt[c] = f"{{:.{decimals}f}}"
        elif pd.api.types.is_numeric_dtype(df[c]):
            fmt[c] = f"{{:.{decimals}f}}"

    styler = df.style.format(fmt, na_rep="—")
    map_fn = styler.map if hasattr(styler, "map") else styler.applymap
    for c in color_pct_cols:
        if c in df.columns:
            styler = map_fn(_color_signed, subset=[c])
    for c in color_ratio_cols:
        if c in df.columns:
            styler = map_fn(_color_ratio, subset=[c])
    for c, (pivot, higher_is_better) in color_pivot_cols.items():
        if c in df.columns:
            styler = map_fn(
                lambda v, p=pivot, h=higher_is_better: _color_pivot(v, p, h),
                subset=[c],
            )

    html = styler.to_html()
    wrap_class = "hni-table-wrap hni-table-scroll" if scroll else "hni-table-wrap"
    st.markdown(f'<div class="{wrap_class}">{html}</div>', unsafe_allow_html=True)


def _heatmap_cell_color(val, vmax):
    """Background tint whose intensity scales with |val| relative to vmax
    (the largest magnitude in that column-group), green for gains, red for
    losses — a translucent overlay on the dark card background so text
    stays legible at every intensity."""
    if pd.isna(val):
        return f"color: {MUTED}; background-color: {CARD_BG_ALT};"
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    intensity = min(abs(val) / vmax, 1.0)
    alpha = 0.10 + 0.38 * intensity
    rgb = "62, 207, 142" if val >= 0 else "255, 107, 107"
    weight = 700 if intensity > 0.45 else 500
    return f"background-color: rgba({rgb}, {alpha:.3f}); color: {PRIMARY}; font-weight: {weight};"


def render_heatmap(df: pd.DataFrame, decimals=2, annual_col="Annual"):
    """Render a Year x Month returns grid as a static, colour-shaded table —
    this IS the monthly return table; the shading is layered on top of the
    same numbers, not a separate calculation."""
    month_cols = [c for c in df.columns if c != annual_col]

    fmt = {c: f"{{:.{decimals}f}}%" for c in df.columns}
    styler = df.style.format(fmt, na_rep="—")
    map_fn = styler.map if hasattr(styler, "map") else styler.applymap

    if month_cols:
        monthly_vals = df[month_cols].values.astype(float)
        finite = monthly_vals[np.isfinite(monthly_vals)]
        vmax_monthly = np.max(np.abs(finite)) if finite.size else 1.0
        for c in month_cols:
            styler = map_fn(lambda v, vmax=vmax_monthly: _heatmap_cell_color(v, vmax), subset=[c])

    if annual_col in df.columns:
        annual_vals = df[annual_col].values.astype(float)
        finite_a = annual_vals[np.isfinite(annual_vals)]
        vmax_annual = np.max(np.abs(finite_a)) if finite_a.size else 1.0
        styler = map_fn(lambda v, vmax=vmax_annual: _heatmap_cell_color(v, vmax), subset=[annual_col])

    html = styler.to_html()
    st.markdown(f'<div class="hni-table-wrap hni-heatmap-table">{html}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="dash-title">SIF Performance Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-subtitle">Fund performance vs. Nifty 50 / Nifty 500, indexed and risk-adjusted, '
    'over any selected period.</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Data loading — auto-fetch only; no data-source toggle or debug tooling
# exposed to the client.
# --------------------------------------------------------------------------
import upvaly_client as uc
import data_pipeline as dp
import benchmark_client as bc

if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0

with st.sidebar:
    st.markdown("### Funds")

    with st.spinner("Loading fund list..."):
        meta = dp.load_all_metadata(force_refresh=False)

    categories = sorted({m.get("category", "Unknown") for m in meta.values()})
    category_filter = st.multiselect("Filter by category", categories)
    fund_filter = st.multiselect("Or select specific fund(s)", uc.FUND_LIST)
    st.caption("Leave blank to include every tracked fund.")

    if st.button("Refresh data"):
        st.session_state.refresh_counter += 1
        dp.load_all_metadata(force_refresh=True)
        st.cache_data.clear()

if category_filter or fund_filter:
    selected_names = set()
    if category_filter:
        selected_names |= {n for n, m in meta.items() if m.get("category", "Unknown") in category_filter}
    if fund_filter:
        selected_names |= set(fund_filter)
    selected_names = [n for n in uc.FUND_LIST if n in selected_names]
else:
    selected_names = uc.FUND_LIST


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_fetch(names_key: tuple, refresh_key: int):
    """Cached on (fund selection, refresh-button click count) — not on the
    date range, since that's filtered client-side afterward."""
    _meta = dp.load_all_metadata(force_refresh=False)
    _df = dp.build_wide_dataframe(list(names_key), _meta)
    return _df


with st.spinner("Loading fund performance data..."):
    df = _cached_fetch(tuple(selected_names), st.session_state.refresh_counter)

fetch_ok = not (df.empty or "Date" not in df.columns or df.shape[1] <= 1)

if not fetch_ok:
    st.error("We're unable to load fund data right now. Please try again shortly.")
    st.stop()

funds, benchmarks = get_series_names(df)
all_cols = funds + benchmarks

# If a benchmark's data source (yfinance) failed on every attempt so far,
# its column never gets created at all — this is different from a benchmark
# being incomplete for a specific date range, and no amount of changing the
# date range will fix it. Surface it clearly instead of letting it vanish
# without explanation.
EXPECTED_BENCHMARKS = ["Nifty 50", "Nifty 500"]
missing_benchmarks = [b for b in EXPECTED_BENCHMARKS if b not in df.columns]
if missing_benchmarks:
    fetch_error_detail = "; ".join(
        f"{b}: {bc.LAST_FETCH_ERRORS[b]}" for b in missing_benchmarks if b in bc.LAST_FETCH_ERRORS
    )
    st.warning(
        f"{', '.join(missing_benchmarks)} data is currently unavailable from the market data "
        "provider (this is unrelated to the selected date range). Fund figures above are "
        "unaffected. Try \u201cRefresh data\u201d in the sidebar; if it persists, the data source "
        "may need attention."
        + (f"\n\nDetail: {fetch_error_detail}" if fetch_error_detail else "")
    )

if df["Date"].isna().all() or df.empty:
    st.error("No usable data is currently available.")
    st.stop()

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

# Default start date is 2025-12-31, clamped into whatever range the data
# actually has (falls back to min_date if the data doesn't go back that far).
DEFAULT_START = dt.date(2025, 12, 31)
default_start = min(max(DEFAULT_START, min_date), max_date)

with st.sidebar:
    st.markdown("### Date Range")
    start_date = st.date_input("Start date", value=default_start, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

    st.markdown("### Benchmarks")
    default_benchmarks = [b for b in benchmarks if b == "Nifty 50"] or benchmarks
    show_benchmarks = st.multiselect("Display benchmarks", benchmarks, default=default_benchmarks)

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Only include funds available for the entire window; benchmarks are
# always assumed available (they existed long before any fund).
available_funds, window, actual_start, actual_end = filter_available_in_window(
    df, start_date, end_date, funds
)
available_benchmarks = [b for b in show_benchmarks if window[b].notna().all()] if not window.empty else []

excluded_funds = [f for f in funds if f not in available_funds]
excluded_benchmarks = [b for b in show_benchmarks if b not in available_benchmarks]

plot_cols = available_funds + available_benchmarks

if not plot_cols:
    st.error("No funds or benchmarks have complete data for this date range. Try a shorter or more recent window.")
    st.stop()

st.markdown(f"#### Performance: {actual_start.date():%d %b %Y} — {actual_end.date():%d %b %Y}")

if actual_start.date() != start_date or actual_end.date() != end_date:
    st.markdown(
        f'<div class="section-caption">Dates snapped to the nearest available trading days: '
        f'{actual_start.date():%d %b %Y} / {actual_end.date():%d %b %Y}.</div>',
        unsafe_allow_html=True,
    )

if excluded_funds:
    st.markdown(
        f'<div class="section-caption">Not shown (incomplete data for this period): '
        f'{", ".join(excluded_funds)}</div>',
        unsafe_allow_html=True,
    )

if excluded_benchmarks:
    st.markdown(
        f'<div class="section-caption">Benchmark not shown for this period (data incomplete for '
        f'one or more days in this window — try a slightly shorter or earlier end date): '
        f'{", ".join(excluded_benchmarks)}</div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Indexed journey
# --------------------------------------------------------------------------
indexed = build_indexed(window, plot_cols, base=100.0)

fig = go.Figure()
color_i = 0
for c in plot_cols:
    is_bench = c in benchmarks
    fig.add_trace(
        go.Scatter(
            x=indexed["Date"],
            y=indexed[c],
            mode="lines",
            name=c,
            line=dict(
                width=3 if is_bench else 2,
                dash="dash" if is_bench else "solid",
                color=PALETTE[color_i % len(PALETTE)],
            ),
        )
    )
    color_i += 1

fig.update_layout(
    height=550,
    hovermode="x unified",
    yaxis_title="Indexed Value (Start = 100)",
    xaxis_title="Date",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=BODY_TEXT)),
    margin=dict(t=30, b=10),
    plot_bgcolor=CARD_BG,
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color=BODY_TEXT, family="Inter, sans-serif"),
)
fig.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Summary — Selected Period Return (always sorted high → low, 2 decimals)
# --------------------------------------------------------------------------
st.markdown("#### Summary — Selected Period Return")
period_summary = pd.DataFrame({
    "Start NAV": window[plot_cols].iloc[0],
    "End NAV": window[plot_cols].iloc[-1],
    "Indexed End (Start=100)": indexed[plot_cols].iloc[-1],
    "Period Return (%)": (indexed[plot_cols].iloc[-1] - 100),
})
period_summary.index.name = "Fund"
period_summary = period_summary.sort_values("Period Return (%)", ascending=False)

render_table(
    period_summary,
    decimals=2,
    pct_cols=["Period Return (%)"],
    color_pct_cols=["Period Return (%)"],
)

st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Risk & Capture Ratio table (selected period — updates with date range,
# everything rounded to a single decimal place)
# --------------------------------------------------------------------------
st.markdown("#### Risk & Capture Ratio — Selected Period")

if len(available_benchmarks) > 1:
    reference_benchmark = st.selectbox(
        "Benchmark used for capture-ratio calculations",
        available_benchmarks,
        index=0,
    )
elif len(available_benchmarks) == 1:
    reference_benchmark = available_benchmarks[0]
else:
    reference_benchmark = None

if reference_benchmark is None:
    st.info("Select at least one benchmark to view risk & capture-ratio metrics.")
else:
    risk_table = build_risk_table(window, plot_cols, reference_benchmark)
    st.markdown(
        f'<div class="section-caption">Computed over the selected period, measured against '
        f'{reference_benchmark}. Standard deviation is annualized; up/down capture reflect '
        f'compounded returns on days the benchmark was positive / negative respectively. '
        f'A negative Down Capture means the fund posted a <em>positive</em> return on days the '
        f'benchmark fell — a defensive characteristic, and favourable (shown in green) even '
        f'though the number itself is negative.</div>',
        unsafe_allow_html=True,
    )
    render_table(
        risk_table,
        decimals=1,
        pct_cols=["Std Dev (Ann., %)", "Max Drawdown (%)", "Up Capture (%)", "Down Capture (%)"],
        ratio_cols=["Capture Ratio"],
        color_pct_cols=["Max Drawdown (%)"],
        color_ratio_cols=["Capture Ratio"],
        color_pivot_cols={
            "Up Capture (%)": (100.0, True),    # higher = captured more upside = favourable
            "Down Capture (%)": (100.0, False), # lower = captured less downside = favourable
        },
    )

st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Trailing returns table (uses FULL history, as-of end_date, 2 decimals)
# --------------------------------------------------------------------------
st.markdown("#### Trailing Returns")
st.markdown(
    '<div class="section-caption">Computed on each fund\'s/benchmark\'s full available history as of the '
    'selected end date. Periods of 1 year or more are annualized (CAGR); '
    'shorter periods are absolute returns.</div>',
    unsafe_allow_html=True,
)
trailing_cols = all_cols
trailing_table = build_trailing_table(df, trailing_cols, actual_end)
return_cols = list(trailing_table.columns)
render_table(
    trailing_table,
    decimals=2,
    pct_cols=return_cols,
    color_pct_cols=return_cols,
    scroll=True,
)

st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Monthly Returns Heatmap — Year x Month grid, full history as-of end_date.
# A dropdown picks the fund (there can be many); benchmarks are shown
# directly below since there are typically only one or two.
# --------------------------------------------------------------------------
st.markdown("#### Monthly Returns Heatmap")
st.markdown(
    '<div class="section-caption">Each cell is a calendar-month return; shading intensity reflects the '
    'size of that month\'s return (green = gain, red = loss) relative to the largest move in the table, '
    'so the heatmap and the underlying monthly return table are the same view — the numbers are shown '
    'directly in each cell. The Annual column is the compounded return for that calendar year. Computed '
    'on full available history as of the selected end date, independent of the date range selected above. '
    'The benchmark heatmap below is trimmed to start from the selected fund\'s own inception, so the two '
    'line up year-for-year for easy comparison.</div>',
    unsafe_allow_html=True,
)

fund_inception = None
if funds:
    heatmap_fund = st.selectbox("Select a fund", funds, index=0, key="heatmap_fund_select")
    fund_dates = df[["Date", heatmap_fund]].dropna()["Date"]
    fund_inception = fund_dates.min() if not fund_dates.empty else None
    monthly_fund = build_monthly_returns(df, heatmap_fund, as_of=actual_end)
    if monthly_fund.empty:
        st.info(f"Not enough monthly history yet for {heatmap_fund}.")
    else:
        render_heatmap(monthly_fund)
else:
    st.info("No funds available for this selection.")

if show_benchmarks:
    st.markdown("##### Benchmark")
    for b in show_benchmarks:
        monthly_bench = build_monthly_returns(df, b, as_of=actual_end, since=fund_inception)
        if monthly_bench.empty:
            st.info(f"Not enough monthly history yet for {b}.")
            continue
        if len(show_benchmarks) > 1:
            st.markdown(f"**{b}**")
        render_heatmap(monthly_bench)

st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

with st.expander("Methodology"):
    st.markdown(
        """
- **Rebasing to 100**: each fund/benchmark's first NAV in the selected period is set to 100, with
  all subsequent values scaled proportionally — this makes funds with very different NAV levels
  directly comparable on one chart.
- **Availability**: a fund is only shown for a given period if it has a valid NAV on every trading
  day within that period (i.e. it was already live for the full window).
- **Trailing returns**: computed on full available history as of the selected end date. Periods ≥ 1
  year are annualized (CAGR); shorter periods are absolute returns.
- **Standard deviation**: annualized volatility of daily returns over the selected period.
- **Up / Down capture**: the proportion of the benchmark's compounded gain (or loss) that a fund
  captured on days the benchmark was positive (or negative) over the selected period. A **negative
  Down Capture** means the fund posted a positive return on days the benchmark fell — i.e. it moved
  in the opposite direction to the market on those days. This is a genuinely defensive characteristic
  and is treated as favourable (shown in green), even though the figure itself is negative.
- **Capture ratio**: up-capture divided by down-capture — a ratio above 1.0 indicates the fund
  has, over this period, captured more upside than downside relative to the benchmark.
- **Maximum drawdown**: the largest peak-to-trough decline in NAV within the selected period.
- **Monthly returns heatmap**: each month's return is NAV at that month's last trading day versus
  NAV at the prior month's last trading day. This uses the fund's/benchmark's full available
  history (not the date range selected above), as of the selected end date, since a multi-year
  view is the point of this table. The Annual column compounds that year's monthly returns.
"""
    )

