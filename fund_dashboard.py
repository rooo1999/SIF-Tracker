"""
SIF Performance Dashboard (Client Edition)
-------------------------------------------
Presents fund performance against Nifty 50 / Nifty 500 benchmarks over any
selected date range: indexed growth chart, period return summary, indexed
NAV table, risk & capture-ratio table, and trailing returns.

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
# Look & feel — clean, HNI-client-facing styling. Hides Streamlit chrome
# (menu / footer / "Deploy" bar) and applies a restrained, professional
# colour palette.
# --------------------------------------------------------------------------
PRIMARY = "#0B1F3A"      # deep navy
ACCENT = "#B8892E"       # muted gold
POSITIVE = "#1B7A43"
NEGATIVE = "#B02A2A"
MUTED = "#6B7280"

st.markdown(
    f"""
    <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
        }}

        h1, h2, h3 {{
            color: {PRIMARY};
            font-weight: 650;
        }}

        .dash-title {{
            font-size: 1.9rem;
            font-weight: 700;
            color: {PRIMARY};
            margin-bottom: 0.1rem;
        }}
        .dash-subtitle {{
            color: {MUTED};
            font-size: 0.95rem;
            margin-bottom: 1.4rem;
        }}
        .section-divider {{
            border: none;
            border-top: 1px solid #E5E7EB;
            margin: 1.8rem 0 1.2rem 0;
        }}
        [data-testid="stMetricValue"] {{
            color: {PRIMARY};
        }}
        .stDataFrame {{
            border: 1px solid #E5E7EB;
            border-radius: 6px;
        }}
        section[data-testid="stSidebar"] {{
            background-color: #F7F8FA;
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
# Styling helpers for polished, HNI-facing tables
# --------------------------------------------------------------------------
def _color_signed(val):
    if pd.isna(val):
        return f"color: {MUTED}"
    if val > 0:
        return f"color: {POSITIVE}; font-weight: 600"
    if val < 0:
        return f"color: {NEGATIVE}; font-weight: 600"
    return ""


def style_return_table(df: pd.DataFrame, pct_cols):
    fmt = {c: "{:.2f}%" for c in pct_cols if c in df.columns}
    styler = df.style.format(fmt, na_rep="—")
    map_fn = styler.map if hasattr(styler, "map") else styler.applymap
    for c in pct_cols:
        if c in df.columns:
            styler = map_fn(_color_signed, subset=[c])
    return styler


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

plot_cols = available_funds + available_benchmarks

if not plot_cols:
    st.error("No funds or benchmarks have complete data for this date range. Try a shorter or more recent window.")
    st.stop()

st.markdown(f"#### Performance: {actual_start.date():%d %b %Y} — {actual_end.date():%d %b %Y}")

if actual_start.date() != start_date or actual_end.date() != end_date:
    st.caption(
        f"Dates snapped to the nearest available trading days: "
        f"{actual_start.date():%d %b %Y} / {actual_end.date():%d %b %Y}."
    )

if excluded_funds:
    st.caption(f"Not shown (incomplete data for this period): {', '.join(excluded_funds)}")

# --------------------------------------------------------------------------
# Indexed journey
# --------------------------------------------------------------------------
indexed = build_indexed(window, plot_cols, base=100.0)

PALETTE = [
    "#0B1F3A", "#B8892E", "#2E7D5B", "#8E44AD", "#C0392B",
    "#1F6FB2", "#D68910", "#5D6D7E", "#117864", "#943126",
]

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
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=30, b=10),
    plot_bgcolor="white",
    font=dict(color=PRIMARY),
)
fig.update_xaxes(showgrid=True, gridcolor="#EEF0F3")
fig.update_yaxes(showgrid=True, gridcolor="#EEF0F3")
st.plotly_chart(fig, use_container_width=True)

st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Summary — Selected Period Return (always sorted high → low)
# --------------------------------------------------------------------------
st.markdown("#### Summary — Selected Period Return")
period_summary = pd.DataFrame({
    "Start NAV": window[plot_cols].iloc[0],
    "End NAV": window[plot_cols].iloc[-1],
    "Indexed End (Start=100)": indexed[plot_cols].iloc[-1],
    "Period Return (%)": (indexed[plot_cols].iloc[-1] - 100),
}).round(2)
period_summary.index.name = "Fund"
period_summary = period_summary.sort_values("Period Return (%)", ascending=False)

st.dataframe(
    style_return_table(period_summary, ["Period Return (%)"]),
    use_container_width=True,
)

st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Indexed values table
# --------------------------------------------------------------------------
st.markdown("#### Indexed NAV Table (Start = 100)")
display_indexed = indexed.copy()
display_indexed["Date"] = display_indexed["Date"].dt.strftime("%d %b %Y")
st.dataframe(display_indexed.round(2), use_container_width=True, height=350)

st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Risk & Capture Ratio table (selected period — updates with date range)
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
    risk_table = risk_table.round(2)
    st.caption(
        f"Computed over the selected period, measured against {reference_benchmark}. "
        "Standard deviation is annualized; up/down capture reflect compounded returns "
        "on days the benchmark was positive / negative respectively."
    )
    st.dataframe(
        style_return_table(
            risk_table,
            ["Max Drawdown (%)", "Up Capture (%)", "Down Capture (%)"],
        ),
        use_container_width=True,
    )

st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Trailing returns table (uses FULL history, as-of end_date)
# --------------------------------------------------------------------------
st.markdown("#### Trailing Returns")
st.caption(
    "Computed on each fund's/benchmark's full available history as of the "
    "selected end date. Periods of 1 year or more are annualized (CAGR); "
    "shorter periods are absolute returns."
)
trailing_cols = all_cols
trailing_table = build_trailing_table(df, trailing_cols, actual_end).round(2)
return_cols = [c for c in trailing_table.columns]
st.dataframe(
    style_return_table(trailing_table, return_cols),
    use_container_width=True,
    height=400,
)

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
  captured on days the benchmark was positive (or negative) over the selected period.
- **Capture ratio**: up-capture divided by down-capture — a ratio above 1.0 indicates the fund
  has, over this period, captured more upside than downside relative to the benchmark.
- **Maximum drawdown**: the largest peak-to-trough decline in NAV within the selected period.
"""
    )
