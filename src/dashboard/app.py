"""Interactive Streamlit dashboard — Autonomous AI Trading Bot.

Shows backtest results from trained agents. No authentication required.

Launch
------
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.eval.metrics import drawdown_series

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

for _k, _v in {"dark_mode": True}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

if st.session_state.dark_mode:
    T: dict = dict(
        bg="#0d1117", surface="#161b22", surface2="#21262d",
        border="#30363d", text="#e6edf3", muted="#8b949e",
        accent="#58a6ff", green="#3fb950", red="#f85149",
        yellow="#d29922", purple="#bc8cff",
        chart_tpl="plotly_dark",
    )
else:
    T = dict(
        bg="#f6f8fa", surface="#ffffff", surface2="#eaeef2",
        border="#d0d7de", text="#1f2328", muted="#636c76",
        accent="#0969da", green="#1a7f37", red="#cf222e",
        yellow="#9a6700", purple="#6639ba",
        chart_tpl="plotly_white",
    )

ALGO_COLORS: dict[str, str] = {
    "dqn":          "#58a6ff",
    "ppo":          "#3fb950",
    "qlearning":    "#d29922",
    "buy_hold":     "#8b949e",
    "ma_crossover": "#bc8cff",
    "baselines":    "#8b949e",
}
CHART_PALETTE = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#ff7b72", "#56d364"]


def _hex_rgba(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        [data-testid="stMainMenu"]               {{ display:none !important; }}
        [data-testid="stToolbarActionButtonIcon"] {{ display:none !important; }}
        header[data-testid="stHeader"]            {{ display:none !important; }}
        .stApp                                   {{ background:{T['bg']}; }}
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"]         {{ display:none !important; }}
        .block-container                         {{ padding-top:1rem !important; max-width:100%; padding-left:1.5rem; padding-right:1.5rem; }}
        .stTabs [data-baseweb="tab-list"]        {{ background:{T['surface']}; border-radius:10px; padding:4px; gap:4px; border:1px solid {T['border']}; }}
        .stTabs [data-baseweb="tab"]             {{ background:transparent; border-radius:8px; color:{T['muted']}; font-weight:500; border:none; padding:8px 20px; }}
        .stTabs [aria-selected="true"]           {{ background:{T['accent']}22; color:{T['accent']} !important; border:1px solid {T['accent']}44 !important; }}
        .stTabs [data-baseweb="tab-border"],
        .stTabs [data-baseweb="tab-highlight"]   {{ display:none; }}
        .stButton > button                       {{ border-radius:8px; font-weight:600; transition:all .2s; cursor:pointer; }}
        p, li, label, .stMarkdown               {{ color:{T['text']}; }}
        h1,h2,h3,h4                             {{ color:{T['text']}; }}
        .stCaption                              {{ color:{T['muted']}; }}
        .stDataFrame                             {{ border-radius:12px; overflow:hidden; border:1px solid {T['border']}; }}
        [data-testid="stExpander"]               {{ background:{T['surface']} !important; border:1px solid {T['border']} !important; border-radius:10px !important; }}
        [data-testid="stExpander"] summary       {{ background:{T['surface']} !important; color:{T['text']} !important; border-radius:10px !important; }}
        [data-testid="stExpander"] summary:hover {{ background:{T['surface2']} !important; }}
        [data-testid="stExpanderDetails"]        {{ background:{T['surface']} !important; }}
        [data-baseweb="select"] > div            {{ background:{T['surface']} !important; border-color:{T['border']} !important; }}
        [data-baseweb="menu"]                    {{ background:{T['surface']} !important; }}
        [data-baseweb="option"]                  {{ background:{T['surface']} !important; color:{T['text']} !important; }}
        [data-baseweb="option"]:hover            {{ background:{T['surface2']} !important; }}
        [data-baseweb="tag"]                     {{ background:{T['accent']}22 !important; color:{T['text']} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Reusable components
# ---------------------------------------------------------------------------

def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}44;'
        f'padding:2px 10px;border-radius:20px;font-size:.78rem;font-weight:700;">{text}</span>'
    )


def _kpi_card(label: str, value: str, suffix: str = "", color: str | None = None) -> str:
    vc = color or T["text"]
    return (
        f'<div style="background:{T["surface"]};border:1px solid {T["border"]};'
        f'border-radius:14px;padding:1.1rem 1rem;text-align:center;flex:1;min-width:120px;">'
        f'<div style="color:{T["muted"]};font-size:.72rem;text-transform:uppercase;'
        f'letter-spacing:.08em;margin-bottom:.4rem;">{label}</div>'
        f'<div style="color:{vc};font-size:1.55rem;font-weight:700;line-height:1.1;">'
        f'{value}<span style="font-size:.95rem;font-weight:500;">{suffix}</span></div></div>'
    )


def _section_hdr(title: str, sub: str = "") -> None:
    sub_html = (
        f'<p style="color:{T["muted"]};margin:.2rem 0 0;font-size:.88rem;">{sub}</p>'
        if sub else ""
    )
    st.markdown(
        f'<div style="margin:1.5rem 0 1rem;">'
        f'<h3 style="color:{T["text"]};margin:0;font-size:1.15rem;font-weight:700;">{title}</h3>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


def _chart_base(fig: go.Figure, *, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        template=T["chart_tpl"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        title=dict(text=title, font=dict(size=14, color=T["text"]), x=0),
        font=dict(family="Inter,system-ui,sans-serif", color=T["muted"]),
        legend=dict(
            bgcolor=T["surface"], bordercolor=T["border"], borderwidth=1,
            font=dict(color=T["text"], size=12), x=0, y=1.08, orientation="h",
        ),
        xaxis=dict(gridcolor=T["border"], linecolor=T["border"], tickfont=dict(color=T["muted"])),
        yaxis=dict(gridcolor=T["border"], linecolor=T["border"], tickfont=dict(color=T["muted"])),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )


def _load_metrics(run_dir: Path) -> dict[str, dict]:
    result = {}
    for f in run_dir.glob("*metrics.json"):
        tag = f.stem.replace("_metrics", "") or "agent"
        with open(f) as fp:
            result[tag] = json.load(fp)
    return result


def _load_equity(run_dir: Path) -> dict[str, pd.Series]:
    curves = {}
    for f in run_dir.glob("*equity_curve.csv"):
        tag = f.stem.replace("_equity_curve", "") or "agent"
        df = pd.read_csv(f, parse_dates=True, index_col=0)
        curves[tag] = df.iloc[:, 0]
    return curves


def _load_trades(run_dir: Path) -> dict[str, pd.DataFrame]:
    trades = {}
    for f in run_dir.glob("*trades.csv"):
        tag = f.stem.replace("_trades", "") or "agent"
        trades[tag] = pd.read_csv(f)
    return trades


def _parse_run_name(name: str) -> dict[str, str]:
    parts = name.split("_")
    if len(parts) >= 4:
        d = parts[0]
        date_fmt = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) >= 8 else d
        algo = parts[2].lower()
        symbol = "_".join(parts[3:])
        return {"date": date_fmt, "algo": algo, "symbol": symbol}
    return {"date": "—", "algo": name.lower(), "symbol": "—"}


# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------

def _topbar() -> None:
    _inject_css()
    col_title, col_gap, col_theme = st.columns([5, 7, 1.2])
    with col_title:
        st.markdown(
            f'<div style="display:flex;align-items:center;height:2.6rem;gap:.6rem;">'
            f'<span style="font-size:1.5rem;">🤖</span>'
            f'<span style="color:{T["text"]};font-size:1.15rem;font-weight:800;'
            f'letter-spacing:-.01em;">AI Trading Bot</span>'
            f'<span style="color:{T["muted"]};font-size:.72rem;margin-left:.4rem;">'
            f'UWE · UFCFXK-30-3</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_theme:
        if st.button(
            "🌞" if st.session_state.dark_mode else "🌙",
            help=None,
            use_container_width=True,
            key="topbar_theme",
        ):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    st.markdown(
        f'<div style="border-bottom:1px solid {T["border"]};margin:.2rem 0 1rem;"></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------

def _page_dashboard() -> None:
    runs = _list_runs()

    st.markdown(
        f'<div style="padding:.5rem 0 .8rem;">'
        f'<h1 style="color:{T["text"]};margin:0;font-size:1.75rem;font-weight:800;'
        f'letter-spacing:-.02em;">📊 Results Dashboard</h1>'
        f'<p style="color:{T["muted"]};margin:.2rem 0 0;font-size:.88rem;">'
        f'Compare trained RL agents and baselines on the test set</p></div>',
        unsafe_allow_html=True,
    )

    if not runs:
        st.markdown(
            f'<div style="background:{T["surface"]};border:1px solid {T["border"]};'
            f'border-radius:14px;padding:3rem;text-align:center;">'
            f'<div style="font-size:3rem;margin-bottom:1rem;">📭</div>'
            f'<h3 style="color:{T["text"]};margin:0 0 .5rem;">No runs yet</h3>'
            f'<p style="color:{T["muted"]};margin:0 0 .5rem;">Run the pipeline first:</p>'
            f'<code style="color:{T["accent"]};">python -m src.cli.pipeline --symbol BTC-USD</code>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    run_names = [r.name for r in runs]

    with st.expander("🔬 Select experiment runs to display", expanded=True):
        sel_col, tag_col = st.columns([3, 4])
        with sel_col:
            selected_runs: list[str] = st.multiselect(
                "Runs",
                options=run_names,
                default=[run_names[0]] if run_names else [],
                label_visibility="collapsed",
                key="selected_runs",
            )
        with tag_col:
            for rn in selected_runs:
                info = _parse_run_name(rn)
                color = ALGO_COLORS.get(info["algo"], T["muted"])
                st.markdown(
                    f'<span style="display:inline-flex;align-items:center;gap:.4rem;'
                    f'background:{color}18;border:1px solid {color}44;border-radius:20px;'
                    f'padding:2px 10px;margin:.15rem .25rem;font-size:.75rem;">'
                    f'{_badge(info["algo"].upper(), color)}'
                    f'<span style="color:{T["muted"]};">'
                    f'{info["symbol"]} · {info["date"]}</span></span>',
                    unsafe_allow_html=True,
                )

    if not selected_runs:
        st.info("Select one or more experiment runs above.")
        return

    selected_dirs = [RUNS_DIR / name for name in selected_runs]

    st.markdown(
        f'<div style="border-top:1px solid {T["border"]};margin:.5rem 0 1.5rem;"></div>',
        unsafe_allow_html=True,
    )

    # ── KPI cards ─────────────────────────────────────────────────────────
    KPI_KEYS = [
        ("cumulative_return", "Total Return",  "%",  True),
        ("sharpe_ratio",      "Sharpe Ratio",  "",   True),
        ("max_drawdown",      "Max Drawdown",  "%",  False),
        ("win_rate",          "Win Rate",      "%",  True),
        ("num_trades",        "# Trades",      "",   None),
    ]
    for rd in selected_dirs:
        for tag, m in _load_metrics(rd).items():
            color = ALGO_COLORS.get(tag.lower(), T["accent"])
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.6rem;">'
                f'{_badge(tag.upper(), color)}'
                f'<span style="color:{T["muted"]};font-size:.8rem;">{rd.name[:50]}</span></div>',
                unsafe_allow_html=True,
            )
            cards = '<div style="display:flex;gap:.75rem;flex-wrap:wrap;margin-bottom:1.2rem;">'
            for key, label, suffix, hib in KPI_KEYS:
                raw = m.get(key, 0.0)
                if key in ("cumulative_return", "win_rate"):
                    display = f"{raw * 100:+.2f}"
                elif key == "max_drawdown":
                    display = f"{raw * 100:.2f}"
                elif key == "num_trades":
                    display = str(int(raw))
                else:
                    display = f"{raw:.3f}"
                if hib is True:
                    vc = T["green"] if raw >= 0 else T["red"]
                elif hib is False:
                    vc = T["red"] if raw > 0.05 else T["green"]
                else:
                    vc = T["text"]
                cards += _kpi_card(label, display, suffix, vc)
            cards += "</div>"
            st.markdown(cards, unsafe_allow_html=True)

    st.markdown(
        f'<div style="border-top:1px solid {T["border"]};margin:0 0 1.5rem;"></div>',
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────
    t_eq, t_dd, t_tr, t_pos, t_met = st.tabs([
        "📈  Equity Curves",
        "📉  Drawdown",
        "🔄  Trades",
        "📊  Positions",
        "🏆  Metrics Table",
    ])

    with t_eq:
        _section_hdr("Equity Curves", "Portfolio value over the test period")
        fig = go.Figure()
        for i, rd in enumerate(selected_dirs):
            for tag, series in _load_equity(rd).items():
                c = ALGO_COLORS.get(tag.lower(), CHART_PALETTE[i % len(CHART_PALETTE)])
                fig.add_trace(go.Scatter(
                    x=series.index, y=series.values, mode="lines",
                    name=f"{tag.upper()} — {rd.name[:25]}",
                    line=dict(color=c, width=2.5),
                    fill="tozeroy", fillcolor=_hex_rgba(c, .07),
                ))
        fig.add_hline(
            y=10_000, line_dash="dot", line_color=T["muted"], opacity=.5,
            annotation_text="Starting Capital $10,000",
            annotation_position="bottom right",
            annotation_font_color=T["muted"],
        )
        _chart_base(fig, height=460)
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    with t_dd:
        _section_hdr("Drawdown", "% decline from peak portfolio value")
        fig = go.Figure()
        dd_rows = []
        for i, rd in enumerate(selected_dirs):
            for tag, series in _load_equity(rd).items():
                c = ALGO_COLORS.get(tag.lower(), CHART_PALETTE[i % len(CHART_PALETTE)])
                eq = series.values.astype(float)
                dd = drawdown_series(eq) * 100
                fig.add_trace(go.Scatter(
                    x=series.index, y=-dd, mode="lines",
                    name=f"{tag.upper()} — {rd.name[:25]}",
                    line=dict(color=c, width=2),
                    fill="tozeroy", fillcolor=_hex_rgba(c, .12),
                ))
                dd_rows.append({
                    "Strategy":     tag.upper(),
                    "Run":          rd.name[:40],
                    "Max Drawdown": f"{dd.max():.2f}%",
                    "Avg Drawdown": f"{dd.mean():.2f}%",
                })
        _chart_base(fig, height=400)
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)
        if dd_rows:
            st.dataframe(pd.DataFrame(dd_rows), use_container_width=True, hide_index=True)

    with t_tr:
        _section_hdr("Trade Signals", "Buy / Sell markers overlaid on price")
        sym_input = st.text_input("Symbol (for price data lookup)", value="BTC-USD")
        price_path = PROJECT_ROOT / "data" / "processed" / sym_input / "test_raw.csv"
        if price_path.exists():
            price_df = pd.read_csv(price_path, parse_dates=True, index_col=0)
            if "Close" in price_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_df.index, y=price_df["Close"],
                    mode="lines", name="BTC Price",
                    line=dict(color=T["muted"], width=1.5),
                ))
                trade_rows = []
                for i, rd in enumerate(selected_dirs):
                    for tag, tdf in _load_trades(rd).items():
                        c = ALGO_COLORS.get(tag.lower(), CHART_PALETTE[i % len(CHART_PALETTE)])
                        for act, sym_mk, clr, lbl in [
                            ("buy",  "triangle-up",   T["green"], "Buy"),
                            ("sell", "triangle-down", T["red"],   "Sell"),
                        ]:
                            sub = tdf[tdf["action"] == act]
                            if not sub.empty and "step" in sub.columns:
                                steps = sub["step"].values.astype(int)
                                valid = steps[steps < len(price_df)]
                                if len(valid):
                                    fig.add_trace(go.Scatter(
                                        x=price_df.index[valid],
                                        y=price_df["Close"].values[valid],
                                        mode="markers",
                                        name=f"{tag.upper()} {lbl}",
                                        marker=dict(
                                            color=clr, size=9, symbol=sym_mk,
                                            line=dict(color=T["surface"], width=1),
                                        ),
                                    ))
                        trade_rows.append({
                            "Strategy":     tag.upper(),
                            "Run":          rd.name[:35],
                            "Buys":         len(tdf[tdf["action"] == "buy"]),
                            "Sells":        len(tdf[tdf["action"] == "sell"]),
                            "Total Trades": len(tdf),
                        })
                _chart_base(fig, height=460)
                fig.update_yaxes(tickprefix="$")
                st.plotly_chart(fig, use_container_width=True)
                if trade_rows:
                    st.dataframe(
                        pd.DataFrame(trade_rows), use_container_width=True, hide_index=True
                    )
        else:
            st.warning(
                f"Price data not found at `{price_path}`. "
                "Run `python -m src.cli.preprocess --symbol BTC-USD` first."
            )

    with t_pos:
        _section_hdr("Portfolio Value Over Time", "Per-strategy equity progression")
        for i, rd in enumerate(selected_dirs):
            for tag, series in _load_equity(rd).items():
                c = ALGO_COLORS.get(tag.lower(), CHART_PALETTE[i % len(CHART_PALETTE)])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=series.index, y=series.values, mode="lines",
                    name=tag.upper(),
                    line=dict(color=c, width=2.5),
                    fill="tozeroy", fillcolor=_hex_rgba(c, .1),
                ))
                _chart_base(fig, title=f"{tag.upper()} · {rd.name[:35]}", height=300)
                fig.update_yaxes(tickprefix="$")
                st.plotly_chart(fig, use_container_width=True)

    with t_met:
        _section_hdr("Full Metrics Comparison")
        rows = []
        for rd in selected_dirs:
            for tag, m in _load_metrics(rd).items():
                rows.append({
                    "Strategy":    tag.upper(),
                    "Run":         rd.name[:35],
                    "Return":      f"{m.get('cumulative_return', 0)*100:+.2f}%",
                    "Ann. Return": f"{m.get('annualised_return', 0)*100:+.2f}%",
                    "Volatility":  f"{m.get('annualised_volatility', 0)*100:.2f}%",
                    "Sharpe":      f"{m.get('sharpe_ratio', 0):.3f}",
                    "Max DD":      f"{m.get('max_drawdown', 0)*100:.2f}%",
                    "Win Rate":    f"{m.get('win_rate', 0)*100:.1f}%",
                    "Trades":      int(m.get("num_trades", 0)),
                    "Exposure":    f"{m.get('exposure_time', 0)*100:.1f}%",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if len(rows) > 1:
                st.markdown(
                    f'<div style="border-top:1px solid {T["border"]};'
                    f'margin:1.5rem 0 1rem;"></div>',
                    unsafe_allow_html=True,
                )
                _section_hdr("Visual Comparison")
                raw_rows = []
                for rd in selected_dirs:
                    for tag, m in _load_metrics(rd).items():
                        raw_rows.append({
                            "label":      tag.upper(),
                            "Return %":   round(m.get("cumulative_return", 0) * 100, 2),
                            "Sharpe":     round(m.get("sharpe_ratio", 0), 3),
                            "Win Rate %": round(m.get("win_rate", 0) * 100, 1),
                        })
                fig_bar = go.Figure()
                for i, row in enumerate(raw_rows):
                    c = CHART_PALETTE[i % len(CHART_PALETTE)]
                    fig_bar.add_trace(go.Bar(
                        name=row["label"],
                        x=["Return %", "Sharpe", "Win Rate %"],
                        y=[row["Return %"], row["Sharpe"], row["Win Rate %"]],
                        marker_color=c,
                        marker_line_color=T["border"],
                        marker_line_width=1,
                    ))
                fig_bar.update_layout(barmode="group")
                _chart_base(fig_bar, title="Key Metrics Comparison", height=380)
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No metrics found. Run evaluation first.")


# ---------------------------------------------------------------------------
# How to Run page
# ---------------------------------------------------------------------------

def _page_commands() -> None:
    _section_hdr("💻 How to Run", "Step-by-step commands to train and evaluate agents")
    st.code(
        "# Step 1 — Download BTC price data from Yahoo Finance\n"
        "python -m src.cli.download_data --symbol BTC-USD --start 2020-01-01 --end 2025-01-01\n\n"
        "# Step 2 — Preprocess: feature engineering + train/val/test split + normalisation\n"
        "python -m src.cli.preprocess --symbol BTC-USD\n\n"
        "# Step 3 — Train each agent\n"
        "python -m src.cli.train --algo qlearning --symbol BTC-USD\n"
        "python -m src.cli.train --algo dqn       --symbol BTC-USD\n"
        "python -m src.cli.train --algo ppo       --symbol BTC-USD\n\n"
        "# Step 4 — Run baselines (Buy & Hold, MA Crossover)\n"
        "python -m src.cli.baselines --symbol BTC-USD\n\n"
        "# Step 5 — Evaluate each trained run\n"
        "python -m src.cli.evaluate --run_dir runs/<run_name> --symbol BTC-USD\n\n"
        "# OR: run everything in one command\n"
        "python -m src.cli.pipeline --symbol BTC-USD",
        language="bash",
    )


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

_topbar()

page = st.radio(
    "Page",
    options=["📊 Dashboard", "💻 How to Run"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown(
    f'<div style="border-top:1px solid {T["border"]};margin:.4rem 0 1.2rem;"></div>',
    unsafe_allow_html=True,
)

if page == "📊 Dashboard":
    _page_dashboard()
else:
    _page_commands()