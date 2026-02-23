"""Interactive Streamlit dashboard — Autonomous AI Trading Bot.

Launch
------
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import auth
from src.dashboard import trader as trader_mod
from src.eval.metrics import drawdown_series

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"

# ---------------------------------------------------------------------------
# Page config  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

for _k, _v in {
    "dark_mode": True,
    "logged_in": False,
    "username": "",
    "page": "dashboard",       # "dashboard" | "settings" | "commands"
    "auth_tab": "home",        # "home" | "login" | "register"
    "is_trading": False,
    "trading_status": "Idle",  # "Idle" | "Live" | "Failed" | "Stopped"
    "trading_mode": "Spot",      # "Spot" | "Futures"
    "trading_error": "",         # last error message, shown after rerun
    "council_runs": [],          # list[str] of selected run names
    "council_result": {},        # last council vote result dict
    "market_info": "",           # last fetched price info string
    "entry_price": 0.0,          # BTC price when trading started
    "entry_time": "",            # ISO datetime string of trade entry
    "profit_target_pct": 2.0,   # take-profit reference line %
    "trade_log": [],             # list[dict] of all executed trades
    "loaded_models": None,       # cached model objects (reused across decisions)
    "last_decision_time": "",    # ISO datetime of last auto-decision
    "auto_trade_interval": 60,   # seconds between automatic bot decisions
}.items():
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
    "dqn": "#58a6ff", "ppo": "#3fb950",
    "qlearning": "#d29922", "baseline": "#8b949e",
}
CHART_PALETTE = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#ff7b72", "#56d364"]


def _hex_rgba(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        /* ── Chrome removal ──────────────────────────────────── */
        [data-testid="stMainMenu"]               {{ display:none !important; }}
        [data-testid="stToolbarActionButtonIcon"] {{ display:none !important; }}
        header[data-testid="stHeader"]            {{ display:none !important; }}

        /* ── App background ──────────────────────────────────── */
        .stApp                                   {{ background:{T['bg']}; }}
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"]         {{ display:none !important; }}
        .block-container                         {{ padding-top:1rem !important; max-width:100%; padding-left:1.5rem; padding-right:1.5rem; }}

        /* ── Tabs ────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"]        {{ background:{T['surface']}; border-radius:10px; padding:4px; gap:4px; border:1px solid {T['border']}; }}
        .stTabs [data-baseweb="tab"]             {{ background:transparent; border-radius:8px; color:{T['muted']}; font-weight:500; border:none; padding:8px 20px; }}
        .stTabs [aria-selected="true"]           {{ background:{T['accent']}22; color:{T['accent']} !important; border:1px solid {T['accent']}44 !important; }}
        .stTabs [data-baseweb="tab-border"],
        .stTabs [data-baseweb="tab-highlight"]   {{ display:none; }}

        /* ── Buttons ─────────────────────────────────────────── */
        .stButton > button                                    {{ border-radius:8px; font-weight:600; transition:all .2s; cursor:pointer; }}
        .stButton > button[data-testid="stBaseButton-secondary"]
                                                              {{ background:{T['surface2']}; color:{T['text']}; border:1px solid {T['border']}; }}
        .stButton > button[data-testid="stBaseButton-secondary"]:hover
                                                              {{ border-color:{T['accent']}; color:{T['accent']}; background:{T['accent']}15; }}
        .stButton > button[data-testid="stBaseButton-primary"]
                                                              {{ background:{T['accent']}; color:#fff; border:1px solid {T['accent']}; }}
        .stButton > button[data-testid="stBaseButton-primary"]:hover
                                                              {{ background:{T['accent']}cc; border-color:{T['accent']}; color:#fff; }}

        /* ── Inputs ──────────────────────────────────────────── */
        .stTextInput input, .stMultiSelect [data-baseweb="select"] > div
                                                 {{ background:{T['surface2']}; border-color:{T['border']}; color:{T['text']}; border-radius:8px; }}

        /* ── Dataframe ───────────────────────────────────────── */
        .stDataFrame                             {{ border-radius:12px; overflow:hidden; border:1px solid {T['border']}; }}

        /* ── Typography ──────────────────────────────────────── */
        p, li, label, .stMarkdown               {{ color:{T['text']}; }}
        h1,h2,h3,h4                             {{ color:{T['text']}; }}
        .stCaption                              {{ color:{T['muted']}; }}

        /* ── Nav button active state ─────────────────────────── */
        .nav-active > button                    {{ border-color:{T['accent']} !important; color:{T['accent']} !important; background:{T['accent']}11 !important; }}

        /* ── Light-mode extra overrides ──────────────────────── */
        {"" if st.session_state.dark_mode else f"""
        .stApp                                  {{ background:{T['bg']} !important; color:{T['text']} !important; }}
        [data-testid="stSidebar"] *             {{ color:{T['text']} !important; }}
        """}
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_css()

# ---------------------------------------------------------------------------
# Reusable HTML components
# ---------------------------------------------------------------------------

def _divider() -> None:
    st.markdown(f'<div style="border-top:1px solid {T["border"]};margin:.8rem 0;"></div>',
                unsafe_allow_html=True)


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}44;'
        f'padding:2px 10px;border-radius:20px;font-size:.78rem;font-weight:700;">{text}</span>'
    )


def _card(content_html: str, padding: str = "1.2rem") -> None:
    st.markdown(
        f'<div style="background:{T["surface"]};border:1px solid {T["border"]};'
        f'border-radius:14px;padding:{padding};">{content_html}</div>',
        unsafe_allow_html=True,
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
    sub_html = f'<p style="color:{T["muted"]};margin:.2rem 0 0;font-size:.88rem;">{sub}</p>' if sub else ""
    st.markdown(
        f'<div style="margin:1.5rem 0 1rem;">'
        f'<h3 style="color:{T["text"]};margin:0;font-size:1.15rem;font-weight:700;">{title}</h3>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


def _chart_base(fig: go.Figure, *, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        template=T["chart_tpl"], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, title=dict(text=title, font=dict(size=14, color=T["text"]), x=0),
        font=dict(family="Inter,system-ui,sans-serif", color=T["muted"]),
        legend=dict(bgcolor=T["surface"], bordercolor=T["border"], borderwidth=1,
                    font=dict(color=T["text"], size=12), x=0, y=1.08, orientation="h"),
        xaxis=dict(gridcolor=T["border"], linecolor=T["border"], tickfont=dict(color=T["muted"])),
        yaxis=dict(gridcolor=T["border"], linecolor=T["border"], tickfont=dict(color=T["muted"])),
        hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()], key=lambda p: p.name, reverse=True)


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
        return {"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "algo": parts[2].upper(),
                "symbol": "_".join(parts[3:])}
    return {"date": "—", "algo": name, "symbol": "—"}


# ===========================================================================
# PAGE: Auth  (home landing | login | sign up)
# ===========================================================================

def _page_auth() -> None:
    """Top-bar nav + three sub-pages: home, login, register."""
    tab = st.session_state.auth_tab  # "home" | "login" | "register"

    # ── Shared top bar ────────────────────────────────────────────────────
    c_login, c_signup, c_gap, c_theme = st.columns([1.6, 1.6, 8, 1.2])
    with c_login:
        if st.button("Login", width='stretch',
                     type="primary" if tab == "login" else "secondary",
                     key="auth_btn_login"):
            st.session_state.auth_tab = "login"
            st.rerun()
    with c_signup:
        if st.button("Sign Up", width='stretch',
                     type="primary" if tab == "register" else "secondary",
                     key="auth_btn_signup"):
            st.session_state.auth_tab = "register"
            st.rerun()
    with c_theme:
        if st.button("🌞" if st.session_state.dark_mode else "🌙",
                     help="Toggle theme", width='stretch',
                     key="auth_btn_theme"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown(
        f'<div style="border-bottom:1px solid {T["border"]};margin:.2rem 0 1.5rem;"></div>',
        unsafe_allow_html=True,
    )

    # ── SUB-PAGE: Login ───────────────────────────────────────────────────
    if tab == "login":
        _, form_col, _ = st.columns([2, 3, 2])
        with form_col:
            st.markdown(
                f"""
                <div style="background:{T['surface']};border:1px solid {T['border']};
                            border-radius:16px;padding:2rem 2rem 1.5rem;margin-bottom:1rem;">
                    <div style="text-align:center;margin-bottom:1.6rem;">
                        <div style="font-size:2.2rem;">🤖</div>
                        <h2 style="color:{T['text']};margin:.5rem 0 .2rem;font-size:1.4rem;
                                   font-weight:800;">Welcome back</h2>
                        <p style="color:{T['muted']};font-size:.85rem;margin:0;">
                            Sign in to your account
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            login_user = st.text_input("Username", key="li_user", placeholder="your username")
            login_pass = st.text_input("Password", key="li_pass",
                                       placeholder="••••••••", type="password")
            st.markdown("<div style='height:.4rem;'></div>", unsafe_allow_html=True)
            if st.button("Login  →", width='stretch', type="primary", key="btn_login_submit"):
                ok, msg = auth.login(login_user, login_pass)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username = login_user.strip().lower()
                    st.session_state.page = "dashboard"
                    st.session_state.auth_tab = "home"
                    st.rerun()
                else:
                    st.error(msg)
            st.markdown(
                f'<p style="color:{T["muted"]};font-size:.8rem;text-align:center;margin-top:1rem;">'
                f"Don't have an account?</p>",
                unsafe_allow_html=True,
            )
            if st.button("Create Account", width='stretch', key="btn_login_goto_register"):
                st.session_state.auth_tab = "register"
                st.rerun()

    # ── SUB-PAGE: Sign Up ─────────────────────────────────────────────────
    elif tab == "register":
        _, form_col, _ = st.columns([1.5, 4, 1.5])
        with form_col:
            st.markdown(
                f"""
                <div style="background:{T['surface']};border:1px solid {T['border']};
                            border-radius:16px;padding:2rem 2rem 1.5rem;margin-bottom:1rem;">
                    <div style="text-align:center;margin-bottom:1.6rem;">
                        <div style="font-size:2.2rem;">🤖</div>
                        <h2 style="color:{T['text']};margin:.5rem 0 .2rem;font-size:1.4rem;
                                   font-weight:800;">Create Account</h2>
                        <p style="color:{T['muted']};font-size:.85rem;margin:0;">
                            UWE Digital Systems Project · UFCFXK-30-3
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Account details ───────────────────────────────────────────
            _section_hdr("Account Details")
            reg_user = st.text_input("Username", key="rg_user", placeholder="choose a username")
            c_pw, c_cf = st.columns(2)
            with c_pw:
                reg_pass = st.text_input("Password", key="rg_pass",
                                         placeholder="min 6 characters", type="password")
            with c_cf:
                reg_conf = st.text_input("Confirm password", key="rg_conf",
                                         placeholder="repeat password", type="password")

            # ── Bybit API keys ────────────────────────────────────────────
            st.markdown(
                f'<div style="border-top:1px solid {T["border"]};margin:1.2rem 0 .8rem;"></div>',
                unsafe_allow_html=True,
            )
            _section_hdr("⚡ Bybit API Keys", "Optional — you can add these later in Account Settings")
            api_mode = st.radio(
                "Account type",
                options=["Demo Account (recommended)", "Testnet"],
                index=0, horizontal=True, key="rg_api_mode",
            )
            reg_key    = st.text_input("API Key",    key="rg_api_key",
                                       placeholder="Paste your Bybit API key",    type="password")
            reg_secret = st.text_input("API Secret", key="rg_api_secret",
                                       placeholder="Paste your Bybit API secret", type="password")

            if st.button("🔌  Test API Connection", width='stretch', key="btn_reg_test_api"):
                if not reg_key or not reg_secret:
                    st.warning("Enter both API Key and API Secret before testing.")
                else:
                    use_demo_test = api_mode.startswith("Demo")
                    with st.spinner("Connecting to Bybit…"):
                        ok, msg = auth.test_bybit_connection(reg_key, reg_secret, use_demo_test)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

            st.markdown(
                f'<p style="color:{T["muted"]};font-size:.75rem;margin-top:.3rem;">'
                f'🔒 Keys are stored locally only and never transmitted to any third party.</p>',
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:.4rem;'></div>", unsafe_allow_html=True)
            if st.button("Create Account  →", width='stretch', type="primary", key="btn_register_submit"):
                if reg_pass != reg_conf:
                    st.error("Passwords do not match.")
                else:
                    use_demo_reg = api_mode.startswith("Demo")
                    ok, msg = auth.register(reg_user, reg_pass,
                                            reg_key, reg_secret, use_demo_reg)
                    if ok:
                        st.success(msg + " You can now log in.")
                        st.session_state.auth_tab = "login"
                        st.rerun()
                    else:
                        st.error(msg)

            st.markdown(
                f'<p style="color:{T["muted"]};font-size:.8rem;text-align:center;margin-top:1rem;">'
                f'Already have an account?</p>',
                unsafe_allow_html=True,
            )
            if st.button("Sign In", width='stretch', key="btn_register_goto_login"):
                st.session_state.auth_tab = "login"
                st.rerun()

    # ── SUB-PAGE: Home (landing) ──────────────────────────────────────────
    else:
        st.markdown(
            f"""
            <div style="max-width:680px;margin:4rem auto 0;text-align:center;padding:0 1rem;">
                <div style="font-size:4rem;margin-bottom:1rem;">🤖</div>
                <h1 style="color:{T['text']};font-size:2.2rem;font-weight:800;
                           letter-spacing:-.03em;margin-bottom:.6rem;">
                    Autonomous AI Trading Bot
                </h1>
                <p style="color:{T['muted']};font-size:1rem;margin-bottom:2.5rem;line-height:1.7;">
                    UWE Digital Systems Project · UFCFXK-30-3<br>
                    Train, evaluate and compare RL agents on cryptocurrency data.
                    Connect your Bybit demo account to simulate live trading.
                </p>
                <div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-bottom:2.5rem;">
                    <div style="background:{T['surface']};border:1px solid {T['border']};
                                border-radius:12px;padding:1.1rem 1.4rem;min-width:140px;">
                        <div style="font-size:1.5rem;">📈</div>
                        <div style="color:{T['text']};font-weight:600;margin-top:.4rem;">RL Agents</div>
                        <div style="color:{T['muted']};font-size:.8rem;">Q-Learning · DQN · PPO</div>
                    </div>
                    <div style="background:{T['surface']};border:1px solid {T['border']};
                                border-radius:12px;padding:1.1rem 1.4rem;min-width:140px;">
                        <div style="font-size:1.5rem;">🛡️</div>
                        <div style="color:{T['text']};font-weight:600;margin-top:.4rem;">Risk Engine</div>
                        <div style="color:{T['muted']};font-size:.8rem;">Stop-loss · ATR sizing</div>
                    </div>
                    <div style="background:{T['surface']};border:1px solid {T['border']};
                                border-radius:12px;padding:1.1rem 1.4rem;min-width:140px;">
                        <div style="font-size:1.5rem;">⚡</div>
                        <div style="color:{T['text']};font-weight:600;margin-top:.4rem;">Bybit Demo</div>
                        <div style="color:{T['muted']};font-size:.8rem;">Paper trading API</div>
                    </div>
                </div>
                <p style="color:{T['muted']};font-size:.85rem;">
                    Use the <strong style="color:{T['accent']};">Login</strong> or
                    <strong style="color:{T['accent']};">Sign Up</strong> buttons above to get started.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# Shared top bar (logged in)
# ===========================================================================

def _topbar_logged_in() -> None:
    """Horizontal top navigation bar replacing the sidebar."""
    user = auth.get_user(st.session_state.username)
    has_api = bool(user.get("bybit_api_key"))
    api_status = "Bybit connected" if has_api else "No API connected"
    api_color  = T["green"] if has_api else T["muted"]

    c_profile, c_title, c_gap, c_theme = st.columns([2, 4, 6, 1.4])

    # ── Profile popover (top-left) ────────────────────────────────────────
    with c_profile:
        with st.popover("👤  " + st.session_state.username, width='stretch'):
            # User info header inside popover
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:.75rem;
                            padding:.4rem 0 .8rem;border-bottom:1px solid {T['border']};
                            margin-bottom:.6rem;">
                    <div style="background:{T['accent']}22;border:2px solid {T['accent']}44;
                                border-radius:50%;width:40px;height:40px;display:flex;
                                align-items:center;justify-content:center;font-size:1.2rem;">
                        👤
                    </div>
                    <div>
                        <div style="color:{T['text']};font-weight:700;font-size:.9rem;">
                            {st.session_state.username}
                        </div>
                        <div style="color:{api_color};font-size:.72rem;">
                            ● {api_status}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Navigation options
            if st.button("📊  Dashboard", width='stretch',
                         type="primary" if st.session_state.page == "dashboard" else "secondary",
                         key="nav_dashboard"):
                st.session_state.page = "dashboard"
                st.rerun()
            if st.button("⚙️  Account Settings", width='stretch',
                         type="primary" if st.session_state.page == "settings" else "secondary",
                         key="nav_settings"):
                st.session_state.page = "settings"
                st.rerun()
            if st.button("💻  Command Help", width='stretch',
                         type="primary" if st.session_state.page == "commands" else "secondary",
                         key="nav_commands"):
                st.session_state.page = "commands"
                st.rerun()
            st.markdown(
                f'<div style="border-top:1px solid {T["border"]};margin:.6rem 0;"></div>',
                unsafe_allow_html=True,
            )
            if st.button("🚪  Logout", width='stretch', key="nav_logout"):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.session_state.page = "dashboard"
                st.session_state.is_trading = False
                st.session_state.trading_status = "Idle"
                st.rerun()

    # ── App title ─────────────────────────────────────────────────────────
    with c_title:
        st.markdown(
            f'<div style="display:flex;align-items:center;height:2.4rem;gap:.6rem;">'
            f'<span style="font-size:1.35rem;">🤖</span>'
            f'<span style="color:{T["text"]};font-size:1.05rem;font-weight:800;'
            f'letter-spacing:-.01em;">AI Trading Bot</span>'
            f'<span style="color:{T["muted"]};font-size:.7rem;margin-left:.3rem;">'
            f'UWE · UFCFXK-30-3</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Theme toggle ──────────────────────────────────────────────────────
    with c_theme:
        if st.button("🌞" if st.session_state.dark_mode else "🌙",
                     help="Toggle light / dark theme", width='stretch',
                     key="topbar_theme"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    # ── Bottom border ─────────────────────────────────────────────────────
    st.markdown(
        f'<div style="border-bottom:1px solid {T["border"]};margin:.2rem 0 1rem;"></div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE: Dashboard
# ===========================================================================

def _page_dashboard() -> None:
    runs = _list_runs()

    # ── Page title ────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="padding:.5rem 0 .8rem;">'
        f'<h1 style="color:{T["text"]};margin:0;font-size:1.75rem;font-weight:800;'
        f'letter-spacing:-.02em;">📊 Dashboard</h1>'
        f'<p style="color:{T["muted"]};margin:.2rem 0 0;font-size:.88rem;">'
        f'Paper Trading Simulation</p></div>',
        unsafe_allow_html=True,
    )

    # ── Experiment run selector ───────────────────────────────────────────
    if not runs:
        _card(
            f'<div style="text-align:center;padding:2rem;">'
            f'<div style="font-size:2.5rem;">📭</div>'
            f'<h3 style="color:{T["text"]};margin:.8rem 0 .4rem;">No runs yet</h3>'
            f'<p style="color:{T["muted"]};">Train an agent first, then run evaluation.</p></div>'
        )
        return

    run_names = [r.name for r in runs]
    with st.expander("🔬 Experiments — select runs to display", expanded=True):
        sel_col, tags_col = st.columns([3, 4])
        with sel_col:
            selected_runs: list[str] = st.multiselect(
                "Runs", options=run_names,
                default=st.session_state.get("selected_runs", [run_names[0]] if run_names else []),
                label_visibility="collapsed",
                key="selected_runs",
            )
        with tags_col:
            for rn in selected_runs:
                info = _parse_run_name(rn)
                color = ALGO_COLORS.get(info["algo"].lower(), T["muted"])
                st.markdown(
                    f'<span style="display:inline-flex;align-items:center;gap:.4rem;'
                    f'background:{color}18;border:1px solid {color}44;border-radius:20px;'
                    f'padding:2px 10px;margin:.15rem .25rem;font-size:.75rem;">'
                    f'{_badge(info["algo"], color)}'
                    f'<span style="color:{T["muted"]};">{info["symbol"]} · {info["date"]}</span></span>',
                    unsafe_allow_html=True,
                )

    if not selected_runs:
        st.info("Select one or more experiment runs above to view results.")
        return

    selected_dirs = [RUNS_DIR / name for name in selected_runs]

    st.markdown(f'<div style="border-top:1px solid {T["border"]};margin:.5rem 0 1.5rem;"></div>',
                unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────────────────
    KPI_KEYS = [
        ("cumulative_return", "Total Return", "%",  True),
        ("sharpe_ratio",      "Sharpe Ratio", "",   True),
        ("max_drawdown",      "Max Drawdown", "%",  False),
        ("win_rate",          "Win Rate",     "%",  True),
        ("number_of_trades",  "# Trades",     "",   None),
    ]
    for rd in selected_dirs:
        for tag, m in _load_metrics(rd).items():
            color = ALGO_COLORS.get(tag.lower(), T["accent"])
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.6rem;">'
                f'{_badge(tag.upper(), color)}'
                f'<span style="color:{T["muted"]};font-size:.8rem;">{rd.name[:40]}</span></div>',
                unsafe_allow_html=True,
            )
            cards = '<div style="display:flex;gap:.75rem;flex-wrap:wrap;margin-bottom:1.2rem;">'
            for key, label, suffix, hib in KPI_KEYS:
                raw = m.get(key, 0.0)
                display = (f"{raw * 100:+.2f}" if key in ("cumulative_return", "win_rate")
                           else f"{raw * 100:.2f}" if key == "max_drawdown"
                           else str(int(raw)) if key == "number_of_trades"
                           else f"{raw:.3f}")
                vc = (T["green"] if raw >= 0 else T["red"]) if hib is True else \
                     (T["red"] if raw > 0.05 else T["green"]) if hib is False else T["text"]
                cards += _kpi_card(label, display, suffix, vc)
            cards += "</div>"
            st.markdown(cards, unsafe_allow_html=True)

    st.markdown(f'<div style="border-top:1px solid {T["border"]};margin:0 0 1.5rem;"></div>',
                unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────
    t_eq, t_dd, t_tr, t_pos, t_met = st.tabs([
        "📈  Equity", "📉  Drawdown", "🔄  Trades", "📊  Positions", "🏆  Metrics",
    ])

    with t_eq:
        _section_hdr("Equity Curves", "Portfolio value over the test period")
        fig = go.Figure()
        for i, rd in enumerate(selected_dirs):
            for tag, series in _load_equity(rd).items():
                c = ALGO_COLORS.get(tag.lower(), CHART_PALETTE[i % len(CHART_PALETTE)])
                fig.add_trace(go.Scatter(
                    x=series.index, y=series.values, mode="lines",
                    name=f"{tag.upper()} — {rd.name[:20]}",
                    line=dict(color=c, width=2.5),
                    fill="tozeroy", fillcolor=_hex_rgba(c, .08),
                ))
        fig.add_hline(y=10_000, line_dash="dot", line_color=T["muted"], opacity=.5,
                      annotation_text="Starting Capital", annotation_position="bottom right",
                      annotation_font_color=T["muted"])
        _chart_base(fig, height=440)
        st.plotly_chart(fig, width='stretch')

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
                    name=f"{tag.upper()} — {rd.name[:20]}",
                    line=dict(color=c, width=2), fill="tozeroy", fillcolor=_hex_rgba(c, .12),
                ))
                dd_rows.append({"Strategy": tag.upper(), "Run": rd.name[:35],
                                 "Max DD": f"{dd.max():.2f}%", "Avg DD": f"{dd.mean():.2f}%"})
        _chart_base(fig, height=400)
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, width='stretch')
        if dd_rows:
            st.dataframe(pd.DataFrame(dd_rows), width='stretch', hide_index=True)

    with t_tr:
        _section_hdr("Trade Signals", "Buy / Sell markers overlaid on price")
        sym = st.text_input("Symbol", value="BTC-USD", label_visibility="collapsed")
        price_path = PROJECT_ROOT / "data" / "processed" / sym / "test_raw.csv"
        if price_path.exists():
            price_df = pd.read_csv(price_path, parse_dates=True, index_col=0)
            if "Close" in price_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=price_df.index, y=price_df["Close"],
                                         mode="lines", name="Price",
                                         line=dict(color=T["muted"], width=1.5)))
                trade_rows = []
                for i, rd in enumerate(selected_dirs):
                    for tag, tdf in _load_trades(rd).items():
                        c = ALGO_COLORS.get(tag.lower(), CHART_PALETTE[i % len(CHART_PALETTE)])
                        for act, sym_mk, clr, lbl in [
                            ("buy",  "triangle-up",   T["green"], "Buy"),
                            ("sell", "triangle-down", T["red"],  "Sell"),
                        ]:
                            sub = tdf[tdf["action"] == act]
                            if not sub.empty and "step" in sub.columns:
                                steps = sub["step"].values.astype(int)
                                valid = steps[steps < len(price_df)]
                                if len(valid):
                                    fig.add_trace(go.Scatter(
                                        x=price_df.index[valid],
                                        y=price_df["Close"].values[valid],
                                        mode="markers", name=f"{tag.upper()} {lbl}",
                                        marker=dict(color=clr, size=9, symbol=sym_mk,
                                                    line=dict(color=T["surface"], width=1)),
                                    ))
                        trade_rows.append({
                            "Strategy": tag.upper(), "Run": rd.name[:35],
                            "Buys": len(tdf[tdf["action"] == "buy"]),
                            "Sells": len(tdf[tdf["action"] == "sell"]),
                            "Avg Cost": f"{tdf['cost'].mean():.4f}" if "cost" in tdf.columns else "—",
                        })
                _chart_base(fig, height=450)
                fig.update_yaxes(tickprefix="$")
                st.plotly_chart(fig, width='stretch')
                if trade_rows:
                    st.dataframe(pd.DataFrame(trade_rows), width='stretch', hide_index=True)
        else:
            st.warning(f"Price data not found. Run preprocessing for {sym} first.")

    with t_pos:
        _section_hdr("Portfolio Value", "Per-run equity progression")
        for i, rd in enumerate(selected_dirs):
            for tag, series in _load_equity(rd).items():
                c = ALGO_COLORS.get(tag.lower(), CHART_PALETTE[i % len(CHART_PALETTE)])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=series.index, y=series.values, mode="lines", name=tag.upper(),
                    line=dict(color=c, width=2.5), fill="tozeroy", fillcolor=_hex_rgba(c, .1),
                ))
                _chart_base(fig, title=f"{tag.upper()} · {rd.name[:35]}", height=300)
                fig.update_yaxes(tickprefix="$")
                st.plotly_chart(fig, width='stretch')

    with t_met:
        _section_hdr("Performance Metrics", "Full metrics comparison")
        rows = []
        for rd in selected_dirs:
            for tag, m in _load_metrics(rd).items():
                rows.append({
                    "Strategy": tag.upper(), "Run": rd.name[:35],
                    "Return":      f"{m.get('cumulative_return', 0)*100:+.2f}%",
                    "Ann. Return": f"{m.get('annualised_return', 0)*100:+.2f}%",
                    "Volatility":  f"{m.get('annualised_volatility', 0)*100:.2f}%",
                    "Sharpe":      f"{m.get('sharpe_ratio', 0):.3f}",
                    "Max DD":      f"{m.get('max_drawdown', 0)*100:.2f}%",
                    "Win Rate":    f"{m.get('win_rate', 0)*100:.1f}%",
                    "Trades":      int(m.get("number_of_trades", 0)),
                    "Exposure":    f"{m.get('exposure_time', 0)*100:.1f}%",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            if len(rows) > 1:
                _divider()
                _section_hdr("Visual Comparison")
                raw_rows = []
                for rd in selected_dirs:
                    for tag, m in _load_metrics(rd).items():
                        raw_rows.append({
                            "label": tag.upper(),
                            "Return %": round(m.get("cumulative_return", 0)*100, 2),
                            "Sharpe":   round(m.get("sharpe_ratio", 0), 3),
                            "Win Rate %": round(m.get("win_rate", 0)*100, 1),
                        })
                fig_bar = go.Figure()
                for i, row in enumerate(raw_rows):
                    c = CHART_PALETTE[i % len(CHART_PALETTE)]
                    fig_bar.add_trace(go.Bar(
                        name=row["label"],
                        x=["Return %", "Sharpe", "Win Rate %"],
                        y=[row["Return %"], row["Sharpe"], row["Win Rate %"]],
                        marker_color=c, marker_line_color=T["border"], marker_line_width=1,
                    ))
                fig_bar.update_layout(barmode="group")
                _chart_base(fig_bar, title="Key Metrics Comparison", height=360)
                st.plotly_chart(fig_bar, width='stretch')
        else:
            st.info("No metrics found. Run evaluation first.")


# ===========================================================================
# PAGE: Settings
# ===========================================================================

def _page_settings() -> None:
    st.markdown(
        f'<div style="padding:.5rem 0 .8rem;">'
        f'<h1 style="color:{T["text"]};margin:0;font-size:1.75rem;font-weight:800;">⚙️ Account Settings</h1>'
        f'<p style="color:{T["muted"]};margin:.2rem 0 0;font-size:.88rem;">'
        f'Profile & API configuration for {st.session_state.username}</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div style="border-top:1px solid {T["border"]};margin:0 0 1.5rem;"></div>',
                unsafe_allow_html=True)

    user = auth.get_user(st.session_state.username)
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT: Profile & Security ──────────────────────────────────────────
    with col_left:
        _section_hdr("👤 Profile", "Your account details")
        st.markdown(
            f"""
            <div style="background:{T['surface']};border:1px solid {T['border']};
                        border-radius:14px;padding:1.2rem;margin-bottom:1.5rem;">
                <div style="display:flex;align-items:center;gap:1rem;">
                    <div style="background:{T['accent']}22;border:2px solid {T['accent']}44;
                                border-radius:50%;width:52px;height:52px;display:flex;
                                align-items:center;justify-content:center;font-size:1.5rem;">
                        👤
                    </div>
                    <div>
                        <div style="color:{T['text']};font-weight:700;font-size:1rem;">
                            {st.session_state.username}
                        </div>
                        <div style="color:{T['muted']};font-size:.78rem;">Local account</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _section_hdr("🔒 Change Password")
        old_pw  = st.text_input("Current password",  type="password", key="s_old_pw")
        new_pw  = st.text_input("New password",       type="password", key="s_new_pw",
                                 help="Minimum 6 characters")
        conf_pw = st.text_input("Confirm new password", type="password", key="s_conf_pw")

        if st.button("Update Password", width='stretch', key="btn_update_password"):
            if new_pw != conf_pw:
                st.error("New passwords do not match.")
            else:
                ok, msg = auth.update_password(st.session_state.username, old_pw, new_pw)
                st.success(msg) if ok else st.error(msg)

    # ── RIGHT: Bybit Demo API ─────────────────────────────────────────────
    with col_right:
        _section_hdr("⚡ Bybit API", "Connect your Bybit demo account")

        # Status indicator
        has_key = bool(user.get("bybit_api_key"))
        use_demo = user.get("bybit_use_demo", True)
        status_color = T["green"] if has_key else T["muted"]
        status_text  = "Configured" if has_key else "Not configured"
        mode_label   = "Demo Account" if use_demo else "Testnet"

        st.markdown(
            f"""
            <div style="background:{T['surface']};border:1px solid {T['border']};
                        border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:1.2rem;
                        display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="color:{T['muted']};font-size:.72rem;text-transform:uppercase;
                                letter-spacing:.08em;">API Status</div>
                    <div style="color:{status_color};font-weight:700;font-size:1rem;margin-top:.2rem;">
                        ● {status_text}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="color:{T['muted']};font-size:.72rem;text-transform:uppercase;
                                letter-spacing:.08em;">Mode</div>
                    <div style="color:{T['accent']};font-weight:600;font-size:.9rem;margin-top:.2rem;">
                        {mode_label}
                    </div>
                </div>
            </div>

            <div style="background:{T['surface2']};border:1px solid {T['border']};
                        border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.2rem;
                        font-size:.82rem;color:{T['muted']};line-height:1.7;">
                <strong style="color:{T['text']};">How to get your Bybit Demo API keys:</strong><br>
                1. Go to <a href="https://www.bybit.com/en/trade/spot/demo" target="_blank"
                   style="color:{T['accent']};">bybit.com → Demo Trading</a><br>
                2. Click your avatar → <em>API</em> → <em>Create New Key</em><br>
                3. Set permissions: <em>Read + Trade</em><br>
                4. Copy the API Key and Secret below
            </div>
            """,
            unsafe_allow_html=True,
        )

        api_mode = st.radio(
            "Account type",
            options=["Demo Account (recommended)", "Testnet"],
            index=0 if use_demo else 1,
            horizontal=True,
            key="s_api_mode",
        )
        use_demo_new = api_mode.startswith("Demo")

        api_key_val    = user.get("bybit_api_key", "")
        api_secret_val = user.get("bybit_api_secret", "")

        new_key    = st.text_input("API Key",    value=api_key_val,    type="password",
                                    placeholder="Paste your Bybit API key", key="s_api_key")
        new_secret = st.text_input("API Secret", value=api_secret_val, type="password",
                                    placeholder="Paste your Bybit API secret", key="s_api_secret")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔌  Test Connection", width='stretch', key="btn_test_conn"):
                with st.spinner("Connecting to Bybit…"):
                    ok, msg = auth.test_bybit_connection(new_key, new_secret, use_demo_new)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        with btn_col2:
            if st.button("💾  Save API Keys", width='stretch', type="primary", key="btn_save_keys"):
                auth.update_api_keys(
                    st.session_state.username, new_key, new_secret, use_demo_new
                )
                st.success("API keys saved.")
                st.rerun()

        # Safety note
        st.markdown(
            f'<p style="color:{T["muted"]};font-size:.75rem;margin-top:.8rem;line-height:1.6;">'
            f'🔒 Keys are stored locally on this machine only and never transmitted '
            f'to any third party. This dashboard uses <strong>paper / demo trading</strong> '
            f'— no real funds are at risk.</p>',
            unsafe_allow_html=True,
        )


# ===========================================================================
# PAGE: Command Help
# ===========================================================================

def _page_commands() -> None:
    _section_hdr("💻 Command Help", "CLI commands to train, evaluate and run the pipeline")

    st.markdown(
        f'<div style="background:{T["surface"]};border:1px solid {T["border"]};'
        f'border-radius:14px;padding:1.4rem 1.6rem;margin-bottom:1.2rem;">',
        unsafe_allow_html=True,
    )

    _section_hdr("🏋️ Training")
    st.code(
        "# Train DQN agent on BTC-USD\n"
        "python -m src.cli.train --algo dqn --symbol BTC-USD\n\n"
        "# Train PPO agent\n"
        "python -m src.cli.train --algo ppo --symbol BTC-USD\n\n"
        "# Train Q-Learning agent\n"
        "python -m src.cli.train --algo qlearning --symbol BTC-USD",
        language="bash",
    )

    _section_hdr("📊 Evaluation")
    st.code(
        "# Evaluate a trained run\n"
        "python -m src.cli.evaluate --run_dir runs/<run_name> --symbol BTC-USD\n\n"
        "# Run all baselines\n"
        "python -m src.cli.baselines --symbol BTC-USD",
        language="bash",
    )

    _section_hdr("⚡ Full Pipeline")
    st.code(
        "# Download data → preprocess → train → evaluate in one command\n"
        "python -m src.cli.pipeline --symbol BTC-USD",
        language="bash",
    )

    _section_hdr("📥 Data")
    st.code(
        "# Download raw OHLCV data\n"
        "python -m src.cli.download_data --symbol BTC-USD\n\n"
        "# Preprocess downloaded data\n"
        "python -m src.cli.preprocess --symbol BTC-USD",
        language="bash",
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# Trading status bar (shown at top of every logged-in page)
# ===========================================================================

def _run_council_decision(
    api_key: str, api_secret: str, use_demo: bool, trade_mode: str
) -> None:
    """Load models → build obs → vote → execute. Updates session state."""
    run_names = st.session_state.council_runs

    with st.spinner("Loading models & fetching market data…"):
        try:
            # Reuse cached models if available, else load fresh
            if not st.session_state.loaded_models:
                st.session_state.loaded_models = [
                    trader_mod.load_model(RUNS_DIR / name) for name in run_names
                ]
            models = st.session_state.loaded_models
            obs, market_info, last_price = trader_mod.build_live_obs("BTC-USD")
            st.session_state.market_info = market_info
        except Exception as exc:
            st.session_state.trading_status = "Failed"
            st.session_state.trading_error  = f"Model/data error: {exc}"
            st.rerun()

    result = trader_mod.council_vote(models, obs)
    st.session_state.council_result = result
    st.session_state.last_decision_time = pd.Timestamp.now().isoformat()
    action_label = result["label"]

    if action_label == "Hold":
        st.session_state.is_trading     = True
        st.session_state.trading_status = "Live"
        st.session_state.trading_error  = ""
        # Log the hold decision
        st.session_state.trade_log.append({
            "time":      pd.Timestamp.now().isoformat(),
            "price":     last_price,
            "action":    0,
            "label":     "Hold",
            "order_msg": "",
        })
    else:
        side = "Buy" if action_label == "Buy" else "Sell"
        with st.spinner(f"Placing {trade_mode} {side} order on Bybit…"):
            ord_ok, ord_msg, _ = auth.place_order(
                api_key, api_secret, use_demo, trade_mode, side=side
            )
        if ord_ok:
            st.session_state.is_trading     = True
            st.session_state.trading_status = "Live"
            st.session_state.trading_error  = ""
            st.session_state.council_result["order_msg"] = ord_msg
            st.session_state.trade_log.append({
                "time":      pd.Timestamp.now().isoformat(),
                "price":     last_price,
                "action":    1 if side == "Buy" else 2,
                "label":     side,
                "order_msg": ord_msg,
            })
        else:
            st.session_state.trading_status = "Failed"
            st.session_state.trading_error  = f"Order failed: {ord_msg}"

    st.rerun()


def _trading_status_bar() -> None:
    """Model selector + status chip + council result + Start/Next/Stop buttons."""

    STATUS_CFG: dict[str, dict] = {
        "Idle":    {"color": T["muted"],  "dot": "○", "label": "Idle"},
        "Live":    {"color": T["green"],  "dot": "●", "label": "Live Trading"},
        "Failed":  {"color": T["red"],    "dot": "●", "label": "Connection Failed"},
        "Stopped": {"color": T["yellow"], "dot": "●", "label": "Stopped"},
    }
    ACTION_COLORS = {"Hold": T["muted"], "Buy": T["green"], "Sell": T["red"]}

    is_trading = st.session_state.is_trading
    status     = st.session_state.trading_status
    cfg        = STATUS_CFG.get(status, STATUS_CFG["Idle"])
    mode       = st.session_state.get("trading_mode", "Spot")

    # ── Model / Council configuration (only when not trading) ──────────────
    if not is_trading:
        all_runs   = trader_mod.list_runs()
        run_labels = {r["name"]: r["label"] for r in all_runs}

        with st.expander("⚙️  Trading Configuration", expanded=False):
            cfg_left, cfg_right = st.columns([3, 2])
            with cfg_left:
                st.markdown(
                    f'<p style="color:{T["muted"]};font-size:.8rem;margin-bottom:.3rem;">'
                    f'Select one model for single-agent trading, or multiple for Council mode '
                    f'(majority vote decides each trade).</p>',
                    unsafe_allow_html=True,
                )
                selected = st.multiselect(
                    "Models",
                    options=[r["name"] for r in all_runs],
                    default=st.session_state.council_runs,
                    format_func=lambda n: run_labels.get(n, n),
                    label_visibility="collapsed",
                    key="council_runs_select",
                )
                st.session_state.council_runs = selected
            with cfg_right:
                st.radio("Trading Mode", options=["Spot", "Futures"],
                         horizontal=True, key="trading_mode")
                n_sel = len(selected)
                if n_sel == 0:
                    st.caption("⚠️ Select at least one model.")
                elif n_sel == 1:
                    st.caption(f"Single-agent: {run_labels.get(selected[0], selected[0])}")
                else:
                    st.caption(f"Council mode: {n_sel} models · majority vote")

    # ── Status chip row ────────────────────────────────────────────────────
    mode_badge = (
        f'<span style="color:{T["muted"]};font-size:.78rem;margin-left:.5rem;">· {mode}</span>'
        if is_trading else ""
    )

    if not is_trading:
        bar_col, btn_col = st.columns([6, 2])
    else:
        bar_col, next_col, stop_col = st.columns([4, 2, 2])

    with bar_col:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:.75rem;
                        background:{T['surface']};border:1px solid {T['border']};
                        border-radius:10px;padding:.55rem 1rem;margin-bottom:.4rem;">
                <span style="font-size:.85rem;font-weight:600;color:{T['text']};">
                    Trading Status
                </span>
                <div style="background:{cfg['color']}1a;border:1px solid {cfg['color']}55;
                            border-radius:20px;padding:3px 12px;display:flex;
                            align-items:center;gap:6px;">
                    <span style="color:{cfg['color']};font-size:.8rem;">{cfg['dot']}</span>
                    <span style="color:{cfg['color']};font-size:.8rem;font-weight:700;">
                        {cfg['label']}
                    </span>
                </div>
                {mode_badge}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not is_trading:
        with btn_col:
            can_start = bool(st.session_state.council_runs)
            if st.button("▶  Start Trading", width='stretch', type="primary",
                         key="btn_start_trading", disabled=not can_start):
                st.session_state.trading_error  = ""
                st.session_state.council_result = {}
                st.session_state.trade_log      = []
                st.session_state.loaded_models  = None

                user       = auth.get_user(st.session_state.username)
                api_key    = user.get("bybit_api_key", "")
                api_secret = user.get("bybit_api_secret", "")
                use_demo   = user.get("bybit_use_demo", True)
                trade_mode = st.session_state.trading_mode

                # Step 1 — Bybit connectivity
                with st.spinner("Connecting to Bybit…"):
                    conn_ok, conn_msg = auth.test_bybit_connection(api_key, api_secret, use_demo)
                if not conn_ok:
                    st.session_state.trading_status = "Failed"
                    st.session_state.trading_error  = conn_msg
                    st.rerun()

                # Step 2 — record entry price & time, then run first decision
                try:
                    _, _, entry_price = trader_mod.build_live_obs("BTC-USD")
                    st.session_state.entry_price = entry_price
                    st.session_state.entry_time  = pd.Timestamp.now().isoformat()
                except Exception as exc:
                    st.session_state.trading_status = "Failed"
                    st.session_state.trading_error  = f"Data fetch error: {exc}"
                    st.rerun()

                _run_council_decision(api_key, api_secret, use_demo, trade_mode)
    else:
        user       = auth.get_user(st.session_state.username)
        api_key    = user.get("bybit_api_key", "")
        api_secret = user.get("bybit_api_secret", "")
        use_demo   = user.get("bybit_use_demo", True)
        trade_mode = st.session_state.trading_mode

        with next_col:
            if st.button("🔄  Next Decision", width='stretch', key="btn_next_decision"):
                st.session_state.trading_error = ""
                _run_council_decision(api_key, api_secret, use_demo, trade_mode)

        with stop_col:
            if st.button("■  Stop Trading", width='stretch', key="btn_stop_trading"):
                st.session_state.is_trading     = False
                st.session_state.trading_status = "Stopped"
                st.session_state.trading_error  = ""
                st.session_state.loaded_models  = None
                st.rerun()

    # ── Persistent error ───────────────────────────────────────────────────
    if st.session_state.trading_error:
        st.error(st.session_state.trading_error)

    # ── Council vote result panel ──────────────────────────────────────────
    result = st.session_state.get("council_result", {})
    if result:
        winning     = result["label"]
        win_color   = ACTION_COLORS.get(winning, T["text"])
        model_votes = result.get("model_votes", [])
        counts      = result.get("counts", {})
        market_info = st.session_state.get("market_info", "")
        order_msg   = result.get("order_msg", "")
        order_html  = (
            f'<span style="color:{T["green"]};font-size:.75rem;">✓ {order_msg}</span>'
            if order_msg else ""
        )

        votes_html = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:.3rem;'
            f'background:{ACTION_COLORS.get(v["label"], T["muted"])}18;'
            f'border:1px solid {ACTION_COLORS.get(v["label"], T["muted"])}44;'
            f'border-radius:16px;padding:2px 8px;margin:.15rem;font-size:.75rem;">'
            f'<b style="color:{T["text"]};">{v["algo"]}</b>'
            f'<span style="color:{ACTION_COLORS.get(v["label"], T["muted"])};font-weight:700;">'
            f'{v["emoji"]}</span></span>'
            for v in model_votes
        )
        vote_summary = " · ".join(
            f'{trader_mod.ACTION_NAMES[k]}: {v}' for k, v in sorted(counts.items())
        )
        st.markdown(
            f"""
            <div style="background:{T['surface']};border:1px solid {win_color}44;
                        border-left:4px solid {win_color};border-radius:10px;
                        padding:.8rem 1.1rem;margin:.3rem 0 .5rem;">
                <div style="display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;">
                    <span style="color:{T['muted']};font-size:.78rem;font-weight:600;
                                 text-transform:uppercase;letter-spacing:.06em;">Council Decision</span>
                    <span style="color:{win_color};font-size:1rem;font-weight:800;">
                        {result['emoji']}</span>
                    <span style="color:{T['muted']};font-size:.75rem;">
                        ({vote_summary}) · {market_info}</span>
                    {order_html}
                </div>
                <div style="margin-top:.5rem;">{votes_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# Live trading chart  (shown when a trading session is active or just stopped)
# ===========================================================================

INTERVAL_OPTIONS = {"30s": 30, "1 min": 60, "5 min": 300, "15 min": 900}
LIVE_BACKEND_URL = "http://localhost:8001"


def _backend_get(endpoint: str, timeout: float = 2.0):
    """GET a JSON endpoint from the live backend. Returns None on failure."""
    try:
        url = f"{LIVE_BACKEND_URL}{endpoint}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


@st.fragment(run_every=timedelta(seconds=2))
def _live_trading_chart() -> None:
    """Auto-refreshing price chart with live backend data + automatic bot decisions."""
    entry_price = st.session_state.get("entry_price", 0.0)
    entry_time  = st.session_state.get("entry_time", "")
    trade_log   = st.session_state.get("trade_log", [])

    if not entry_price or not entry_time:
        return

    # ── Try live backend first ────────────────────────────────────────────
    live_prices = _backend_get("/prices")
    live_history = _backend_get("/history/BTCUSDT?limit=300") if live_prices else None
    live_decisions = _backend_get("/decisions?limit=50") if live_prices else None
    backend_ok = live_prices is not None and "BTCUSDT" in (live_prices or {})

    # ── Auto-decision logic (runs every fragment refresh) ─────────────────
    if st.session_state.is_trading and st.session_state.council_runs:
        interval = st.session_state.get("auto_trade_interval", 60)
        last_str = st.session_state.get("last_decision_time", "")
        now = pd.Timestamp.now()
        run_decision = False
        if not last_str:
            run_decision = True
        else:
            elapsed = (now - pd.Timestamp(last_str)).total_seconds()
            if elapsed >= interval:
                run_decision = True

        if run_decision:
            try:
                user       = auth.get_user(st.session_state.username)
                api_key    = user.get("bybit_api_key", "")
                api_secret = user.get("bybit_api_secret", "")
                use_demo   = user.get("bybit_use_demo", True)
                trade_mode = st.session_state.trading_mode

                if not st.session_state.loaded_models:
                    st.session_state.loaded_models = [
                        trader_mod.load_model(RUNS_DIR / name)
                        for name in st.session_state.council_runs
                    ]
                models = st.session_state.loaded_models
                obs, market_info, last_price = trader_mod.build_live_obs("BTC-USD")
                st.session_state.market_info = market_info

                # Use live price if backend is available
                if backend_ok:
                    last_price = live_prices["BTCUSDT"]["last"]

                result = trader_mod.council_vote(models, obs)
                st.session_state.council_result = result
                st.session_state.last_decision_time = now.isoformat()
                action_label = result["label"]

                if action_label == "Hold":
                    st.session_state.trade_log.append({
                        "time": now.isoformat(), "price": last_price,
                        "action": 0, "label": "Hold", "order_msg": "",
                    })
                else:
                    side = "Buy" if action_label == "Buy" else "Sell"
                    ord_ok, ord_msg, _ = auth.place_order(
                        api_key, api_secret, use_demo, trade_mode, side=side,
                    )
                    if ord_ok:
                        st.session_state.council_result["order_msg"] = ord_msg
                        st.session_state.trade_log.append({
                            "time": now.isoformat(), "price": last_price,
                            "action": 1 if side == "Buy" else 2,
                            "label": side, "order_msg": ord_msg,
                        })
                    else:
                        st.session_state.trading_error = f"Order failed: {ord_msg}"
                trade_log = st.session_state.get("trade_log", [])
            except Exception as exc:
                st.session_state.trading_error = f"Auto-trade error: {exc}"

    _divider()

    # ── Header + connection badge ─────────────────────────────────────────
    conn_badge = (
        f'<span style="background:{T["green"]}22;color:{T["green"]};border:1px solid {T["green"]}44;'
        f'padding:2px 8px;border-radius:12px;font-size:.7rem;font-weight:700;margin-left:.6rem;">'
        f'● LIVE</span>'
        if backend_ok else
        f'<span style="background:{T["yellow"]}22;color:{T["yellow"]};border:1px solid {T["yellow"]}44;'
        f'padding:2px 8px;border-radius:12px;font-size:.7rem;font-weight:700;margin-left:.6rem;">'
        f'● Yahoo Finance</span>'
    )
    st.markdown(
        f'<h3 style="color:{T["text"]};font-size:1.1rem;font-weight:700;margin:.4rem 0 .8rem;">'
        f'📡 Live Trade Monitor{conn_badge}</h3>',
        unsafe_allow_html=True,
    )

    # ── Config row: timeframe + auto-interval + sliders ───────────────────
    tf_col, iv_col, sl_col, tp_col = st.columns([2, 2, 2, 2])
    with tf_col:
        timeframe = st.radio(
            "Timeframe",
            options=["1H", "1W", "1M", "1Y"],
            index=0 if backend_ok else 2,
            horizontal=True,
            key="chart_timeframe",
        )
    with iv_col:
        interval_label = st.radio(
            "Bot interval",
            options=list(INTERVAL_OPTIONS.keys()),
            index=1,
            horizontal=True,
            key="auto_interval_radio",
        )
        st.session_state.auto_trade_interval = INTERVAL_OPTIONS[interval_label]
    with sl_col:
        profit_pct = st.slider("Take-profit %", 0.5, 20.0,
                               st.session_state.profit_target_pct, 0.5,
                               key="slider_tp")
        st.session_state.profit_target_pct = profit_pct
    with tp_col:
        stop_pct = st.slider("Stop-loss %", 0.5, 15.0, 1.5, 0.5,
                             key="slider_sl")

    # ── Auto-decision status line ─────────────────────────────────────────
    last_str = st.session_state.get("last_decision_time", "")
    if last_str and st.session_state.is_trading:
        ago = (pd.Timestamp.now() - pd.Timestamp(last_str)).total_seconds()
        interval_sec = st.session_state.get("auto_trade_interval", 60)
        next_in = max(0, interval_sec - ago)
        feed_label = "Bybit WS" if backend_ok else "Yahoo Finance"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:1rem;'
            f'background:{T["surface"]};border:1px solid {T["border"]};'
            f'border-radius:8px;padding:.4rem .8rem;margin-bottom:.6rem;font-size:.78rem;">'
            f'<span style="color:{T["green"]};">● Auto-trading active</span>'
            f'<span style="color:{T["muted"]};">Last: {int(ago)}s ago</span>'
            f'<span style="color:{T["muted"]};">Next: ~{int(next_in)}s</span>'
            f'<span style="color:{T["muted"]};">Feed: {feed_label}</span>'
            f'<span style="color:{T["muted"]};">Refresh: 2s</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Fetch price data ───────────────────────────────────────────────────
    # Source A: live tick history from backend
    # Source B: Yahoo Finance (fallback)
    df = None
    current_price = 0.0

    if backend_ok and live_history and len(live_history) > 2:
        # Build a DataFrame from live tick history
        hist_df = pd.DataFrame(live_history)
        hist_df["datetime"] = pd.to_datetime(hist_df["ts"], unit="s")
        hist_df = hist_df.set_index("datetime").sort_index()
        df = hist_df
        current_price = live_prices["BTCUSDT"]["last"]
    else:
        # Fallback: Yahoo Finance
        try:
            yf_df = trader_mod.fetch_price_df("BTC-USD", timeframe=timeframe)
            df = yf_df.rename(columns={"Close": "price"})
            current_price = float(yf_df["Close"].iloc[-1])
        except Exception as exc:
            st.warning(f"Could not fetch price data: {exc}")
            return

    if df is None or df.empty:
        st.warning("No price data available.")
        return

    entry_dt        = pd.Timestamp(entry_time)
    profit_target   = entry_price * (1 + profit_pct / 100)
    stop_loss_price = entry_price * (1 - stop_pct  / 100)
    pnl             = current_price - entry_price
    pnl_pct         = (pnl / entry_price) * 100

    # ── KPI strip ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Entry Price",   f"${entry_price:,.2f}")
    k2.metric("Current Price", f"${current_price:,.2f}")
    k3.metric("Unrealised P&L",
              f"${pnl:+,.2f}",
              delta=f"{pnl_pct:+.2f}%")
    k4.metric("Target Price",  f"${profit_target:,.2f}",
              delta=f"+{profit_pct:.1f}%")

    # ── Multi-symbol live ticker strip (when backend available) ───────────
    if backend_ok and live_prices:
        sym_cards = ""
        for sym, pdata in live_prices.items():
            sym_cards += (
                f'<div style="background:{T["surface"]};border:1px solid {T["border"]};'
                f'border-radius:10px;padding:.5rem .8rem;min-width:140px;text-align:center;">'
                f'<div style="color:{T["muted"]};font-size:.7rem;font-weight:600;">{sym}</div>'
                f'<div style="color:{T["text"]};font-size:1.1rem;font-weight:800;">'
                f'${pdata["last"]:,.2f}</div>'
                f'<div style="color:{T["muted"]};font-size:.65rem;">'
                f'Bid: ${pdata["bid"]:,.2f}  Ask: ${pdata["ask"]:,.2f}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div style="display:flex;gap:.6rem;margin-bottom:.8rem;flex-wrap:wrap;">'
            f'{sym_cards}</div>',
            unsafe_allow_html=True,
        )

    # ── Plotly chart ───────────────────────────────────────────────────────
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["price"], mode="lines",
        name="BTC/USD", line=dict(color=T["accent"], width=2),
        fill="tozeroy", fillcolor=_hex_rgba(T["accent"], 0.06),
    ))

    # Entry point marker
    fig.add_trace(go.Scatter(
        x=[entry_dt], y=[entry_price],
        mode="markers+text", name="Entry",
        marker=dict(color=T["green"], size=14, symbol="triangle-up",
                    line=dict(color=T["surface"], width=2)),
        text=["Entry"], textposition="top center",
        textfont=dict(color=T["green"], size=11),
    ))

    # Entry price horizontal reference
    fig.add_hline(
        y=entry_price, line_color=T["accent"], line_dash="dot", line_width=1.5,
        annotation_text=f"Entry  ${entry_price:,.0f}",
        annotation_font_color=T["accent"], annotation_position="right",
    )

    # Profit target line
    fig.add_hline(
        y=profit_target, line_color=T["green"], line_dash="dash", line_width=1.5,
        annotation_text=f"TP +{profit_pct}%  ${profit_target:,.0f}",
        annotation_font_color=T["green"], annotation_position="right",
    )

    # Stop-loss line
    fig.add_hline(
        y=stop_loss_price, line_color=T["red"], line_dash="dash", line_width=1.5,
        annotation_text=f"SL -{stop_pct}%  ${stop_loss_price:,.0f}",
        annotation_font_color=T["red"], annotation_position="right",
    )

    # Current price annotation
    fig.add_hline(
        y=current_price, line_color=T["muted"], line_dash="dot",
        line_width=1, opacity=0.6,
        annotation_text=f"Now  ${current_price:,.0f}",
        annotation_font_color=T["muted"], annotation_position="right",
    )

    # Trade log markers (from session state)
    for trade in trade_log:
        t_dt  = pd.Timestamp(trade["time"])
        t_lbl = trade["label"]
        if t_lbl == "Hold":
            continue
        t_color  = T["green"] if t_lbl == "Buy" else T["red"]
        t_symbol = "triangle-up" if t_lbl == "Buy" else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[t_dt], y=[trade["price"]],
            mode="markers+text", name=t_lbl,
            marker=dict(color=t_color, size=12, symbol=t_symbol,
                        line=dict(color=T["surface"], width=1.5)),
            text=[t_lbl], textposition="bottom center" if t_lbl == "Sell" else "top center",
            textfont=dict(color=t_color, size=10),
            showlegend=False,
        ))

    # Backend decision markers (from live server)
    if live_decisions:
        for dec in live_decisions:
            if dec.get("action") == "HOLD":
                continue
            d_ts = pd.Timestamp(dec["ts"], unit="s")
            d_lbl = dec["action"]
            d_color = T["green"] if d_lbl == "BUY" else T["red"]
            d_sym = "triangle-up" if d_lbl == "BUY" else "triangle-down"
            fig.add_trace(go.Scatter(
                x=[d_ts], y=[dec["price"]],
                mode="markers", name=f"AI {d_lbl}",
                marker=dict(color=d_color, size=10, symbol=d_sym,
                            line=dict(color=T["surface"], width=1)),
                showlegend=False,
            ))

    _chart_base(fig, title="BTC/USD — Live Trade Session", height=460)
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(fig, width='stretch')

    # ── Trade log table ────────────────────────────────────────────────────
    if trade_log:
        rows = [
            {
                "Time":   pd.Timestamp(t["time"]).strftime("%Y-%m-%d %H:%M:%S"),
                "Action": t["label"],
                "Price":  f"${t['price']:,.2f}",
                "Order":  t.get("order_msg", "—"),
            }
            for t in trade_log
        ]
        _section_hdr("Trade Log")
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # ── Backend decisions table ────────────────────────────────────────────
    if live_decisions and len(live_decisions) > 0:
        dec_rows = [
            {
                "Time":   pd.Timestamp(d["ts"], unit="s").strftime("%H:%M:%S"),
                "Symbol": d.get("symbol", ""),
                "Action": d["action"],
                "Price":  f"${d['price']:,.2f}",
                "Reason": d.get("reason", ""),
            }
            for d in reversed(live_decisions[-20:])
        ]
        _section_hdr("AI Decision Stream", "Decisions from the live strategy engine")
        st.dataframe(pd.DataFrame(dec_rows), width='stretch', hide_index=True)


# ===========================================================================
# Main router
# ===========================================================================

if not st.session_state.logged_in:
    _page_auth()
    st.stop()

_topbar_logged_in()
_trading_status_bar()

# Show live chart whenever a trading session has started (even if now stopped)
if st.session_state.get("entry_price", 0.0) > 0:
    _live_trading_chart()

if st.session_state.page == "settings":
    _page_settings()
elif st.session_state.page == "commands":
    _page_commands()
else:
    _page_dashboard()
