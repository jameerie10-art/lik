"""
fx_app.py
Run with:  streamlit run fx_app.py

A validation bench for reward-to-risk strategies on FX. It does not forecast price.
It measures whether a fixed rule set held a positive expectancy on data it was never
fitted to, and tells you how much evidence you actually have.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fx_vision import (
    CHART_READ_PROMPT, DEFAULT_MODEL, FALLBACK_MODELS, VisionError, integrity_check,
    plan_from_invalidation, read_chart, resolve_api_key,
)
from fx_engine import (
    AccountConfig, CostConfig, RiskConfig, STRATEGIES, backtest, bar_interval_hours,
    breakeven_win_rate, current_setup, losing_streak_probability, monte_carlo,
    expectancy_per_dollar, growth_projection, normalise_ohlc, parameter_sweep, pip_size,
    rr_sweep, synthetic_ohlc, trades_to_multiple, walk_forward, win_rate_standard_error,
)

st.set_page_config(page_title="Reward-to-risk bench", page_icon="◧",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

THEMES = {
    "dark": dict(
        base="#0E1420", panel="#151D2C", raised="#1C2638", line="#27324A",
        text="#E4EAF6", muted="#7F8DA8", faint="#5A6782",
        accent="#4C7CF3", win="#35B37E", loss="#E5484D", warn="#E0A32E",
        plot="plotly_dark", grid="rgba(255,255,255,0.06)",
    ),
    "light": dict(
        base="#F1F4F9", panel="#FFFFFF", raised="#F7F9FC", line="#D9DFEA",
        text="#141C2B", muted="#5D6980", faint="#8A94A8",
        accent="#2B5CE0", win="#1E8E5E", loss="#C92A30", warn="#9A6B08",
        plot="plotly_white", grid="rgba(0,0,0,0.07)",
    ),
}


def inject_css(t: dict) -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

html, body, .stApp {{
  background:{t['base']}; color:{t['text']};
  font-family:'Inter',system-ui,sans-serif; font-feature-settings:'tnum' 1,'cv05' 1;
}}
#MainMenu, footer, .stDeployButton {{ display:none; }}
.block-container {{ padding-top:1.4rem; padding-bottom:4rem; max-width:1340px; }}

section[data-testid="stSidebar"] {{ background:{t['panel']}; border-right:1px solid {t['line']}; }}
section[data-testid="stSidebar"] .stMarkdown p {{ color:{t['muted']}; font-size:.82rem; }}

h1, h2, h3, h4 {{ color:{t['text']}; letter-spacing:-.014em; font-weight:600; }}
h1 {{ font-size:1.5rem; margin-bottom:.1rem; }}

.masthead {{ display:flex; align-items:baseline; gap:.7rem; border-bottom:1px solid {t['line']};
  padding-bottom:.85rem; margin-bottom:1.3rem; flex-wrap:wrap; }}
.masthead .name {{ font-size:1.28rem; font-weight:700; color:{t['text']}; letter-spacing:-.02em; }}
.masthead .sub {{ font-size:.84rem; color:{t['muted']}; }}

.verdict {{ border:1px solid {t['line']}; border-left:5px solid var(--vc); background:{t['panel']};
  border-radius:6px; padding:1.05rem 1.3rem; margin-bottom:1.3rem; }}
.verdict .head {{ font-size:1.06rem; font-weight:650; color:var(--vc); margin-bottom:.3rem; }}
.verdict .body {{ font-size:.9rem; color:{t['muted']}; line-height:1.62; max-width:78ch; }}
.verdict ul {{ margin:.55rem 0 0 1.05rem; padding:0; }}
.verdict li {{ font-size:.86rem; color:{t['muted']}; margin-bottom:.2rem; line-height:1.5; }}

.panel {{ background:{t['panel']}; border:1px solid {t['line']}; border-radius:6px; padding:1rem 1.15rem; }}
.panel-title {{ font-size:.8rem; font-weight:600; color:{t['muted']}; margin-bottom:.65rem; }}

.row {{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem;
  padding:.42rem 0; border-bottom:1px solid {t['line']}; }}
.row:last-child {{ border-bottom:none; }}
.row .k {{ font-size:.83rem; color:{t['muted']}; }}
.row .v {{ font-family:'Roboto Mono',monospace; font-size:.87rem; font-weight:500; color:{t['text']}; }}
.v.good {{ color:{t['win']}; }} .v.bad {{ color:{t['loss']}; }} .v.warn {{ color:{t['warn']}; }}

.note {{ font-size:.83rem; color:{t['faint']}; line-height:1.65; max-width:80ch; }}
.callout {{ border:1px solid {t['line']}; border-radius:6px; background:{t['raised']};
  padding:.85rem 1.05rem; font-size:.86rem; color:{t['muted']}; line-height:1.68; max-width:88ch; }}

.stTabs [data-baseweb="tab-list"] {{ gap:.15rem; border-bottom:1px solid {t['line']}; }}
.stTabs [data-baseweb="tab"] {{ background:transparent; color:{t['muted']}; font-size:.85rem;
  font-weight:500; padding:.55rem .95rem; }}
.stTabs [aria-selected="true"] {{ color:{t['text']}; border-bottom:2px solid {t['accent']}; }}

.stButton > button {{ background:{t['accent']}; color:#fff; border:none; border-radius:5px;
  font-weight:600; font-size:.86rem; padding:.5rem 1.1rem; width:100%; }}
.stButton > button:hover {{ filter:brightness(1.08); color:#fff; }}
.stButton > button:focus-visible {{ outline:2px solid {t['text']}; outline-offset:2px; }}

div[data-testid="stMetricValue"] {{ font-family:'Roboto Mono',monospace; }}
hr {{ border-color:{t['line']}; }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important; transition:none !important; }} }}
</style>
""", unsafe_allow_html=True)


def fig_theme(fig: go.Figure, t: dict, height: int = 320) -> go.Figure:
    fig.update_layout(
        template=t["plot"], height=height, margin=dict(l=48, r=18, t=28, b=38),
        paper_bgcolor=t["panel"], plot_bgcolor=t["panel"],
        font=dict(family="Inter, sans-serif", size=12, color=t["muted"]),
        legend=dict(orientation="h", y=1.13, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=t["grid"], zeroline=False, linecolor=t["line"])
    fig.update_yaxes(gridcolor=t["grid"], zeroline=False, linecolor=t["line"])
    return fig


def row(k: str, v: str, cls: str = "") -> str:
    return f'<div class="row"><span class="k">{k}</span><span class="v {cls}">{v}</span></div>'


def panel(title: str, rows_html: str) -> str:
    return f'<div class="panel"><div class="panel-title">{title}</div>{rows_html}</div>'


def fmt(x, spec="{:.2f}", dash="n/a"):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return dash
        return spec.format(x)
    except Exception:
        return dash


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

with st.sidebar:
    st.markdown("### Bench settings")
    dark = st.toggle("Dark theme", value=st.session_state.theme == "dark")
    st.session_state.theme = "dark" if dark else "light"
    mode = st.radio("Mode", ["Chart read", "Strategy bench"],
                    captions=["Upload a chart image and have Gemini describe it",
                              "Backtest and validate a rule set on price history"])
T = THEMES[st.session_state.theme]
inject_css(T)

with st.sidebar:
    st.divider()
    source, uploaded, strat_name, spec, params = "Synthetic demo", None, "", None, {}

    if mode == "Strategy bench":
        st.markdown("#### Price data")
        source = st.radio("Source", ["Upload CSV", "Synthetic demo"], label_visibility="collapsed")

    uploaded = None
    if mode == "Strategy bench" and source == "Upload CSV":
        uploaded = st.file_uploader("OHLC history", type=["csv", "txt"])
        st.caption("Export from MT5 (Tools, then History Centre) or download free history "
                   "from HistData or Dukascopy. Needs a date column plus open, high, low, close. "
                   "Aim for three years or more.")
    elif mode == "Strategy bench":
        st.caption("Synthetic prices for exercising the bench. Results carry no evidential "
                   "weight about any real market.")

    symbol = st.text_input("Instrument", value="EURUSD").strip().upper() or "EURUSD"
    st.caption(f"Pip size resolved as {pip_size(symbol)}")

    if mode == "Strategy bench":
        st.divider()
        st.markdown("#### Rule set")
        strat_name = st.selectbox("Strategy", list(STRATEGIES.keys()))
        spec = STRATEGIES[strat_name]
        st.caption(spec["note"])

        params = dict(spec["defaults"])
        with st.expander("Entry parameters"):
            for key, default in spec["defaults"].items():
                if isinstance(default, bool):
                    params[key] = st.checkbox(key.replace("_", " "), value=default)
                elif isinstance(default, int):
                    params[key] = st.number_input(key.replace("_", " "), value=int(default), step=1)
                else:
                    params[key] = st.number_input(key.replace("_", " "), value=float(default), step=0.5)

    st.divider()
    st.markdown("#### Risk and exits")
    rr = st.slider("Reward to risk target (1 : R)", 1.0, 12.0, 4.0, 0.5)
    atr_mult = st.slider("Stop distance in ATRs", 0.5, 5.0, 1.5, 0.1)
    risk_pct = st.slider("Risk per trade, percent of equity", 0.1, 3.0, 0.5, 0.1)
    compound = st.checkbox("Size off current equity", value=True)

    with st.expander("Trade management", expanded=False):
        be_at = st.number_input("Move stop to breakeven at +R (0 disables)", value=0.0, step=0.5)
        be_off = st.number_input("Breakeven offset in R", value=0.1, step=0.05)
        tr_at = st.number_input("Start ATR trail at +R (0 disables)", value=0.0, step=0.5)
        tr_mult = st.number_input("Trail distance in ATRs", value=2.5, step=0.5)
        time_stop = st.number_input("Time stop in bars (0 disables)", value=0, step=10)
    st.caption("Breakeven stops and trails raise the win rate and cut the average winner. "
               "On wide targets they usually lower expectancy. Test both.")

    st.divider()
    st.markdown("#### Costs")
    spread = st.number_input("Spread, pips", value=1.2, step=0.1)
    commission = st.number_input("Commission, pips round turn", value=0.5, step=0.1)
    slip_in = st.number_input("Entry slippage, pips", value=0.2, step=0.1)
    slip_stop = st.number_input("Stop slippage, pips", value=0.8, step=0.1)
    swap = st.number_input("Swap, pips per day held", value=0.0, step=0.1)
    st.caption("Take these from your own account history, not from the broker's advertised "
               "spread. Understated costs are the most common reason a backtest that looked "
               "profitable loses money live.")

    st.divider()
    st.markdown("#### Account")
    equity0 = st.number_input("Starting equity", value=10_000.0, step=500.0)
    allow_long = st.checkbox("Allow longs", value=True)
    allow_short = st.checkbox("Allow shorts", value=True)
    pip_value = st.number_input("Pip value per standard lot, account currency", value=10.0, step=0.5)

risk_cfg = RiskConfig(risk_pct=risk_pct, rr_target=rr, atr_stop_mult=atr_mult,
                      atr_period=int(params.get("atr_period", 14)), compound=compound,
                      breakeven_at_r=be_at, breakeven_offset_r=be_off,
                      trail_start_r=tr_at, trail_atr_mult=tr_mult,
                      time_stop_bars=int(time_stop))
cost_cfg = CostConfig(spread_pips=spread, commission_pips=commission,
                      entry_slippage_pips=slip_in, stop_slippage_pips=slip_stop,
                      swap_pips_per_day=swap)
acct_cfg = AccountConfig(starting_equity=equity0, allow_longs=allow_long, allow_shorts=allow_short)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(raw_bytes: bytes) -> pd.DataFrame:
    for sep in [None, ",", ";", "\t"]:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, engine="python")
            if df.shape[1] >= 4:
                return normalise_ohlc(df)
        except Exception:
            continue
    raise ValueError("Could not parse this file as delimited OHLC data.")


@st.cache_data(show_spinner=False)
def load_synth() -> pd.DataFrame:
    return synthetic_ohlc(n=26_000)


def cfg_key(*objs) -> str:
    """Cheap, stable cache key. Hashing a large DataFrame on every rerun is what makes
    these apps feel broken on a small container, so the frame itself is never hashed."""
    from dataclasses import asdict, is_dataclass
    parts = []
    for o in objs:
        parts.append(json.dumps(asdict(o) if is_dataclass(o) else o, sort_keys=True, default=str))
    return "|".join(parts)


@st.cache_data(show_spinner=False, max_entries=24)
def cached_signals(_df, _fn, data_id: str, strat: str, params_key: str):
    return _fn(_df, **json.loads(params_key))


@st.cache_data(show_spinner=False, max_entries=24)
def cached_backtest(_df, _signals, data_id: str, symbol: str, key: str,
                    _risk, _costs, _acct):
    return backtest(_df, _signals, symbol, _risk, _costs, _acct)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_rr_sweep(_df, _signals, data_id: str, symbol: str, key: str,
                    _risk, _costs, _acct, rr_values):
    return rr_sweep(_df, _signals, symbol, _risk, _costs, _acct, rr_values=rr_values)


if mode == "Chart read":
    st.markdown(
        f'<div class="masthead"><span class="name">Chart read</span>'
        f'<span class="sub">{symbol} · target 1:{rr:g} · {risk_pct:g}% risk per trade'
        f'</span></div>', unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="masthead"><span class="name">Reward-to-risk bench</span>'
        f'<span class="sub">{symbol} · {strat_name} · target 1:{rr:g} · '
        f'{risk_pct:g}% risk per trade</span></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chart read
# ---------------------------------------------------------------------------

def render_chart_read() -> None:
    st.markdown('<div class="callout">Gemini reads the image and reports what is visibly on '
                'it: structure, levels, the last few candles, the case each side would make, '
                'and the price at which the read stops being true. It is not asked for a '
                'direction or a confidence score. The order below is then calculated from '
                'the invalidation level it found, not written by the model.</div>',
                unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        img = st.file_uploader("Chart screenshot", type=["png", "jpg", "jpeg", "webp"])
        question = st.text_area(
            "Anything specific you want checked (optional)", height=90,
            placeholder="e.g. is the 1.1000 level holding, or how many times has that "
                        "trendline actually been touched")
    with c2:
        model_name = st.selectbox("Model", FALLBACK_MODELS,
                                  index=FALLBACK_MODELS.index(DEFAULT_MODEL)
                                  if DEFAULT_MODEL in FALLBACK_MODELS else 0)
        key_found = resolve_api_key() is not None
        manual_key = "" if key_found else st.text_input(
            "Gemini API key", type="password",
            help="Better: put GEMINI_API_KEY in Streamlit Secrets so it is never typed "
                 "or committed.")
        st.markdown(panel("Key", row("Source",
                    "Secrets or environment" if key_found else "typed in, this session only",
                    "good" if key_found else "warn")), unsafe_allow_html=True)
        go = st.button("Read the chart")

    if img is not None:
        st.image(img, caption="Image sent to the model", width="stretch")

    if go:
        if img is None:
            st.error("Upload a chart screenshot first.")
            return
        try:
            with st.spinner("Reading the image"):
                st.session_state.chart_read = read_chart(
                    img.getvalue(), img.type, resolve_api_key(manual_key or None),
                    model=model_name, extra_question=question)
        except VisionError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"The request failed. {e}")
            return

    read = st.session_state.get("chart_read")
    if not read:
        return

    st.write("")
    if not read.get("is_price_chart", False):
        st.error("The model says this is not a price chart. "
                 + str(read.get("not_a_chart_reason") or ""))
        return

    problems = integrity_check(read)
    if problems:
        st.markdown(f'<div class="callout" style="border-color:{T["warn"]}">'
                    '<b>The model broke its own rules on this run.</b><ul>'
                    + "".join(f"<li>{p}</li>" for p in problems) +
                    '</ul>Treat this read as unreliable and run it again rather than '
                    'working around it.</div>', unsafe_allow_html=True)
        st.write("")

    if not read.get("readable", True):
        st.warning("The model says the image is not legible enough for a full read. "
                   "A tighter crop with the price axis visible usually fixes it.")

    struct = read.get("structure") or {}
    inval = read.get("invalidation") or {}
    L, R = st.columns([2, 3], gap="large")
    with L:
        st.markdown(panel("What it identified", "".join([
            row("Instrument", str(read.get("instrument") or "not labelled")),
            row("Timeframe", str(read.get("timeframe") or "not labelled")),
            row("Visible period", str(read.get("visible_period") or "unclear")),
            row("Regime", str(struct.get("regime", "unclear"))),
            row("Axis precision", str(read.get("axis_precision", ""))[:40]),
            row("Model", str(read.get("_model_used", ""))),
        ])), unsafe_allow_html=True)
        levels = read.get("levels") or []
        if levels:
            st.write("")
            st.markdown(panel("Levels, by touches counted", "".join(
                row(f"{lv.get('approx_price')} ({lv.get('kind','')})",
                    f"{lv.get('touches', 0)} touches")
                for lv in levels)), unsafe_allow_html=True)
    with R:
        st.markdown(f'<div class="panel"><div class="panel-title">Structure</div>'
                    f'<div class="note">{struct.get("evidence", "")}</div></div>',
                    unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<div class="panel"><div class="panel-title">Most recent bars</div>'
                    f'<div class="note">{read.get("recent_bars", "")}</div></div>',
                    unsafe_allow_html=True)

    st.write("")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(f'<div class="panel"><div class="panel-title">The continuation case</div>'
                    f'<div class="note">{read.get("continuation_case", "")}</div></div>',
                    unsafe_allow_html=True)
    with b:
        st.markdown(f'<div class="panel"><div class="panel-title">The reversal case</div>'
                    f'<div class="note">{read.get("reversal_case", "")}</div></div>',
                    unsafe_allow_html=True)
    st.caption("Both are on screen on purpose. Reading only the one you already agreed with "
               "is the most reliable way to lose money with a tool like this.")

    st.write("")
    inval_price = inval.get("price")
    st.markdown(f'<div class="verdict" style="--vc:{T["accent"]}">'
                f'<div class="head">Invalidation at '
                f'{inval_price if inval_price is not None else "not determinable"}</div>'
                f'<div class="body">{inval.get("what_it_breaks", "")}</div></div>',
                unsafe_allow_html=True)

    if inval_price:
        st.markdown("#### Order, calculated from that level")
        o1, o2 = st.columns([1, 1])
        entry_px = o1.number_input("Your intended entry price", value=float(inval_price),
                                   format="%.5f", step=0.0001)
        buffer = o2.number_input("Stop buffer beyond the level, pips", 0.0, 100.0, 3.0, 0.5)
        try:
            plan = plan_from_invalidation(entry_px, float(inval_price), rr, equity0,
                                          risk_pct, symbol, pip_value, buffer)
            p1, p2 = st.columns(2, gap="large")
            with p1:
                st.markdown(panel("Order", "".join([
                    row("Direction", plan.direction,
                        "good" if plan.direction == "long" else "bad"),
                    row("Entry", f"{plan.entry:.5f}"),
                    row("Stop", f"{plan.stop:.5f}"),
                    row("Target", f"{plan.target:.5f}"),
                ])), unsafe_allow_html=True)
            with p2:
                st.markdown(panel("Size", "".join([
                    row("Stop distance", f"{plan.stop_pips:.1f} pips"),
                    row("Target distance", f"{plan.target_pips:.1f} pips"),
                    row("Cash at risk", f"{plan.risk_amount:,.2f}"),
                    row("Lots", f"{plan.lots:.2f}"),
                ])), unsafe_allow_html=True)
            st.caption(plan.basis + " Set the entry to the price you would actually get "
                       "rather than the level itself, since the distance is what sets "
                       "the size.")
        except VisionError as e:
            st.warning(str(e))

    gaps = read.get("cannot_determine") or []
    if gaps:
        st.write("")
        st.markdown(panel("What this image does not tell you",
                          "".join(row(g, "") for g in gaps)), unsafe_allow_html=True)
    notes = read.get("notes_for_the_trader")
    if notes:
        st.markdown(f'<div class="callout">{notes}</div>', unsafe_allow_html=True)

    st.write("")
    with st.expander("The prompt being used"):
        st.code(CHART_READ_PROMPT, language="markdown")
    with st.expander("Raw JSON returned"):
        st.json(read)


if mode == "Chart read":
    render_chart_read()
    st.stop()

df = None
if source == "Upload CSV" and uploaded is not None:
    try:
        df = load_csv(uploaded.getvalue())
    except Exception as e:
        st.error(f"Could not read that file. {e}")
elif source == "Synthetic demo":
    df = load_synth()

if df is None or len(df) < 500:
    st.markdown(
        '<div class="callout">Load at least 500 bars to run the bench. Upload a CSV with a '
        'date column and open, high, low and close, or switch the source to the synthetic '
        'demo in the sidebar to see how the bench behaves.</div>', unsafe_allow_html=True)
    st.stop()

if source == "Synthetic demo":
    st.markdown('<div class="callout"><b>Synthetic prices.</b> These bars were generated, not '
                'observed. The generator is tuned so a trend follower lands near breakeven on '
                'it, which is roughly what an efficient market looks like. Nothing here is '
                'evidence about EURUSD or any other pair.</div>', unsafe_allow_html=True)

MAX_BARS = 400_000
if len(df) > MAX_BARS:
    st.warning(f"That file has {len(df):,} bars. Using the most recent {MAX_BARS:,} so the "
               f"app stays inside a small container's memory. If you need the older history "
               f"too, resample to a higher timeframe rather than loading every minute.")
    df = df.iloc[-MAX_BARS:]

interval_h = bar_interval_hours(df.index)
years = (df.index[-1] - df.index[0]).days / 365.25
data_id = f"{source}|{symbol}|{len(df)}|{df.index[0]}|{df.index[-1]}"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

params_key = json.dumps(params, sort_keys=True, default=str)
try:
    signals = cached_signals(df, spec["fn"], data_id, strat_name, params_key)
except Exception as e:
    st.error(f"The rule set failed to build signals. {e}")
    st.stop()

run_key = cfg_key(risk_cfg, cost_cfg, acct_cfg, params_key)
res = cached_backtest(df, signals, data_id, symbol, run_key, risk_cfg, cost_cfg, acct_cfg)
S = res.stats
n = int(S["n_trades"])

if n == 0:
    st.warning("This configuration produced no trades. Loosen the filters, shorten the "
               "channel, or load more history.")
    st.stop()

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

wf_summary = st.session_state.get("wf_summary", {})
checks = []
checks.append(("Sample size", n >= 100,
               f"{n} trades over {years:.1f} years. 100 is the floor for any read, "
               f"300 is where a wide target starts to be measurable."))
checks.append(("Expectancy after costs", S["expectancy_r"] > 0,
               f"{S['expectancy_r']:+.3f} R per trade, with costs of about "
               f"{fmt(S['cost_r'], '{:.2f}')} R per round turn already deducted."))
checks.append(("Edge is bigger than noise", np.isfinite(S["t_stat"]) and S["t_stat"] > 2.0,
               f"t statistic {fmt(S['t_stat'])}. Below 2 the result is indistinguishable "
               f"from luck at this sample size."))
checks.append(("Win rate clears breakeven", S["win_rate"] > S["breakeven_win_rate"],
               f"{S['win_rate']:.1f}% observed against {S['breakeven_win_rate']:.1f}% needed "
               f"at 1:{rr:g} after costs."))
if wf_summary:
    eff = wf_summary.get("efficiency", float("nan"))
    checks.append(("Survived walk forward", np.isfinite(eff) and eff > 0.4,
                   f"Out-of-sample expectancy is {fmt(eff * 100, '{:.0f}')}% of in-sample "
                   f"across {int(wf_summary.get('folds', 0))} folds."))
else:
    checks.append(("Survived walk forward", None,
                   "Not run yet. Open the walk forward tab. Until this passes, everything "
                   "above is an in-sample number and in-sample numbers are free."))

hard_fails = [c for c in checks if c[1] is False]
pending = [c for c in checks if c[1] is None]

if hard_fails:
    vcolor, headline = T["loss"], "Do not trade this configuration"
    body = ("The rule set fails at least one gate that has to hold before money is at risk. "
            "Fix the failures below rather than adjusting the target until the equity curve "
            "looks better, because that is curve fitting and it does not transfer.")
elif pending:
    vcolor, headline = T["warn"], "In-sample only, not yet validated"
    body = ("Every gate that has been tested passes, but the numbers below were produced on "
            "the same data the parameters were chosen against. Run walk forward and Monte "
            "Carlo before treating any of this as evidence.")
else:
    vcolor, headline = T["win"], "Passed the validation gates. Forward test next."
    body = ("The configuration cleared every gate. That earns it a demo account, not a live "
            "one. Trade it on demo for the number of trades listed in the protocol tab and "
            "compare the live win rate against the backtest before funding it.")

items = "".join(
    f'<li><b style="color:{T["win"] if c[1] else (T["warn"] if c[1] is None else T["loss"])}">'
    f'{"pass" if c[1] else ("pending" if c[1] is None else "fail")}</b> &nbsp;{c[0]}. {c[2]}</li>'
    for c in checks)
st.markdown(f'<div class="verdict" style="--vc:{vcolor}"><div class="head">{headline}</div>'
            f'<div class="body">{body}</div><ul>{items}</ul></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Equity and drawdown
# ---------------------------------------------------------------------------

left, right = st.columns([3, 2], gap="large")

with left:
    eq = res.equity
    peak = eq.cummax()
    dd = (eq - peak) / peak * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity", mode="lines",
                             line=dict(color=T["accent"], width=2)))
    fig.add_hline(y=acct_cfg.starting_equity, line=dict(color=T["faint"], width=1, dash="dot"))
    st.plotly_chart(fig_theme(fig, T, 300), width="stretch")

    figd = go.Figure()
    figd.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", mode="lines",
                              fill="tozeroy", line=dict(color=T["loss"], width=1)))
    figd.update_yaxes(title="Drawdown %")
    st.plotly_chart(fig_theme(figd, T, 200), width="stretch")

with right:
    st.markdown(panel("Outcome", "".join([
        row("Trades", f"{n}"),
        row("Period", f"{years:.1f} yr"),
        row("Bar interval", f"{fmt(interval_h, '{:.1f}')} h"),
        row("Expectancy", f"{S['expectancy_r']:+.3f} R",
            "good" if S["expectancy_r"] > 0 else "bad"),
        row("Win rate", f"{S['win_rate']:.1f}%"),
        row("Breakeven win rate", f"{S['breakeven_win_rate']:.1f}%"),
        row("Profit factor", fmt(S["profit_factor"])),
        row("Return", f"{S['total_return_pct']:+.1f}%",
            "good" if S["total_return_pct"] > 0 else "bad"),
        row("Max drawdown", f"{S['max_dd_pct']:.1f}%", "bad" if S["max_dd_pct"] > 25 else ""),
        row("Max drawdown in R", f"{S['max_dd_r']:.1f} R"),
        row("Longest losing run", f"{int(S['longest_loss_streak'])}"),
        row("Sharpe", fmt(S["sharpe"])),
        row("Calmar", fmt(S["calmar"])),
    ])), unsafe_allow_html=True)

st.write("")
c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown(panel("How long trades last", "".join([
        row("Mean hold", f"{fmt(S['avg_hours_held'], '{:.1f}')} h"),
        row("Mean hold in bars", fmt(S["avg_bars_held"], "{:.0f}")),
        row("Median winner", f"{fmt(S['median_hours_winners'], '{:.1f}')} h"),
        row("Median loser", f"{fmt(S['median_hours_losers'], '{:.1f}')} h"),
        row("Hit target", f"{S['target_hit_rate']:.0f}%"),
        row("Hit stop", f"{S['stop_hit_rate']:.0f}%"),
    ])), unsafe_allow_html=True)
with c2:
    se = win_rate_standard_error(S["win_rate"] / 100, n) * 100
    lo, hi = S["win_rate"] - 1.96 * se, S["win_rate"] + 1.96 * se
    clears = lo > S["breakeven_win_rate"]
    st.markdown(panel("How much you actually know", "".join([
        row("Win rate 95% range", f"{lo:.1f} to {hi:.1f}%"),
        row("Range clears breakeven", "yes" if clears else "no", "good" if clears else "bad"),
        row("t statistic", fmt(S["t_stat"]), "good" if S["t_stat"] > 2 else "warn"),
        row("Trades for significance", fmt(S["trades_needed"], "{:.0f}")),
        row("Trades per year", fmt(S["trades_per_year"], "{:.0f}")),
        row("Cost per round turn", f"{fmt(S['cost_r'], '{:.2f}')} R"),
    ])), unsafe_allow_html=True)
with c3:
    k = S["kelly_pct"]
    suggested = max(0.0, min(k / 4.0, 2.0))
    st.markdown(panel("Position sizing", "".join([
        row("Avg winner", f"{S['avg_win_r']:+.2f} R"),
        row("Avg loser", f"{-S['avg_loss_r']:+.2f} R"),
        row("Full Kelly", f"{fmt(k, '{:.1f}')}%", "warn"),
        row("Quarter Kelly", f"{fmt(suggested, '{:.2f}')}%", "good"),
        row("Currently risking", f"{risk_pct:g}%",
            "bad" if risk_pct > suggested and suggested > 0 else "good"),
        row("Risk in cash", f"{acct_cfg.starting_equity * risk_pct / 100:,.0f}"),
    ])), unsafe_allow_html=True)
st.caption("Full Kelly is the growth-optimal fraction if the measured edge is exactly right "
           "and stable. It never is. Quarter Kelly or less is the working range, and the "
           "estimate is only as good as the sample it came from.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tabs = st.tabs(["Target sweep", "Account growth", "Walk forward", "Monte Carlo",
                "Parameter sensitivity", "Trades", "Live reading", "Protocol"])

# --- Target sweep ----------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="callout">This holds your entry logic fixed and moves only the '
                'target. It is the direct answer to whether 1:4 or 1:8 is reachable with '
                '<i>this</i> entry. Win rate must fall as the target widens. What matters is '
                'whether expectancy stays above zero while it falls.</div>',
                unsafe_allow_html=True)
    st.write("")
    sweep = cached_rr_sweep(df, signals, data_id, symbol, run_key, risk_cfg, cost_cfg,
                            acct_cfg, (1, 1.5, 2, 3, 4, 5, 6, 8, 10))

    f = go.Figure()
    f.add_trace(go.Bar(x=sweep["rr"], y=sweep["expectancy_r"], name="Expectancy, R",
                       marker_color=[T["win"] if v > 0 else T["loss"] for v in sweep["expectancy_r"]]))
    f.update_xaxes(title="Reward to risk target")
    f.update_yaxes(title="Expectancy in R")
    st.plotly_chart(fig_theme(f, T, 290), width="stretch")

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=sweep["rr"], y=sweep["win_rate"], name="Observed win rate",
                            line=dict(color=T["accent"], width=2)))
    f2.add_trace(go.Scatter(x=sweep["rr"], y=sweep["breakeven_win_rate"], name="Breakeven after costs",
                            line=dict(color=T["loss"], width=2, dash="dash")))
    f2.update_xaxes(title="Reward to risk target")
    f2.update_yaxes(title="Win rate %")
    st.plotly_chart(fig_theme(f2, T, 290), width="stretch")
    st.caption("Where the blue line crosses below the red one, that target is unreachable "
               "with this entry and this cost structure. Widening past the crossing point "
               "makes the account smaller, not larger.")

    show = sweep.copy()
    show.columns = ["RR", "Expectancy R", "Win rate %", "Breakeven %", "Trades",
                    "Profit factor", "Max DD %", "Mean hold h", "Return %"]
    st.dataframe(show.round(2), width="stretch", hide_index=True)

# --- Account growth --------------------------------------------------------
with tabs[1]:
    st.markdown('<div class="callout">Turning one dollar into four is a different question '
                'from risking one dollar to make four. The reward-to-risk setting decides '
                'what a single winning trade pays. Account growth is decided by expectancy '
                'per trade, how many trades you get, and what fraction of equity you risk. '
                'A rule set with a real edge can still fail to reach 4x, because the '
                'drawdown on the way there takes you out first. This tab measures which of '
                'the two happens.</div>', unsafe_allow_html=True)
    st.write("")

    g1, g2, g3 = st.columns([1, 1, 1])
    give_up = g1.number_input("Drawdown you would stop at, %", 5.0, 90.0, 40.0, 5.0, key="gu")
    horizon_tr = g2.number_input("Trade horizon", 200, 6_000, 3_000, 200, key="gh")
    g_paths = g3.number_input("Paths", 500, 8_000, 3_000, 500, key="gp")

    per_trade = expectancy_per_dollar(S["expectancy_r"], risk_pct)
    st.markdown(panel("Growth rate per trade", "".join([
        row("Expectancy", f"{S['expectancy_r']:+.3f} R"),
        row("Risk per trade", f"{risk_pct:g}% of equity"),
        row("Average growth per trade", f"{per_trade * 100:+.3f}%",
            "good" if per_trade > 0 else "bad"),
        row("Trades per year", fmt(S["trades_per_year"], "{:.0f}")),
        row("Trades to double, ignoring path risk", fmt(trades_to_multiple(S["expectancy_r"], risk_pct, 2), "{:.0f}")),
        row("Trades to 4x, ignoring path risk", fmt(trades_to_multiple(S["expectancy_r"], risk_pct, 4), "{:.0f}")),
        row("Trades to 8x, ignoring path risk", fmt(trades_to_multiple(S["expectancy_r"], risk_pct, 8), "{:.0f}")),
    ])), unsafe_allow_html=True)
    st.caption("Those three figures assume you never get stopped out of the plan by a "
               "drawdown. The table below removes that assumption, which is what makes it "
               "the number that matters.")
    st.write("")

    if S["expectancy_r"] <= 0:
        st.markdown(f'<div class="callout" style="border-color:{T["loss"]}">Expectancy is '
                    f'negative on this configuration, so the account compounds downward. '
                    f'There is no risk setting that turns a negative edge into 4x. Fix the '
                    f'edge first.</div>', unsafe_allow_html=True)
    else:
        gp = growth_projection(res.trades["r"].to_numpy(), risk_pct=risk_pct,
                               targets=(2.0, 3.0, 4.0, 6.0, 8.0),
                               give_up_dd_pct=float(give_up), max_trades=int(horizon_tr),
                               n_paths=int(g_paths), trades_per_year=S["trades_per_year"],
                               compound=compound)
        if not gp.empty:
            fg = go.Figure()
            fg.add_trace(go.Bar(x=gp["target_multiple"], y=gp["prob_reach_first_pct"],
                                name=f"Reaches the multiple", marker_color=T["win"]))
            fg.add_trace(go.Bar(x=gp["target_multiple"], y=gp["prob_give_up_first_pct"],
                                name=f"Hits {give_up:.0f}% drawdown first", marker_color=T["loss"]))
            fg.add_trace(go.Bar(x=gp["target_multiple"], y=gp["prob_neither_pct"],
                                name="Neither within the horizon", marker_color=T["faint"]))
            fg.update_layout(barmode="stack")
            fg.update_xaxes(title="Account multiple")
            fg.update_yaxes(title="Share of simulated paths, %")
            st.plotly_chart(fig_theme(fg, T, 300), width="stretch")

            disp_g = gp.copy()
            disp_g.columns = ["Multiple", "Reaches it %", "Gives up first %", "Neither %",
                              "Median trades", "90th pct trades", "Median years"]
            st.dataframe(disp_g.round(1), width="stretch", hide_index=True)

            four = gp.loc[gp["target_multiple"] == 4.0]
            if not four.empty:
                fr = four.iloc[0]
                st.markdown(panel("Reading the 4x row", "".join([
                    row("Reaches 4x before giving up", f"{fr['prob_reach_first_pct']:.0f}%",
                        "good" if fr["prob_reach_first_pct"] > 60 else "warn"),
                    row("Stops out first", f"{fr['prob_give_up_first_pct']:.0f}%",
                        "bad" if fr["prob_give_up_first_pct"] > 25 else ""),
                    row("Median trades if it gets there", fmt(fr["median_trades_to_target"], "{:.0f}")),
                    row("Median years if it gets there", fmt(fr["median_years_to_target"], "{:.1f}")),
                ])), unsafe_allow_html=True)

            st.caption("Raise the risk per trade in the sidebar and watch both columns move. "
                       "More risk shortens the median time to the target and raises the "
                       "chance of stopping out before you get there. Somewhere between those "
                       "two is the fraction you can actually live with, and for most people "
                       "it sits below 1%.")

# --- Walk forward ----------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="callout">Each fold tunes the entry parameters on a training '
                'window, then applies that single choice to the next untouched window. Only '
                'the untouched windows are scored. The headline number is efficiency: '
                'out-of-sample expectancy divided by in-sample expectancy. Above 0.5 is '
                'respectable, below 0.3 means the parameters were fitted to noise.</div>',
                unsafe_allow_html=True)
    st.write("")
    a, b, c = st.columns([1, 1, 2])
    folds_n = a.number_input("Folds", 2, 12, 5)
    train_mult = b.number_input("Train window, multiples of test", 1.0, 8.0, 3.0, 0.5)
    run_wf = c.button("Run walk forward", key="wf")

    if run_wf:
        with st.spinner("Fitting each training window and scoring the untouched windows"):
            try:
                fold_df, oos, summary = walk_forward(
                    df, spec["fn"], params, spec["grid"], symbol, risk_cfg, cost_cfg,
                    acct_cfg, n_folds=int(folds_n), train_multiple=float(train_mult))
                st.session_state.wf_summary = summary
                st.session_state.wf_folds = fold_df
                st.session_state.wf_oos = oos
            except Exception as e:
                st.error(f"Walk forward could not run. {e}")

    fold_df = st.session_state.get("wf_folds")
    summary = st.session_state.get("wf_summary", {})
    if fold_df is not None and not fold_df.empty:
        eff = summary.get("efficiency", float("nan"))
        cls = "good" if np.isfinite(eff) and eff > 0.5 else ("warn" if np.isfinite(eff) and eff > 0.3 else "bad")
        st.markdown(panel("Out-of-sample result", "".join([
            row("Folds completed", f"{int(summary['folds'])}"),
            row("In-sample expectancy", f"{summary['is_expectancy_r']:+.3f} R"),
            row("Out-of-sample expectancy", f"{summary['oos_expectancy_r']:+.3f} R",
                "good" if summary["oos_expectancy_r"] > 0 else "bad"),
            row("Efficiency", fmt(eff, "{:.2f}"), cls),
            row("Profitable folds", f"{summary['profitable_folds_pct']:.0f}%"),
            row("Out-of-sample trades", f"{int(summary['oos_trades'])}"),
            row("Out-of-sample win rate", fmt(summary.get("oos_win_rate"), "{:.1f}%")),
        ])), unsafe_allow_html=True)
        st.write("")

        fw = go.Figure()
        fw.add_trace(go.Bar(x=fold_df["fold"], y=fold_df["train_expectancy_r"],
                            name="In sample", marker_color=T["faint"]))
        fw.add_trace(go.Bar(x=fold_df["fold"], y=fold_df["test_expectancy_r"],
                            name="Out of sample", marker_color=T["accent"]))
        fw.update_xaxes(title="Fold")
        fw.update_yaxes(title="Expectancy in R")
        st.plotly_chart(fig_theme(fw, T, 280), width="stretch")

        disp = fold_df.copy()
        disp["params"] = disp["params"].astype(str)
        for col in ["train_start", "train_end", "test_start", "test_end"]:
            disp[col] = pd.to_datetime(disp[col]).dt.strftime("%Y-%m-%d")
        st.dataframe(disp.round(3), width="stretch", hide_index=True)
        st.caption("If the chosen parameters jump around between folds, the strategy has no "
                   "stable setting and the in-sample optimum is an artefact.")
    else:
        st.info("Not run yet.")

# --- Monte Carlo -----------------------------------------------------------
with tabs[3]:
    st.markdown('<div class="callout">The realised trades are reshuffled thousands of times '
                'to show what the same edge could have looked like in a different order. '
                'This creates no new information. It answers one question: how ugly can the '
                'path get before the edge shows up, and can you survive that path.</div>',
                unsafe_allow_html=True)
    st.write("")
    m1, m2, m3 = st.columns([1, 1, 2])
    paths = m1.number_input("Paths", 500, 10_000, 4_000, 500)
    ruin_dd = m2.number_input("Give-up drawdown %", 5.0, 90.0, 40.0, 5.0)
    horizon = m3.number_input("Trades per path", 20, 2000, min(max(n, 100), 1000), 20)

    mc = monte_carlo(res.trades["r"].to_numpy(), risk_pct=risk_pct, n_paths=int(paths),
                     path_len=int(horizon), ruin_dd_pct=float(ruin_dd), compound=compound)
    if mc:
        k1, k2 = st.columns(2, gap="large")
        with k1:
            st.markdown(panel("Where the account lands", "".join([
                row("5th percentile", f"{mc['final_p05']:+.1f}%",
                    "bad" if mc["final_p05"] < 0 else "good"),
                row("Median", f"{mc['final_p50']:+.1f}%"),
                row("95th percentile", f"{mc['final_p95']:+.1f}%"),
                row("Chance of finishing up", f"{mc['prob_profit']:.1f}%",
                    "good" if mc["prob_profit"] > 75 else "warn"),
            ])), unsafe_allow_html=True)
        with k2:
            st.markdown(panel("What you have to sit through", "".join([
                row("Median worst drawdown", f"{mc['dd_p50']:.1f}%"),
                row("95th percentile drawdown", f"{mc['dd_p95']:.1f}%", "warn"),
                row("Worst path seen", f"{mc['dd_worst']:.1f}%", "bad"),
                row(f"Chance of a {ruin_dd:.0f}% drawdown", f"{mc['prob_ruin']:.1f}%",
                    "bad" if mc["prob_ruin"] > 10 else "good"),
            ])), unsafe_allow_html=True)

        st.write("")
        fm = go.Figure()
        fm.add_trace(go.Histogram(x=mc["max_dd_samples"], nbinsx=60,
                                  marker_color=T["loss"], opacity=.75, name="Worst drawdown"))
        fm.add_vline(x=S["max_dd_pct"], line=dict(color=T["accent"], width=2),
                     annotation_text="backtest", annotation_position="top")
        fm.update_xaxes(title="Worst drawdown reached on a path, %")
        fm.update_yaxes(title="Paths")
        st.plotly_chart(fig_theme(fm, T, 300), width="stretch")
        st.caption("Your backtest drawdown is one sample from this distribution, and it is "
                   "usually a flattering one. Size for the 95th percentile, not the backtest.")

        wr = S["win_rate"] / 100
        streaks = [5, 8, 10, 12, 15, 20]
        probs = [losing_streak_probability(wr, int(horizon), s) * 100 for s in streaks]
        st.markdown(panel(f"Chance of at least one losing run this long, over {int(horizon)} trades",
                          "".join(row(f"{s} losses in a row", f"{p:.1f}%",
                                      "warn" if p > 50 else "")
                                  for s, p in zip(streaks, probs))), unsafe_allow_html=True)
        st.caption("At a low win rate these runs are ordinary arithmetic, not a sign the "
                   "strategy has broken. Deciding in advance which runs are normal is what "
                   "stops you abandoning a working system at the worst moment.")

# --- Sensitivity -----------------------------------------------------------
with tabs[4]:
    st.markdown('<div class="callout">A healthy rule set shows a broad plateau of workable '
                'parameter values. A single tall spike surrounded by losses is a curve fit '
                'and will not survive your broker. Read the shape, not the maximum.</div>',
                unsafe_allow_html=True)
    st.write("")
    if st.button("Run sensitivity grid", key="sens"):
        with st.spinner("Sweeping the parameter grid"):
            st.session_state.sweep_df = parameter_sweep(
                df, spec["fn"], params, spec["grid"], symbol, risk_cfg, cost_cfg, acct_cfg)

    sw = st.session_state.get("sweep_df")
    if sw is not None and not sw.empty:
        keys = [c for c in sw.columns if c not in
                ("n_trades", "expectancy_r", "win_rate", "profit_factor", "max_dd_pct")]
        if len(keys) == 2:
            pivot = sw.pivot_table(index=keys[0], columns=keys[1], values="expectancy_r")
            hm = go.Figure(go.Heatmap(z=pivot.values, x=[str(c) for c in pivot.columns],
                                      y=[str(i) for i in pivot.index],
                                      colorscale=[[0, T["loss"]], [0.5, T["panel"]], [1, T["win"]]],
                                      zmid=0, colorbar=dict(title="R")))
            hm.update_xaxes(title=keys[1].replace("_", " "))
            hm.update_yaxes(title=keys[0].replace("_", " "))
            st.plotly_chart(fig_theme(hm, T, 340), width="stretch")
        st.dataframe(sw.round(3).sort_values("expectancy_r", ascending=False),
                     width="stretch", hide_index=True)
        frac = float((sw["expectancy_r"] > 0).mean() * 100)
        st.markdown(panel("Robustness", row(
            "Grid points with positive expectancy", f"{frac:.0f}%",
            "good" if frac > 60 else ("warn" if frac > 35 else "bad"))), unsafe_allow_html=True)
        st.caption("Below roughly 35 percent, the settings that work are the exception rather "
                   "than the rule, which means you found them by searching rather than because "
                   "the underlying idea holds.")
    else:
        st.info("Not run yet.")

# --- Trades ----------------------------------------------------------------
with tabs[5]:
    t = res.trades.copy()
    fr = go.Figure()
    fr.add_trace(go.Histogram(x=t["r"], nbinsx=50, marker_color=T["accent"], opacity=.8))
    fr.add_vline(x=0, line=dict(color=T["faint"], width=1))
    fr.update_xaxes(title="Trade result in R")
    fr.update_yaxes(title="Trades")
    st.plotly_chart(fig_theme(fr, T, 260), width="stretch")

    disp = t.copy()
    disp["entry_time"] = disp["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
    disp["exit_time"] = disp["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
    for col in ["entry", "exit", "stop", "target"]:
        disp[col] = disp[col].round(5)
    disp = disp[["entry_time", "exit_time", "direction", "entry", "stop", "target", "exit",
                 "stop_pips", "bars_held", "hours_held", "mae_r", "mfe_r", "r", "pnl",
                 "equity", "reason"]].round(3)
    st.dataframe(disp, width="stretch", hide_index=True, height=420)
    st.download_button("Download trades as CSV", disp.to_csv(index=False).encode(),
                       file_name=f"{symbol}_{strat_name.replace(' ', '_')}_trades.csv",
                       mime="text/csv")
    st.caption("MAE is the worst unrealised excursion before the trade resolved, in R. If "
               "most winners show an MAE beyond minus 0.8 R, your stop is barely surviving "
               "and small changes in slippage will flip those trades into losses.")

# --- Live reading ----------------------------------------------------------
with tabs[6]:
    setup = current_setup(df, signals, symbol, risk_cfg, cost_cfg,
                          float(res.equity.iloc[-1]), pip_value_per_lot=pip_value)
    st.markdown('<div class="callout">This reports whether the rule set\'s entry conditions '
                'are met on the last closed bar, and what the resulting order would be. There '
                'is no probability attached to it, because the rule set does not produce one. '
                'Any number claiming to be a confidence score for the next candle is invented.'
                '</div>', unsafe_allow_html=True)
    st.write("")
    if setup["state"] != "setup active":
        st.markdown(panel("Last closed bar", "".join([
            row("State", setup["state"]),
            row("Bar", str(setup.get("bar_time", "n/a"))),
            row("Close", fmt(setup.get("close"), "{:.5f}")),
        ])), unsafe_allow_html=True)
        st.caption(setup["detail"])
    else:
        L, R = st.columns(2, gap="large")
        with L:
            st.markdown(panel("Order", "".join([
                row("Direction", setup["direction"],
                    "good" if setup["direction"] == "long" else "bad"),
                row("Signal bar", str(setup["bar_time"])),
                row("Reference entry", f"{setup['reference_entry']:.5f}"),
                row("Stop", f"{setup['stop']:.5f}"),
                row("Target", f"{setup['target']:.5f}"),
            ])), unsafe_allow_html=True)
        with R:
            st.markdown(panel("Size", "".join([
                row("Stop distance", f"{setup['stop_pips']:.1f} pips"),
                row("Target distance", f"{setup['target_pips']:.1f} pips"),
                row("Cash at risk", f"{setup['risk_amount']:,.2f}"),
                row("Lots", f"{setup['lots']:.2f}"),
                row("Typical hold", f"{fmt(S['avg_hours_held'], '{:.0f}')} h"),
            ])), unsafe_allow_html=True)
        st.caption(setup["detail"] + " Confirm the pip value per lot with your broker before "
                   "sending this size, particularly on crosses where the account currency is "
                   "neither of the two quoted currencies.")

# --- Protocol --------------------------------------------------------------
with tabs[7]:
    need = S["trades_needed"]
    tpy = S["trades_per_year"]
    months = (need / tpy * 12) if np.isfinite(need) and tpy > 0 else float("nan")
    st.markdown(f"""
<div class="note">

**What this bench is measuring.** A fixed set of rules, applied to every bar, with costs
deducted, stops placed by volatility and targets set as a multiple of the risk. It produces
no forecast. If a tool ever hands you a percentage confidence for the next candle, ask where
the number came from. In almost every retail product the answer is that it was generated,
not measured.

**Two different questions that both get written as 1:4.** Risking one dollar to make four
is the reward-to-risk setting, and it is the slider in the sidebar. Turning one dollar into
four is account growth, and the reward-to-risk setting barely touches it. Growth is
expectancy per trade multiplied by how much of the account you risk, compounded over how
many trades you get. At your current settings that is
{expectancy_per_dollar(S['expectancy_r'], risk_pct) * 100:+.3f}% per trade, which reaches 4x
in about {fmt(trades_to_multiple(S['expectancy_r'], risk_pct, 4), '{:.0f}')} trades if
nothing knocks you off the plan first. The account growth tab prices in the chance that
something does.

**Why a wide target is a trade-off, not an upgrade.** Moving from 1:2 to 1:8 does not multiply
your returns. It lowers the win rate roughly in proportion. At 1:4 you need
{breakeven_win_rate(4.0) * 100:.1f}% of trades to win before costs. At 1:8 you need
{breakeven_win_rate(8.0) * 100:.1f}%. Your current configuration needs
{S['breakeven_win_rate']:.1f}% after costs and is delivering {S['win_rate']:.1f}%. The
target sweep tab shows where your entry stops being able to pay for a wider target.

**Costs bite harder at wide targets.** Your round-turn cost is about
{fmt(S['cost_r'], '{:.2f}')} R. That is deducted from every trade, winner or loser, and at a
{S['win_rate']:.0f}% win rate it is paid many more times than it is earned back. Halving your
stop distance doubles this number. That is why tight stops with huge targets look
irresistible in theory and lose money in practice.

**The sample size you need.** At the measured variability of results, distinguishing this
edge from zero at 95% confidence takes roughly {fmt(need, '{:.0f}')} trades. At
{fmt(tpy, '{:.0f}')} trades per year that is about {fmt(months, '{:.0f}')} months of data.
Anything shorter and you are reading noise, however good the equity curve looks.

**Order of work. Do not skip steps.**

1. Load three to five years of real bars for the one pair and one timeframe you intend to
   trade. More pairs is not more evidence, it is more chances to find a fluke.
2. Set the costs from your own account statement, including the spread during the sessions
   you actually trade. Widen them by 30% and confirm the edge survives.
3. Run the target sweep. Choose the reward-to-risk from the plateau of that curve, not from
   the single best point.
4. Run walk forward. If efficiency is below 0.3, the idea does not work and no amount of
   parameter adjustment will fix that.
5. Run the sensitivity grid. If fewer than a third of the grid is profitable, stop.
6. Run Monte Carlo. Reduce your risk per trade until the 95th percentile drawdown is a
   number you would keep trading through. Most people discover this is below 1%.
7. Demo trade the exact rules for at least 50 trades or three months, whichever is longer.
   Log every trade by hand. Compare the demo win rate against the backtest.
8. Go live at a quarter of your intended size for the first 30 trades.

**Things this bench does not model.** Weekend gaps and news gaps that jump straight through
your stop. Variable spreads during releases and at the session rollover. Requotes and
rejected orders. Broker-side stop hunting on retail accounts. Correlation between pairs if
you run more than one. Each of these makes real results worse than what you see here, never
better.

**On regulation.** Retail forex brokers serving Nigerian clients are generally licensed
offshore rather than by the Securities and Exchange Commission in Nigeria, which affects
what recourse you have in a dispute. Check where your broker is actually licensed and what
that licence covers before funding an account.

**On the numbers on this screen.** Every one of them describes the past. A positive
expectancy measured over three years is evidence that a pattern existed, not a promise that
it persists. Markets adapt, spreads widen, and the regime that made a breakout system work
can end without warning. Risk only money whose loss would not change your life.

</div>
""", unsafe_allow_html=True)
