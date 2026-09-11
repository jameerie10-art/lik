"""
fx_engine.py
Deterministic backtesting and validation engine for high reward-to-risk FX strategies.

Design rules this file obeys, because breaking any of them is how backtests lie:

1. No lookahead. Every indicator value at bar i uses only bars <= i. Signals fire on
   the CLOSE of bar i and are filled at the OPEN of bar i+1.
2. No randomness in signal generation or in trade outcomes. Randomness appears only
   inside explicitly labelled Monte Carlo resampling of ALREADY REALISED trades.
3. Costs are charged on every trade: half-spread in, half-spread out, commission in
   pip terms, plus adverse slippage on stop exits.
4. When a bar's range contains both the stop and the target, the STOP is assumed to
   have been hit first. This is the pessimistic assumption and it is the correct
   default when you only have OHLC bars.
5. Results are reported in R multiples first. R is currency agnostic, which means the
   statistics survive changes in account size, leverage and pair.

Author-facing note: nothing in this file predicts price. It measures whether a fixed
rule set produced a positive expectancy on data it was not fitted to.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

OHLC_COLUMNS = ["open", "high", "low", "close"]


# ---------------------------------------------------------------------------
# Instrument conventions
# ---------------------------------------------------------------------------

def pip_size(symbol: str) -> float:
    """Price increment of one pip. JPY crosses quote to 3 decimals, so a pip is 0.01."""
    s = symbol.upper().replace("/", "").replace("=X", "").replace("_", "")
    if s.endswith("JPY"):
        return 0.01
    if s.startswith("XAU"):
        return 0.1
    return 0.0001


def lots_for_risk(risk_amount_acct: float, stop_pips: float, pip_value_per_lot: float) -> float:
    """
    Standard lot sizing.

    pip_value_per_lot is the account-currency value of one pip on one standard lot
    (100,000 units). For a USD account: EURUSD is 10.00, GBPUSD is 10.00, USDJPY is
    roughly 1000 / USDJPY price. Pull the live figure from your broker rather than
    hardcoding it, because this is the single most common source of size errors.
    """
    if stop_pips <= 0 or pip_value_per_lot <= 0:
        return 0.0
    return risk_amount_acct / (stop_pips * pip_value_per_lot)


# ---------------------------------------------------------------------------
# Indicators. All causal.
# ---------------------------------------------------------------------------

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder smoothed average true range."""
    return true_range(df).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def donchian(df: pd.DataFrame, length: int) -> Tuple[pd.Series, pd.Series]:
    """Prior-N-bar high and low, shifted so the current bar never sees itself."""
    hi = df["high"].rolling(length, min_periods=length).max().shift(1)
    lo = df["low"].rolling(length, min_periods=length).min().shift(1)
    return hi, lo


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CostConfig:
    """Everything the broker takes. Set these from YOUR account statement, not defaults."""
    spread_pips: float = 1.0           # typical round-trip spread on the pair and session
    commission_pips: float = 0.0       # commission expressed in pips (ECN accounts)
    entry_slippage_pips: float = 0.2   # adverse fill on market entries
    stop_slippage_pips: float = 0.8    # adverse fill when a stop is triggered
    swap_pips_per_day: float = 0.0     # negative if you are paying to hold


@dataclass
class RiskConfig:
    risk_pct: float = 0.5              # percent of equity risked per trade
    rr_target: float = 4.0             # the 1:R target. This is the headline setting.
    atr_period: int = 14
    atr_stop_mult: float = 1.5         # initial stop distance in ATRs
    compound: bool = True              # size off current equity vs fixed starting equity

    breakeven_at_r: float = 0.0        # 0 disables. Move stop to entry once price is +X R
    breakeven_offset_r: float = 0.1    # park the stop slightly in profit to cover costs
    trail_start_r: float = 0.0         # 0 disables. Begin ATR trailing once price is +X R
    trail_atr_mult: float = 2.5
    time_stop_bars: int = 0            # 0 disables. Force exit after N bars.
    max_bars_held: int = 0             # hard cap, 0 disables

    def __post_init__(self) -> None:
        if self.rr_target <= 0:
            raise ValueError("rr_target must be positive")
        if self.atr_stop_mult <= 0:
            raise ValueError("atr_stop_mult must be positive")


@dataclass
class AccountConfig:
    starting_equity: float = 10_000.0
    allow_longs: bool = True
    allow_shorts: bool = True


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    stats: Dict[str, float]
    config: Dict[str, dict]


# ---------------------------------------------------------------------------
# Strategies
#
# A strategy takes an OHLC frame plus parameters and returns a frame with three
# columns: long (bool), short (bool), stop_ref (float, the ATR used for stops).
# Every value must be computable at that bar's close.
# ---------------------------------------------------------------------------

def _session_mask(index: pd.DatetimeIndex, start_hour: int, end_hour: int) -> pd.Series:
    """UTC hour filter. start 7, end 16 approximates London plus the New York overlap."""
    if start_hour == end_hour:
        return pd.Series(True, index=index)
    h = index.hour
    if start_hour < end_hour:
        m = (h >= start_hour) & (h < end_hour)
    else:  # wraps midnight
        m = (h >= start_hour) | (h < end_hour)
    return pd.Series(m, index=index)


def strat_donchian_breakout(
    df: pd.DataFrame,
    channel: int = 55,
    trend_ema: int = 200,
    adx_len: int = 14,
    adx_min: float = 20.0,
    atr_period: int = 14,
    session_start: int = 0,
    session_end: int = 0,
) -> pd.DataFrame:
    """
    Classic long-horizon breakout. This family is the natural home for 1:4 and wider,
    because a small number of sustained trends pay for a long tail of small losses.
    Expect a win rate in the 25 to 40 percent band and losing streaks in the teens.
    """
    hi, lo = donchian(df, channel)
    e = ema(df["close"], trend_ema)
    a = adx(df, adx_len)
    sess = _session_mask(df.index, session_start, session_end)

    long = (df["close"] > hi) & (df["close"] > e) & (a >= adx_min) & sess
    short = (df["close"] < lo) & (df["close"] < e) & (a >= adx_min) & sess
    return pd.DataFrame(
        {"long": long.fillna(False), "short": short.fillna(False), "stop_ref": atr(df, atr_period)}
    )


def strat_trend_pullback(
    df: pd.DataFrame,
    fast_ema: int = 21,
    slow_ema: int = 100,
    rsi_len: int = 14,
    rsi_long_max: float = 45.0,
    rsi_short_min: float = 55.0,
    atr_period: int = 14,
    session_start: int = 7,
    session_end: int = 16,
) -> pd.DataFrame:
    """
    Buy weakness inside an uptrend, sell strength inside a downtrend. Higher win rate
    than breakout, but the wide targets get hit less often, so test 1:3 to 1:5 here
    rather than 1:8.
    """
    f = ema(df["close"], fast_ema)
    s = ema(df["close"], slow_ema)
    r = rsi(df["close"], rsi_len)
    sess = _session_mask(df.index, session_start, session_end)

    up = (f > s) & (df["close"] > s)
    dn = (f < s) & (df["close"] < s)
    long = up & (r < rsi_long_max) & (r.shift(1) >= rsi_long_max) & sess
    short = dn & (r > rsi_short_min) & (r.shift(1) <= rsi_short_min) & sess
    return pd.DataFrame(
        {"long": long.fillna(False), "short": short.fillna(False), "stop_ref": atr(df, atr_period)}
    )


def strat_volatility_squeeze(
    df: pd.DataFrame,
    squeeze_len: int = 40,
    squeeze_pct: float = 0.35,
    breakout_len: int = 10,
    trend_ema: int = 200,
    atr_period: int = 14,
    session_start: int = 7,
    session_end: int = 16,
) -> pd.DataFrame:
    """
    Enter when range compresses into the bottom percentile of its own recent history
    and then resolves. Compression before expansion is one of the few genuinely robust
    regularities in FX, but it is also heavily traded, so costs matter more here.
    """
    a = atr(df, atr_period)
    rank = a.rolling(squeeze_len, min_periods=squeeze_len).rank(pct=True)
    squeezed = (rank <= squeeze_pct).shift(1)
    hi, lo = donchian(df, breakout_len)
    e = ema(df["close"], trend_ema)
    sess = _session_mask(df.index, session_start, session_end)

    long = squeezed & (df["close"] > hi) & (df["close"] > e) & sess
    short = squeezed & (df["close"] < lo) & (df["close"] < e) & sess
    return pd.DataFrame(
        {"long": long.fillna(False), "short": short.fillna(False), "stop_ref": a}
    )


STRATEGIES: Dict[str, dict] = {
    "Donchian breakout": {
        "fn": strat_donchian_breakout,
        "defaults": dict(channel=55, trend_ema=200, adx_len=14, adx_min=20.0,
                         atr_period=14, session_start=0, session_end=0),
        "grid": {"channel": [20, 34, 55, 89], "adx_min": [0.0, 18.0, 25.0]},
        "note": "Low win rate, fat right tail. The correct home for 1:5 and wider.",
    },
    "Trend pullback": {
        "fn": strat_trend_pullback,
        "defaults": dict(fast_ema=21, slow_ema=100, rsi_len=14, rsi_long_max=45.0,
                         rsi_short_min=55.0, atr_period=14, session_start=7, session_end=16),
        "grid": {"slow_ema": [50, 100, 200], "rsi_long_max": [35.0, 40.0, 45.0]},
        "note": "Middling win rate. Test 1:3 to 1:5 before reaching for 1:8.",
    },
    "Volatility squeeze": {
        "fn": strat_volatility_squeeze,
        "defaults": dict(squeeze_len=40, squeeze_pct=0.35, breakout_len=10,
                         trend_ema=200, atr_period=14, session_start=7, session_end=16),
        "grid": {"squeeze_pct": [0.25, 0.35, 0.5], "breakout_len": [8, 10, 20]},
        "note": "Cost sensitive. Do not run this on a pair with a wide spread.",
    },
}


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

def backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    symbol: str,
    risk: RiskConfig,
    costs: CostConfig,
    account: AccountConfig,
) -> BacktestResult:
    """
    Single position at a time, next-bar-open fills, pessimistic intrabar sequencing.

    Returns realised trades, an equity curve stepped at each exit, and summary stats.
    """
    _validate_ohlc(df)
    pip = pip_size(symbol)
    half_spread = 0.5 * costs.spread_pips * pip
    entry_slip = costs.entry_slippage_pips * pip
    stop_slip = costs.stop_slippage_pips * pip
    commission_price = costs.commission_pips * pip

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    idx = df.index

    long_sig = signals["long"].to_numpy(bool)
    short_sig = signals["short"].to_numpy(bool)
    stop_ref = signals["stop_ref"].to_numpy(float)

    n = len(df)
    equity = float(account.starting_equity)
    peak_equity = equity

    pos = None            # dict describing the open trade
    pending = 0           # +1 long, -1 short, fill at next bar open
    pending_ref = np.nan

    trades: List[dict] = []

    for i in range(n):
        # ---- 1. manage an open position using this bar -------------------
        if pos is not None:
            exit_price = None
            reason = ""
            direction = pos["dir"]

            # Pessimistic ordering: stop is checked before target.
            if direction == 1:
                if l[i] <= pos["stop"]:
                    exit_price = pos["stop"] - stop_slip
                    reason = "stop"
                elif h[i] >= pos["target"]:
                    exit_price = pos["target"]
                    reason = "target"
            else:
                if h[i] >= pos["stop"]:
                    exit_price = pos["stop"] + stop_slip
                    reason = "stop"
                elif l[i] <= pos["target"]:
                    exit_price = pos["target"]
                    reason = "target"

            bars_held = i - pos["entry_i"]

            if exit_price is None and risk.time_stop_bars and bars_held >= risk.time_stop_bars:
                exit_price = c[i]
                reason = "time stop"
            if exit_price is None and risk.max_bars_held and bars_held >= risk.max_bars_held:
                exit_price = c[i]
                reason = "max hold"
            if exit_price is None and i == n - 1:
                exit_price = c[i]
                reason = "end of data"

            if exit_price is None:
                # No exit. Update protective stops for the NEXT bar only, never this one.
                mfe_price = h[i] if direction == 1 else l[i]
                r_now = direction * (mfe_price - pos["entry"]) / pos["risk_price"]
                pos["mfe_r"] = max(pos["mfe_r"], r_now)
                mae_price = l[i] if direction == 1 else h[i]
                pos["mae_r"] = min(pos["mae_r"], direction * (mae_price - pos["entry"]) / pos["risk_price"])

                if risk.breakeven_at_r > 0 and not pos["be_done"] and pos["mfe_r"] >= risk.breakeven_at_r:
                    be = pos["entry"] + direction * risk.breakeven_offset_r * pos["risk_price"]
                    pos["stop"] = max(pos["stop"], be) if direction == 1 else min(pos["stop"], be)
                    pos["be_done"] = True

                if risk.trail_start_r > 0 and pos["mfe_r"] >= risk.trail_start_r:
                    a_now = stop_ref[i]
                    if np.isfinite(a_now) and a_now > 0:
                        trail = (c[i] - direction * risk.trail_atr_mult * a_now) if direction == 1 else \
                                (c[i] + risk.trail_atr_mult * a_now)
                        pos["stop"] = max(pos["stop"], trail) if direction == 1 else min(pos["stop"], trail)
            else:
                # Exit side of the spread, plus commission and any carry.
                fill = exit_price - half_spread if direction == 1 else exit_price + half_spread
                gross = direction * (fill - pos["entry"])
                days_held = _bars_to_days(idx, pos["entry_i"], i)
                carry = costs.swap_pips_per_day * pip * days_held
                net = gross - commission_price + carry
                r_mult = net / pos["risk_price"]
                pnl = r_mult * pos["risk_amount"]

                equity += pnl
                peak_equity = max(peak_equity, equity)

                trades.append(dict(
                    entry_time=idx[pos["entry_i"]],
                    exit_time=idx[i],
                    direction="long" if direction == 1 else "short",
                    entry=pos["entry"],
                    exit=fill,
                    stop=pos["init_stop"],
                    target=pos["target"],
                    stop_pips=pos["risk_price"] / pip,
                    bars_held=bars_held,
                    hours_held=_bars_to_days(idx, pos["entry_i"], i) * 24.0,
                    r=r_mult,
                    mfe_r=pos["mfe_r"],
                    mae_r=pos["mae_r"],
                    risk_amount=pos["risk_amount"],
                    pnl=pnl,
                    equity=equity,
                    reason=reason,
                ))
                pos = None

        # ---- 2. fill a pending order at this bar's open -------------------
        if pos is None and pending != 0 and np.isfinite(pending_ref) and pending_ref > 0:
            direction = pending
            raw = o[i]
            entry = raw + direction * (half_spread + entry_slip)
            risk_price = risk.atr_stop_mult * pending_ref
            if risk_price > 0:
                init_stop = entry - direction * risk_price
                target = entry + direction * risk.rr_target * risk_price
                base_equity = equity if risk.compound else account.starting_equity
                risk_amount = base_equity * risk.risk_pct / 100.0
                pos = dict(dir=direction, entry=entry, stop=init_stop, init_stop=init_stop,
                           target=target, risk_price=risk_price, risk_amount=risk_amount,
                           entry_i=i, mfe_r=0.0, mae_r=0.0, be_done=False)
            pending, pending_ref = 0, np.nan

        # ---- 3. read this bar's close for a signal ------------------------
        if pos is None and pending == 0:
            ref = stop_ref[i]
            if np.isfinite(ref) and ref > 0:
                if long_sig[i] and account.allow_longs:
                    pending, pending_ref = 1, ref
                elif short_sig[i] and account.allow_shorts:
                    pending, pending_ref = -1, ref

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        eq = pd.Series([account.starting_equity], index=[idx[0]], name="equity")
        return BacktestResult(tdf, eq, empty_stats(account.starting_equity),
                              _config_dump(risk, costs, account))

    eq = pd.concat([
        pd.Series([account.starting_equity], index=[idx[0]]),
        tdf.set_index("exit_time")["equity"],
    ])
    eq.name = "equity"
    stats = compute_stats(tdf, eq, account.starting_equity, risk, costs, symbol, idx)
    return BacktestResult(tdf, eq, stats, _config_dump(risk, costs, account))


def _bars_to_days(index: pd.DatetimeIndex, i0: int, i1: int) -> float:
    return max(0.0, (index[i1] - index[i0]).total_seconds() / 86400.0)


def _validate_ohlc(df: pd.DataFrame) -> None:
    missing = [c for c in OHLC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLC frame is missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted ascending")
    bad = (df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"]) | \
          (df["low"] > df["open"]) | (df["low"] > df["close"])
    if bool(bad.any()):
        raise ValueError(f"{int(bad.sum())} bars violate high/low bounds. Clean the data first.")


def _config_dump(risk, costs, account) -> Dict[str, dict]:
    from dataclasses import asdict
    return {"risk": asdict(risk), "costs": asdict(costs), "account": asdict(account)}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def empty_stats(start_equity: float) -> Dict[str, float]:
    keys = ["n_trades", "win_rate", "expectancy_r", "r_std", "profit_factor", "total_return_pct",
            "max_dd_pct", "max_dd_r", "longest_loss_streak", "sharpe", "sortino", "calmar",
            "avg_bars_held", "avg_hours_held", "median_hours_winners", "median_hours_losers",
            "breakeven_win_rate", "win_rate_edge", "kelly_pct", "trades_needed", "cost_r",
            "final_equity", "target_hit_rate", "stop_hit_rate", "avg_win_r", "avg_loss_r",
            "t_stat", "years"]
    d = {k: float("nan") for k in keys}
    d["n_trades"] = 0
    d["final_equity"] = start_equity
    return d


def compute_stats(
    trades: pd.DataFrame,
    equity: pd.Series,
    start_equity: float,
    risk: RiskConfig,
    costs: CostConfig,
    symbol: str,
    index: pd.DatetimeIndex,
) -> Dict[str, float]:
    r = trades["r"].to_numpy(float)
    n = len(r)
    wins, losses = r[r > 0], r[r <= 0]

    years = max((index[-1] - index[0]).days / 365.25, 1e-9)
    expectancy = float(r.mean())
    r_std = float(r.std(ddof=1)) if n > 1 else float("nan")

    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    curve = equity.to_numpy(float)
    running_peak = np.maximum.accumulate(curve)
    dd = (curve - running_peak) / running_peak
    max_dd_pct = float(-dd.min() * 100)

    # Drawdown measured in R, which is independent of position sizing choices.
    cum_r = np.concatenate([[0.0], np.cumsum(r)])
    peak_r = np.maximum.accumulate(cum_r)
    max_dd_r = float((peak_r - cum_r).max())

    streak = best_streak = 0
    for x in r:
        if x <= 0:
            streak += 1
            best_streak = max(best_streak, streak)
        else:
            streak = 0

    trades_per_year = n / years
    sharpe = float("nan")
    if n > 1 and r_std > 0:
        sharpe = expectancy / r_std * math.sqrt(trades_per_year)
    downside = r[r < 0]
    sortino = float("nan")
    if downside.size > 1:
        dstd = float(downside.std(ddof=1))
        if dstd > 0:
            sortino = expectancy / dstd * math.sqrt(trades_per_year)

    total_return_pct = (curve[-1] / start_equity - 1) * 100
    if curve[-1] > 0:
        try:
            cagr = ((curve[-1] / start_equity) ** (1 / years) - 1) * 100
        except (OverflowError, FloatingPointError):
            cagr = float("inf")
        if not np.isfinite(cagr):
            cagr = float("inf")
    else:
        cagr = -100.0
    calmar = cagr / max_dd_pct if max_dd_pct > 0 else float("nan")

    # Cost drag expressed in R, using the median stop distance.
    median_stop_pips = float(trades["stop_pips"].median())
    cost_pips = costs.spread_pips + costs.commission_pips + costs.entry_slippage_pips
    cost_r = cost_pips / median_stop_pips if median_stop_pips > 0 else float("nan")

    win_rate = float((r > 0).mean())
    breakeven_wr = (1.0 + cost_r) / (risk.rr_target + 1.0) if np.isfinite(cost_r) else \
                   1.0 / (risk.rr_target + 1.0)

    # Fractional Kelly on an R-multiple bet, capped for sanity.
    avg_win_r = float(wins.mean()) if wins.size else 0.0
    avg_loss_r = float(-losses.mean()) if losses.size else 0.0
    kelly = 0.0
    if avg_loss_r > 0 and avg_win_r > 0:
        b = avg_win_r / avg_loss_r
        kelly = max(0.0, (win_rate * (b + 1) - 1) / b)

    t_stat = expectancy / (r_std / math.sqrt(n)) if n > 1 and r_std > 0 else float("nan")
    trades_needed = (1.96 * r_std / expectancy) ** 2 if n > 1 and r_std > 0 and expectancy > 0 else float("nan")

    win_mask = trades["r"] > 0
    return {
        "n_trades": float(n),
        "years": float(years),
        "win_rate": win_rate * 100,
        "expectancy_r": expectancy,
        "r_std": r_std,
        "profit_factor": profit_factor,
        "total_return_pct": float(total_return_pct),
        "cagr_pct": float(cagr),
        "max_dd_pct": max_dd_pct,
        "max_dd_r": max_dd_r,
        "longest_loss_streak": float(best_streak),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "avg_bars_held": float(trades["bars_held"].mean()),
        "avg_hours_held": float(trades["hours_held"].mean()),
        "median_hours_winners": float(trades.loc[win_mask, "hours_held"].median()) if win_mask.any() else float("nan"),
        "median_hours_losers": float(trades.loc[~win_mask, "hours_held"].median()) if (~win_mask).any() else float("nan"),
        "breakeven_win_rate": breakeven_wr * 100,
        "win_rate_edge": (win_rate - breakeven_wr) * 100,
        "kelly_pct": kelly * 100,
        "trades_needed": trades_needed,
        "cost_r": cost_r,
        "final_equity": float(curve[-1]),
        "target_hit_rate": float((trades["reason"] == "target").mean()) * 100,
        "stop_hit_rate": float((trades["reason"] == "stop").mean()) * 100,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "t_stat": t_stat,
        "trades_per_year": float(trades_per_year),
    }


def breakeven_win_rate(rr: float, cost_r: float = 0.0) -> float:
    """Win rate needed just to break even at a given reward-to-risk, after costs."""
    return (1.0 + cost_r) / (rr + 1.0)


def win_rate_standard_error(p: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return math.sqrt(max(p * (1 - p), 0.0) / n)


# ---------------------------------------------------------------------------
# Monte Carlo on realised trades
# ---------------------------------------------------------------------------

def monte_carlo(
    r_series: Sequence[float],
    risk_pct: float,
    n_paths: int = 4000,
    path_len: Optional[int] = None,
    ruin_dd_pct: float = 40.0,
    seed: int = 7,
    compound: bool = True,
    chunk: int = 400,
) -> Dict[str, object]:
    """
    Bootstrap the realised R sequence to see what the SAME edge could have looked like
    in a different order. This does not create new information. It answers one question
    only: given this distribution of outcomes, how ugly can the path get?

    Paths are simulated in chunks in float32 so peak memory stays flat regardless of
    n_paths. A naive vectorised version allocates n_paths x path_len several times over,
    which is how these apps get killed on a small container.
    """
    r = np.asarray(list(r_series), dtype=np.float32)
    if r.size == 0:
        return {}
    rng = np.random.default_rng(seed)
    m = int(path_len or r.size)
    f = np.float32(risk_pct / 100.0)

    max_dd = np.empty(n_paths, dtype=np.float32)
    finals = np.empty(n_paths, dtype=np.float32)

    done = 0
    while done < n_paths:
        k = min(chunk, n_paths - done)
        draws = rng.choice(r, size=(k, m), replace=True) * f
        if compound:
            curve = np.cumprod(np.maximum(1.0 + draws, 1e-9), axis=1)
        else:
            curve = 1.0 + np.cumsum(draws, axis=1)
        peaks = np.maximum.accumulate(curve, axis=1)
        dd = (peaks - curve) / peaks
        max_dd[done:done + k] = dd.max(axis=1)
        finals[done:done + k] = curve[:, -1]
        done += k
        del draws, curve, peaks, dd

    max_dd = max_dd * 100
    return {
        "final_p05": float(np.percentile(finals, 5) - 1) * 100,
        "final_p50": float(np.percentile(finals, 50) - 1) * 100,
        "final_p95": float(np.percentile(finals, 95) - 1) * 100,
        "prob_profit": float((finals > 1.0).mean()) * 100,
        "dd_p50": float(np.percentile(max_dd, 50)),
        "dd_p95": float(np.percentile(max_dd, 95)),
        "dd_worst": float(max_dd.max()),
        "prob_ruin": float((max_dd >= ruin_dd_pct).mean()) * 100,
        "max_dd_samples": max_dd,
        "final_samples": (finals - 1) * 100,
    }


def growth_projection(
    r_series: Sequence[float],
    risk_pct: float,
    targets: Sequence[float] = (2.0, 4.0, 8.0),
    give_up_dd_pct: float = 40.0,
    max_trades: int = 3000,
    n_paths: int = 3000,
    trades_per_year: float = float("nan"),
    seed: int = 11,
    compound: bool = True,
    chunk: int = 250,
) -> pd.DataFrame:
    """
    First passage analysis. For each account multiple in `targets`, bootstrap the realised
    R sequence and ask which comes first: the account reaches that multiple, or it draws
    down by `give_up_dd_pct` and you stop.

    This is the honest answer to "can it turn 1 into 4". Turning 1 into 4 is not a property
    of the reward-to-risk setting. It is a property of expectancy per trade, how many trades
    you get, and how much you risk on each. A system with a real edge can still fail to
    reach 4x because the drawdown along the way takes you out first, and this measures how
    often that happens.

    Simulated in float32 chunks so peak memory does not scale with n_paths.
    """
    r = np.asarray(list(r_series), dtype=np.float32)
    targets = list(targets)
    if r.size == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    f = np.float32(risk_pct / 100.0)
    miss = max_trades + 1

    ruin_all = np.empty(n_paths, dtype=np.int32)
    hit_all = {t: np.empty(n_paths, dtype=np.int32) for t in targets}

    done = 0
    while done < n_paths:
        k = min(chunk, n_paths - done)
        draws = rng.choice(r, size=(k, max_trades), replace=True) * f
        if compound:
            curve = np.cumprod(np.maximum(1.0 + draws, 1e-9), axis=1)
        else:
            curve = 1.0 + np.cumsum(draws, axis=1)
        curve = np.concatenate([np.ones((k, 1), np.float32), curve], axis=1)

        peaks = np.maximum.accumulate(curve, axis=1)
        ruin = ((peaks - curve) / peaks) >= np.float32(give_up_dd_pct / 100.0)
        ruin_all[done:done + k] = np.where(ruin.any(axis=1), ruin.argmax(axis=1), miss)
        del peaks, ruin

        for t in targets:
            hit = curve >= np.float32(t)
            hit_all[t][done:done + k] = np.where(hit.any(axis=1), hit.argmax(axis=1), miss)
            del hit

        done += k
        del draws, curve

    out: List[dict] = []
    for t in targets:
        hit_idx = hit_all[t]
        reached = hit_idx < ruin_all
        stopped = ruin_all < hit_idx
        neither = ~reached & ~stopped

        got = hit_idx[reached]
        median_trades = float(np.median(got)) if got.size else float("nan")
        p90_trades = float(np.percentile(got, 90)) if got.size else float("nan")
        years = (median_trades / trades_per_year
                 if np.isfinite(median_trades) and np.isfinite(trades_per_year) and trades_per_year > 0
                 else float("nan"))
        out.append({
            "target_multiple": t,
            "prob_reach_first_pct": float(reached.mean() * 100),
            "prob_give_up_first_pct": float(stopped.mean() * 100),
            "prob_neither_pct": float(neither.mean() * 100),
            "median_trades_to_target": median_trades,
            "p90_trades_to_target": p90_trades,
            "median_years_to_target": years,
        })
    return pd.DataFrame(out)


def expectancy_per_dollar(expectancy_r: float, risk_pct: float) -> float:
    """
    Average account growth per trade, as a fraction. Risking `risk_pct` of equity to win
    `expectancy_r` R on average compounds at roughly this rate per trade. This is the only
    figure that turns 1 dollar into 4, and it is unrelated to the reward-to-risk setting
    except through its effect on expectancy.
    """
    return expectancy_r * risk_pct / 100.0


def trades_to_multiple(expectancy_r: float, risk_pct: float, multiple: float) -> float:
    """Deterministic approximation, ignoring path risk: how many trades to compound to X."""
    g = expectancy_per_dollar(expectancy_r, risk_pct)
    if g <= 0 or multiple <= 1:
        return float("nan")
    return math.log(multiple) / math.log(1.0 + g)


def losing_streak_probability(win_rate: float, n_trades: int, streak: int) -> float:
    """
    Probability of seeing at least one losing run of `streak` in `n_trades`.
    Useful sanity check before you risk money on a 1:8 system: at a 15 percent win
    rate, runs of 15 losses are ordinary, not evidence the system broke.
    """
    q = 1.0 - win_rate
    if q <= 0 or streak <= 0:
        return 0.0
    if streak > n_trades:
        return 0.0
    # Markov chain over current run length.
    state = np.zeros(streak + 1)
    state[0] = 1.0
    for _ in range(n_trades):
        nxt = np.zeros(streak + 1)
        nxt[streak] += state[streak]
        for k in range(streak):
            nxt[0] += state[k] * win_rate
            nxt[k + 1] += state[k] * q
        state = nxt
    return float(state[streak])


# ---------------------------------------------------------------------------
# Walk forward validation
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict
    train_expectancy_r: float
    train_trades: int
    test_expectancy_r: float
    test_trades: int


def _score(stats: Dict[str, float], min_trades: int) -> float:
    """Rank by expectancy scaled by sample size, so a 3 trade fluke cannot win."""
    if stats.get("n_trades", 0) < min_trades:
        return -1e9
    e = stats["expectancy_r"]
    if not np.isfinite(e):
        return -1e9
    return e * math.sqrt(stats["n_trades"])


def walk_forward(
    df: pd.DataFrame,
    strategy_fn: Callable[..., pd.DataFrame],
    base_params: dict,
    grid: Dict[str, list],
    symbol: str,
    risk: RiskConfig,
    costs: CostConfig,
    account: AccountConfig,
    n_folds: int = 5,
    train_multiple: float = 3.0,
    min_train_trades: int = 15,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Rolling walk forward. For each fold: grid search on the training window, then take
    the single best parameter set into the untouched test window. Only the stitched
    test trades are reported as out of sample.

    The number that matters is the ratio of out-of-sample expectancy to in-sample
    expectancy. Above roughly 0.5 is respectable. Below 0.3 means you fitted noise.
    """
    n = len(df)
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    step = n // (n_folds + int(train_multiple))
    if step < 200:
        raise ValueError("Not enough bars for this many folds. Use more data or fewer folds.")
    train_bars = int(step * train_multiple)

    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))] or [{}]

    folds: List[WalkForwardFold] = []
    oos_frames: List[pd.DataFrame] = []

    for k in range(n_folds):
        test_start = train_bars + k * step
        test_end = min(test_start + step, n)
        train_start = max(0, test_start - train_bars)
        if test_end - test_start < 100:
            break

        train_df = df.iloc[train_start:test_start]
        test_df = df.iloc[test_start:test_end]

        best, best_score, best_stats = None, -np.inf, None
        for combo in combos:
            params = {**base_params, **combo}
            try:
                sig = strategy_fn(train_df, **params)
                res = backtest(train_df, sig, symbol, risk, costs, account)
            except Exception:
                continue
            sc = _score(res.stats, min_train_trades)
            if sc > best_score:
                best, best_score, best_stats = params, sc, res.stats

        if best is None:
            continue

        test_sig = strategy_fn(test_df, **best)
        test_res = backtest(test_df, test_sig, symbol, risk, costs, account)

        folds.append(WalkForwardFold(
            fold=k + 1,
            train_start=train_df.index[0], train_end=train_df.index[-1],
            test_start=test_df.index[0], test_end=test_df.index[-1],
            params={kk: best[kk] for kk in keys} if keys else {},
            train_expectancy_r=best_stats["expectancy_r"],
            train_trades=int(best_stats["n_trades"]),
            test_expectancy_r=test_res.stats["expectancy_r"],
            test_trades=int(test_res.stats["n_trades"]),
        ))
        if not test_res.trades.empty:
            t = test_res.trades.copy()
            t["fold"] = k + 1
            oos_frames.append(t)

    fold_df = pd.DataFrame([f.__dict__ for f in folds])
    oos = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()

    summary: Dict[str, float] = {}
    if not fold_df.empty:
        is_mean = float(np.nanmean(fold_df["train_expectancy_r"]))
        oos_mean = float(np.nanmean(fold_df["test_expectancy_r"]))
        summary = {
            "folds": float(len(fold_df)),
            "is_expectancy_r": is_mean,
            "oos_expectancy_r": oos_mean,
            "efficiency": oos_mean / is_mean if is_mean not in (0.0,) and np.isfinite(is_mean) else float("nan"),
            "oos_trades": float(len(oos)),
            "profitable_folds_pct": float((fold_df["test_expectancy_r"] > 0).mean() * 100),
        }
        if not oos.empty:
            summary["oos_win_rate"] = float((oos["r"] > 0).mean() * 100)
            summary["oos_r_std"] = float(oos["r"].std(ddof=1)) if len(oos) > 1 else float("nan")
    return fold_df, oos, summary


def parameter_sweep(
    df: pd.DataFrame,
    strategy_fn: Callable[..., pd.DataFrame],
    base_params: dict,
    grid: Dict[str, list],
    symbol: str,
    risk: RiskConfig,
    costs: CostConfig,
    account: AccountConfig,
) -> pd.DataFrame:
    """
    Grid search for SENSITIVITY inspection, not for picking the winner. A healthy
    strategy shows a broad plateau of acceptable results. A single tall spike beside
    a field of losses is a curve fit, and it will not survive contact with your broker.
    """
    keys = list(grid.keys())
    rows = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        combo = dict(zip(keys, vals))
        params = {**base_params, **combo}
        try:
            sig = strategy_fn(df, **params)
            res = backtest(df, sig, symbol, risk, costs, account)
        except Exception:
            continue
        rows.append({**combo,
                     "n_trades": res.stats["n_trades"],
                     "expectancy_r": res.stats["expectancy_r"],
                     "win_rate": res.stats["win_rate"],
                     "profit_factor": res.stats["profit_factor"],
                     "max_dd_pct": res.stats["max_dd_pct"]})
    return pd.DataFrame(rows)


def rr_sweep(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    symbol: str,
    risk: RiskConfig,
    costs: CostConfig,
    account: AccountConfig,
    rr_values: Sequence[float] = (1, 2, 3, 4, 5, 6, 8, 10),
) -> pd.DataFrame:
    """
    Hold the entry logic fixed and vary only the target. This is the single most
    informative chart in the whole app, because it shows where YOUR entry stops
    being able to pay for a wider target.
    """
    rows = []
    for rr in rr_values:
        res = backtest(df, signals, symbol, replace(risk, rr_target=float(rr)), costs, account)
        s = res.stats
        rows.append({"rr": rr, "expectancy_r": s["expectancy_r"], "win_rate": s["win_rate"],
                     "breakeven_win_rate": s["breakeven_win_rate"], "n_trades": s["n_trades"],
                     "profit_factor": s["profit_factor"], "max_dd_pct": s["max_dd_pct"],
                     "avg_hours_held": s["avg_hours_held"],
                     "total_return_pct": s["total_return_pct"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Live setup reading. Not a forecast.
# ---------------------------------------------------------------------------

def current_setup(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    symbol: str,
    risk: RiskConfig,
    costs: CostConfig,
    equity: float,
    pip_value_per_lot: float = 10.0,
) -> Dict[str, object]:
    """
    Report whether the rule set's entry conditions are satisfied on the most recent
    CLOSED bar, and what the resulting order would look like. There is no probability
    attached, because the rule set does not produce one.
    """
    pip = pip_size(symbol)
    i = len(df) - 1
    ref = float(signals["stop_ref"].iloc[i])
    close = float(df["close"].iloc[i])

    if not np.isfinite(ref) or ref <= 0:
        return {"state": "warming up", "detail": "Not enough bars to compute the stop reference."}

    is_long = bool(signals["long"].iloc[i])
    is_short = bool(signals["short"].iloc[i])
    if not (is_long or is_short):
        return {"state": "no setup", "bar_time": df.index[i], "close": close,
                "detail": "Entry conditions are not met on the last closed bar."}

    direction = 1 if is_long else -1
    half_spread = 0.5 * costs.spread_pips * pip
    entry = close + direction * (half_spread + costs.entry_slippage_pips * pip)
    stop_dist = risk.atr_stop_mult * ref
    stop = entry - direction * stop_dist
    target = entry + direction * risk.rr_target * stop_dist
    risk_amount = equity * risk.risk_pct / 100.0
    stop_pips = stop_dist / pip

    return {
        "state": "setup active",
        "bar_time": df.index[i],
        "direction": "long" if is_long else "short",
        "reference_entry": entry,
        "stop": stop,
        "target": target,
        "stop_pips": stop_pips,
        "target_pips": stop_pips * risk.rr_target,
        "risk_amount": risk_amount,
        "lots": lots_for_risk(risk_amount, stop_pips, pip_value_per_lot),
        "detail": "Fills at the next bar open. Any later entry changes the stop distance "
                  "and therefore the size.",
    }


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------

def normalise_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Accept the usual exports (MT4/MT5 CSV, Dukascopy, HistData, yfinance) and return a
    clean UTC-indexed OHLC frame. Duplicate timestamps and zero-range bars are dropped
    because both inflate backtest results.
    """
    df = raw.copy()
    df.columns = [str(c).strip().lower().strip("<>").replace(" ", "_") for c in df.columns]

    # MT5 splits the stamp across <DATE> and <TIME>. Rejoin before anything else.
    if "date" in df.columns and "time" in df.columns:
        df["timestamp"] = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
        df = df.drop(columns=["date", "time"])

    rename = {"date": "timestamp", "datetime": "timestamp", "time": "timestamp",
              "gmt_time": "timestamp", "local_time": "timestamp", "<date>": "timestamp",
              "o": "open", "h": "high", "l": "low", "c": "close",
              "price": "close", "vol": "volume",
              "tickvol": "volume", "tick_volume": "volume"}
    df = df.rename(columns=rename)
    # yfinance ships both close and adj_close. Prefer the raw close for FX.
    if "adj_close" in df.columns:
        df = df.drop(columns=["adj_close"]) if "close" in df.columns else df.rename(columns={"adj_close": "close"})
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
        else:
            df = df.reset_index().rename(columns={"index": "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    df.index = df.index.tz_convert("UTC").tz_localize(None)

    for c in OHLC_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Could not find a '{c}' column after normalising. Columns seen: {list(df.columns)}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = OHLC_COLUMNS + (["volume"] if "volume" in df.columns else [])
    df = df[keep].dropna(subset=OHLC_COLUMNS)
    df = df[~df.index.duplicated(keep="first")]
    df = df[(df["high"] > df["low"])]
    return df


def bar_interval_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return float("nan")
    return float(pd.Series(index).diff().dt.total_seconds().median() / 3600.0)


def synthetic_ohlc(
    n: int = 8000,
    start: str = "2021-01-04",
    freq: str = "1h",
    seed: int = 3,
    base: float = 1.1000,
    trendiness: float = 0.0035,
) -> pd.DataFrame:
    """
    SYNTHETIC DATA. For exercising the engine only. Any result produced on this frame
    is meaningless as evidence about a real market. It exists so the app has something
    to render before you load your own history.

    The default trendiness is deliberately tuned so this series behaves like a nearly
    efficient market: a trend follower lands close to breakeven on it. If a demo data
    generator made every strategy look profitable it would be teaching you the wrong
    reflex before you ever touched real prices.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq)
    vol = 0.0007 * (1 + 0.5 * np.sin(np.arange(n) / 500.0))
    drift = np.zeros(n)
    state = 0.0
    for i in range(n):
        state = 0.995 * state + rng.normal(0, trendiness)
        drift[i] = state * vol[i]
    rets = drift + rng.normal(0, 1, n) * vol
    close = base * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[base], close[:-1]])
    wick = np.abs(rng.normal(0, 1, n)) * vol * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    return df[df.index.dayofweek < 5]
