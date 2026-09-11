"""
test_engine.py
Run with: python test_engine.py

These are correctness tests, not performance tests. They check the properties that,
if broken, would make every number the app prints a lie.
"""

import numpy as np
import pandas as pd

from fx_engine import (
    AccountConfig, CostConfig, RiskConfig, backtest, breakeven_win_rate,
    current_setup, losing_streak_probability, monte_carlo, normalise_ohlc,
    parameter_sweep, rr_sweep, strat_donchian_breakout, strat_trend_pullback,
    strat_volatility_squeeze, synthetic_ohlc, walk_forward, STRATEGIES,
)

PASS, FAIL = "ok  ", "FAIL"
results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition), detail))
    print(f"[{PASS if condition else FAIL}] {name} {detail}")


def hand_built_frame():
    """A tiny deterministic market where the trade outcome is known by hand."""
    idx = pd.date_range("2024-01-01", periods=8, freq="1h")
    return pd.DataFrame({
        "open":  [1.0000, 1.0000, 1.0020, 1.0040, 1.0060, 1.0080, 1.0100, 1.0120],
        "high":  [1.0010, 1.0020, 1.0040, 1.0060, 1.0080, 1.0100, 1.0500, 1.0130],
        "low":   [0.9990, 0.9995, 1.0010, 1.0030, 1.0050, 1.0070, 1.0090, 1.0110],
        "close": [1.0000, 1.0020, 1.0040, 1.0060, 1.0080, 1.0100, 1.0120, 1.0125],
    }, index=idx)


def test_no_lookahead_fill():
    df = hand_built_frame()
    sig = pd.DataFrame({"long": False, "short": False, "stop_ref": 0.0010}, index=df.index)
    sig.loc[df.index[1], "long"] = True  # signal on the close of bar 1

    risk = RiskConfig(risk_pct=1.0, rr_target=4.0, atr_stop_mult=1.0, compound=False)
    costs = CostConfig(spread_pips=0, commission_pips=0, entry_slippage_pips=0, stop_slippage_pips=0)
    res = backtest(df, sig, "EURUSD", risk, costs, AccountConfig(starting_equity=10_000))

    check("entry fills at next bar open, not signal bar",
          len(res.trades) == 1 and abs(res.trades.iloc[0]["entry"] - df["open"].iloc[2]) < 1e-12,
          f"entry={res.trades.iloc[0]['entry']:.5f} expected={df['open'].iloc[2]:.5f}")

    t = res.trades.iloc[0]
    check("target exit produces exactly +R target", abs(t["r"] - 4.0) < 1e-9, f"r={t['r']:.6f}")
    check("pnl equals R times risk amount", abs(t["pnl"] - 4.0 * 100.0) < 1e-6, f"pnl={t['pnl']:.2f}")


def test_stop_wins_ambiguous_bar():
    """A bar that spans both stop and target must be recorded as a loss."""
    idx = pd.date_range("2024-01-01", periods=4, freq="1h")
    df = pd.DataFrame({
        "open":  [1.0000, 1.0000, 1.0000, 1.0000],
        "high":  [1.0005, 1.0005, 1.0500, 1.0005],
        "low":   [0.9995, 0.9995, 0.9500, 0.9995],
        "close": [1.0000, 1.0000, 1.0000, 1.0000],
    }, index=idx)
    sig = pd.DataFrame({"long": False, "short": False, "stop_ref": 0.0010}, index=idx)
    sig.loc[idx[0], "long"] = True

    risk = RiskConfig(rr_target=4.0, atr_stop_mult=1.0, compound=False)
    costs = CostConfig(spread_pips=0, commission_pips=0, entry_slippage_pips=0, stop_slippage_pips=0)
    res = backtest(df, sig, "EURUSD", risk, costs, AccountConfig())
    t = res.trades.iloc[0]
    check("ambiguous bar resolves to the stop", t["reason"] == "stop" and abs(t["r"] + 1.0) < 1e-9,
          f"reason={t['reason']} r={t['r']:.4f}")


def test_costs_reduce_expectancy():
    df = synthetic_ohlc(n=6000)
    sig = strat_donchian_breakout(df)
    risk, acct = RiskConfig(rr_target=4.0), AccountConfig()
    free = backtest(df, sig, "EURUSD", risk, CostConfig(0, 0, 0, 0), acct)
    real = backtest(df, sig, "EURUSD", risk, CostConfig(1.4, 0.7, 0.3, 1.0), acct)
    check("adding costs lowers expectancy",
          real.stats["expectancy_r"] < free.stats["expectancy_r"],
          f"{real.stats['expectancy_r']:.4f} < {free.stats['expectancy_r']:.4f}")


def test_breakeven_math():
    check("breakeven win rate at 1:4 with no cost is 20 percent",
          abs(breakeven_win_rate(4.0) - 0.20) < 1e-12)
    check("breakeven win rate at 1:8 with no cost is 11.11 percent",
          abs(breakeven_win_rate(8.0) - 1 / 9) < 1e-12)


def test_streak_probability():
    p = losing_streak_probability(win_rate=0.20, n_trades=200, streak=10)
    check("a 10 loss run at 20 percent win rate over 200 trades is common",
          0.85 < p <= 1.0, f"p={p:.3f}")
    p2 = losing_streak_probability(win_rate=0.55, n_trades=200, streak=10)
    check("the same run is rare at a 55 percent win rate", p2 < 0.15, f"p={p2:.3f}")


def test_monte_carlo_shape():
    r = np.array([-1.0] * 80 + [4.0] * 20)
    mc = monte_carlo(r, risk_pct=1.0, n_paths=2000, seed=1)
    check("monte carlo reports a drawdown distribution",
          mc["dd_p95"] > mc["dd_p50"] > 0, f"p50={mc['dd_p50']:.1f} p95={mc['dd_p95']:.1f}")
    check("zero expectancy sequence is roughly a coin flip",
          30 < mc["prob_profit"] < 70, f"prob_profit={mc['prob_profit']:.1f}")


def test_normalise_variants():
    idx = pd.date_range("2024-01-01", periods=50, freq="1h")
    raw = pd.DataFrame({"Date": idx, "Open": 1.0, "High": 1.01, "Low": 0.99, "Close": 1.005})
    out = normalise_ohlc(raw)
    check("normaliser handles capitalised MT style columns", list(out.columns[:4]) == ["open", "high", "low", "close"])
    check("normaliser keeps all valid rows", len(out) == 50, f"rows={len(out)}")


def test_walk_forward_runs():
    df = synthetic_ohlc(n=14000)
    spec = STRATEGIES["Donchian breakout"]
    folds, oos, summary = walk_forward(
        df, spec["fn"], spec["defaults"], spec["grid"], "EURUSD",
        RiskConfig(rr_target=4.0), CostConfig(1.2, 0.5, 0.2, 0.8), AccountConfig(),
        n_folds=4, train_multiple=3.0, min_train_trades=8,
    )
    check("walk forward produced folds", len(folds) >= 2, f"folds={len(folds)}")
    check("walk forward reports an efficiency ratio", "efficiency" in summary)
    if not folds.empty:
        overlap = all(folds.iloc[i]["test_start"] >= folds.iloc[i]["train_end"] for i in range(len(folds)))
        check("test window never starts before the training window ends", overlap)


def test_rr_sweep_monotone_win_rate():
    df = synthetic_ohlc(n=9000)
    sig = strat_donchian_breakout(df)
    sweep = rr_sweep(df, sig, "EURUSD", RiskConfig(), CostConfig(1.2, 0.5, 0.2, 0.8),
                     AccountConfig(), rr_values=(1, 2, 4, 8))
    wr = sweep["win_rate"].to_numpy()
    check("win rate falls as the target widens", all(wr[i] >= wr[i + 1] - 1e-9 for i in range(len(wr) - 1)),
          f"win rates={np.round(wr, 2).tolist()}")


def test_all_strategies_execute():
    df = synthetic_ohlc(n=9000)
    for name, spec in STRATEGIES.items():
        sig = spec["fn"](df, **spec["defaults"])
        res = backtest(df, sig, "EURUSD", RiskConfig(), CostConfig(), AccountConfig())
        check(f"strategy runs: {name}", isinstance(res.stats["n_trades"], float),
              f"trades={int(res.stats['n_trades'])}")


def test_current_setup_is_not_a_forecast():
    df = synthetic_ohlc(n=3000)
    sig = strat_donchian_breakout(df)
    out = current_setup(df, sig, "EURUSD", RiskConfig(), CostConfig(), 10_000)
    check("live reading returns a state, never a probability",
          out["state"] in {"setup active", "no setup", "warming up"} and "confidence" not in out,
          f"state={out['state']}")


if __name__ == "__main__":
    test_no_lookahead_fill()
    test_stop_wins_ambiguous_bar()
    test_costs_reduce_expectancy()
    test_breakeven_math()
    test_streak_probability()
    test_monte_carlo_shape()
    test_normalise_variants()
    test_walk_forward_runs()
    test_rr_sweep_monotone_win_rate()
    test_all_strategies_execute()
    test_current_setup_is_not_a_forecast()

    failed = [r for r in results if not r[1]]
    print("\n" + "=" * 60)
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
    if failed:
        raise SystemExit(1)
