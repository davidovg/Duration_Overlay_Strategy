"""
backtest.py
===========
Backtest harness for the EUR duration overlay (Bund / OAT / BTP), run off the
panel from build_panel.py.

WHAT IT COMPARES (the whole point of the project)
-------------------------------------------------
  1. carry        — vol-targeted carry only. The Bauer-Hamilton-defensible
                    floor: only the curve slope/roll-down, nothing fancy.
  2. carry+mom    — carry + time-series momentum (the durable factor pair).
  3. finca_static — carry + momentum + mean-reversion, equal-weight summed and
                    per-instrument vol-targeted. This mirrors the live FINCA
                    construction that blew up −11.8% in Mar-2026.
  4. improved     — same signals, but with the two overlays that were missing:
                    a PORTFOLIO-level vol target and a DRAWDOWN de-gross. The
                    test: does this contain the Mar-2026 tail without giving
                    up the long-run carry premium?

METHOD / NO LOOK-AHEAD
----------------------
  * Signals come from the panel, which is already built on expanding windows.
  * Here we additionally z-score on an EXPANDING window and lag every position
    one day before applying it to returns (signal at close t → position for
    t+1). Per-instrument vol uses trailing realized vol, lagged.
  * Sizing: w_i = signal_i · (target_inst_vol / realised_vol_i), so each market
    contributes ~target_inst_vol of risk at unit signal. Portfolio daily return
    = Σ_i w_{i,t-1}·ret_{i,t} − costs.
  * Costs: linear on turnover (bond futures are cheap; default 0.5bp per unit).

This is a research backtest, NOT investment advice. Past performance of a
historical simulation is not indicative of future results.

USAGE
    python backtest.py
    python backtest.py --outputs-dir outputs --inst-vol 0.04 --port-vol 0.06
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MKT2CC = {"bund": "de", "oat": "fr", "btp": "it"}
TRADING_DAYS = 252


# ─── helpers ────────────────────────────────────────────────────────────────
def zexp(s: pd.Series, minp: int = 252) -> pd.Series:
    """Expanding-window z-score (de-meaned). Used for genuinely symmetric
    signals like momentum."""
    m = s.expanding(minp).mean()
    sd = s.expanding(minp).std()
    return ((s - m) / sd.replace(0, np.nan)).clip(-3, 3)


def scale_only(s: pd.Series, minp: int = 252) -> pd.Series:
    """Standardise MAGNITUDE without de-meaning, so a structurally-positive
    signal (carry is ~always positive on an upward curve) stays a structural
    LONG. This is what makes carry harvest the term premium — and what makes it
    negatively skewed and tail-prone, i.e. the FINCA construction we must test."""
    sd = s.expanding(minp).std()
    return (s / sd.replace(0, np.nan)).clip(-3, 3)


def metrics(ret: pd.Series, name: str) -> dict:
    r = ret.dropna()
    if len(r) < 60:
        return {"strategy": name, "n": len(r)}
    ann_ret = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol else np.nan
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1)
    maxdd = dd.min()
    calmar = ann_ret / abs(maxdd) if maxdd else np.nan
    # worst rolling 21-day (≈1m) return
    worst_1m = (1 + r).rolling(21).apply(np.prod, raw=True).min() - 1
    return {
        "strategy": name, "n": len(r),
        "ann_ret%": ann_ret * 100, "ann_vol%": ann_vol * 100,
        "sharpe": sharpe, "maxDD%": maxdd * 100, "calmar": calmar,
        "skew": r.skew(), "hit%": (r > 0).mean() * 100,
        "worst1m%": worst_1m * 100,
    }


def stress_window(ret: pd.Series, lo="2026-02-01", hi="2026-04-30") -> float:
    w = ret.loc[lo:hi].dropna()
    return ((1 + w).prod() - 1) * 100 if len(w) else np.nan


# ─── position construction ──────────────────────────────────────────────────
def build_positions(panel, rets, signal_weights, inst_vol, wcap=4.0):
    """signal_weights: dict like {'carry':1} or {'carry':1,'mom':1,'mr':1}.
    Returns (positions DataFrame [date×market], per-market vols df)."""
    pos = {}
    for mkt, cc in MKT2CC.items():
        comp = {}
        if "carry" in signal_weights:
            comp["carry"] = scale_only(panel[f"{cc}_carry_roll"])  # structural long
        if "mom" in signal_weights:
            comp["mom"] = zexp(panel[f"{mkt}_mom"])                # symmetric trend
        if "mr" in signal_weights:
            comp["mr"] = scale_only(panel[f"{cc}_meanrev"])        # expected reversion
        num = sum(signal_weights[k] * comp[k].fillna(0) for k in comp)
        den = sum(signal_weights[k] * comp[k].notna() for k in comp).replace(0, np.nan)
        sig = num / den  # average of available components
        vol = panel[f"{mkt}_rvol21"].shift(1).clip(lower=0.02)
        w = (sig * (inst_vol / vol)).clip(-wcap, wcap)
        # only hold where the market actually trades
        w = w.where(rets[f"{mkt}_ret"].reindex(panel.index).notna())
        pos[mkt] = w
    return pd.DataFrame(pos)


def strat_returns(positions, rets, tc=5e-4):
    """Portfolio daily return: Σ_i w_{t-1}·ret_t − turnover cost."""
    pos = positions.reindex(rets.index).fillna(0.0)
    pnl = pd.Series(0.0, index=rets.index)
    for mkt in positions.columns:
        pnl = pnl.add((pos[mkt].shift(1) * rets[f"{mkt}_ret"]).fillna(0.0), fill_value=0.0)
    turnover = pos.diff().abs().sum(axis=1).fillna(0.0)
    return pnl - tc * turnover


def apply_overlays(base_ret, port_vol, dd_start=-0.05, dd_floor=-0.15,
                   min_scale=0.3, vol_win=63, max_lev=2.0):
    """Portfolio vol target + drawdown de-gross, both causal (shifted).
    Returns the scale series to multiply the base strategy return by."""
    # 1) vol target
    realized = base_ret.rolling(vol_win).std() * np.sqrt(TRADING_DAYS)
    vscale = (port_vol / realized.shift(1)).clip(0, max_lev).fillna(0.0)
    voltgt = base_ret * vscale
    # 2) drawdown de-gross off the vol-targeted equity (reacts next day)
    eq = (1 + voltgt).cumprod()
    dd = (eq / eq.cummax() - 1).shift(1).fillna(0.0)
    # linear taper from 1.0 at dd_start to min_scale at dd_floor
    dscale = ((dd - dd_floor) / (dd_start - dd_floor)).clip(min_scale, 1.0)
    return vscale * dscale


# ─── main ───────────────────────────────────────────────────────────────────
def run(outdir: Path, inst_vol: float, port_vol: float, tc: float) -> None:
    panel = pd.read_csv(outdir / "analysis_panel.csv", index_col=0, parse_dates=True)
    rets = pd.read_csv(outdir / "returns_markets.csv", index_col=0, parse_dates=True)
    panel = panel.sort_index()
    rets = rets.reindex(panel.index)

    specs = {
        "carry":        {"carry": 1},
        "carry+mom":    {"carry": 1, "mom": 1},
        "finca_static": {"carry": 1, "mom": 1, "mr": 1},
    }
    strat_ret = {}
    for name, w in specs.items():
        pos = build_positions(panel, rets, w, inst_vol)
        strat_ret[name] = strat_returns(pos, rets, tc)

    # improved = finca_static signals + portfolio overlays
    base = strat_ret["finca_static"]
    scale = apply_overlays(base, port_vol)
    strat_ret["improved"] = base * scale

    # passive reference: constant long 10y duration (inverse-vol Bund)
    bvol = panel["bund_rvol21"].shift(1).clip(lower=0.01)
    passive = ((inst_vol / bvol).clip(0, 5).shift(1) * rets["bund_ret"]).fillna(0.0)
    strat_ret["passive_long_bund"] = passive

    # ── report ──
    rows = [metrics(strat_ret[k], k) for k in
            ["passive_long_bund", "carry", "carry+mom", "finca_static", "improved"]]
    tbl = pd.DataFrame(rows).set_index("strategy")
    cols = ["n", "ann_ret%", "ann_vol%", "sharpe", "maxDD%", "calmar",
            "skew", "hit%", "worst1m%"]
    print("\n" + "=" * 78)
    print(" Duration overlay — strategy comparison (full sample)")
    print("=" * 78)
    print(tbl[cols].round(2).to_string())

    print("\n" + "-" * 78)
    print(" Mar-2026 stress window (cumulative return, Feb 1 – Apr 30 2026)")
    print("-" * 78)
    for k in ["passive_long_bund", "finca_static", "improved"]:
        print(f"   {k:20s} {stress_window(strat_ret[k]):+6.2f}%")
    print("   (FINCA-static is the construction that lost in Mar-2026; "
          "'improved' adds the overlays)")

    # save the equity curves and daily returns for inspection
    out = pd.DataFrame(strat_ret)
    out.to_csv(outdir / "backtest_returns.csv")
    (1 + out.fillna(0)).cumprod().to_csv(outdir / "backtest_equity.csv")
    print(f"\n  wrote backtest_returns.csv and backtest_equity.csv to {outdir}/")
    print("\n  NB: research simulation, not advice. Benchmark is a passive long-"
          "duration proxy;\n  the mandate's Bloomberg 1-10Y index is not on the feed.")


def main() -> None:
    p = argparse.ArgumentParser(description="EUR duration overlay backtest")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--inst-vol", type=float, default=0.04,
                   help="per-instrument annual vol target at unit signal")
    p.add_argument("--port-vol", type=float, default=0.06,
                   help="portfolio annual vol target for the 'improved' overlay")
    p.add_argument("--tc", type=float, default=2e-4,
                   help="transaction cost per unit of position turnover")
    args = p.parse_args()
    run(Path(args.outputs_dir), args.inst_vol, args.port_vol, args.tc)


if __name__ == "__main__":
    main()
