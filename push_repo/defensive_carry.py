"""
defensive_carry.py
==================
Rebuild the carry signal properly and add a crash-avoidance gate, then show the
drawdown reduction on the real panel.

WHY CARRY CRASHES, AND THE FIX
------------------------------
Structural-long carry harvests the term premium but is short convexity to rate
selloffs (−50% DD in this sample, almost all of it the 2022 bear off record-low
yields). The research-backed protector is TREND (it hedges carry, has no
negative skew, and is convex in high vol) plus VALUE (don't run the long when
yields are rich / the term premium is thin — the 2022 setup).

SIGNALS (per market, all causal)
  carry+roll   : (y10 − short)/12 + ModDur·(local curve slope)·(1/12), the
                 expected 1-month return if the curve is unchanged. The local
                 slope is taken from a smooth 3-point (5/10/30Y) fit, not the
                 crude (y10−y5)/5.
  term premium : y10 − E[avg short rate], E[·] from an expanding AR(1) on the
                 short rate. Scales carry down when the premium is thin/negative.
  trend        : absolute momentum — trailing 6m return of the long-duration
                 position, vol-scaled. The carry-crash hedge.
  value        : y10 vs its trailing 5y mean, vol-scaled. Positive (long ok)
                 when yields are high/cheap, negative when rich/crash-prone.

GATE
  naive    : w ∝ carry  (structural long, the thing that crashed)
  gated    : w ∝ carry · g, g ∈ [g_min, 1] from trend+value — cuts the long (to
             cash, optionally a mild short) when trend is adverse AND yields rich.

This is a research simulation, not investment advice.

USAGE
    python defensive_carry.py --outputs-dir outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MKT2CC = {"bund": "de", "oat": "fr", "btp": "it"}
MODDUR = {"bund": 8.5, "oat": 8.5, "btp": 7.5}
TD = 252


def scale_only(s, minp=252):
    return (s / s.expanding(minp).std().replace(0, np.nan)).clip(-3, 3)


def local_slope_10y(y5, y10, y30):
    """dy/dτ at τ=10 from the parabola through (5,y5),(10,y10),(30,y30)."""
    # Lagrange derivative at x=10 for nodes 5,10,30
    x1, x2, x3 = 5.0, 10.0, 30.0
    d = (y5 * (2 * 10 - x2 - x3) / ((x1 - x2) * (x1 - x3))
         + y10 * (2 * 10 - x1 - x3) / ((x2 - x1) * (x2 - x3))
         + y30 * (2 * 10 - x1 - x2) / ((x3 - x1) * (x3 - x2)))
    return d  # %/yr


def ar1_expected_avg_short(short, horizon_m=120, minp=252):
    """Expanding AR(1) on the short rate → expected average short over `horizon_m`
    months. Used for a term-premium proxy. Causal."""
    s = short.dropna()
    out = pd.Series(index=short.index, dtype=float)
    # refit monthly for speed; apply forward
    refits = s.resample("ME").last().index
    bounds = list(refits) + [s.index[-1] + pd.Timedelta(days=1)]
    for i, rd in enumerate(refits):
        tr = s.loc[:rd]
        if len(tr) < minp:
            continue
        x0, x1 = tr.shift(1).dropna(), tr.iloc[1:]
        x0 = x0.loc[x1.index]
        b = np.polyfit(x0.values, x1.values, 1)
        phi = float(np.clip(b[0], 0.0, 0.9999)); c = float(b[1])
        m = c / (1 - phi)                       # long-run mean (monthly persistence≈daily here; rough)
        # average of AR(1) forecast over horizon (geometric decay to mean)
        # E[avg] = m + (s_now - m) * (1 - phi^H) / (H * (1 - phi))
        nxt = bounds[i + 1]
        win = (short.index > rd) & (short.index < nxt)
        s_now = short.where(win).ffill()
        decay = (1 - phi ** horizon_m) / (horizon_m * (1 - phi)) if phi < 1 else 1.0
        out.loc[win] = m + (s_now.loc[win] - m) * decay
    return out


def build(outdir: Path):
    yds = {cc: pd.read_csv(outdir / f"yields_{cc}.csv", index_col=0, parse_dates=True)
           for cc in ("de", "fr", "it")}
    sr = pd.read_csv(outdir / "short_rates.csv", index_col=0, parse_dates=True)
    short = sr["euribor_3m"]
    rets = pd.read_csv(outdir / "returns_markets.csv", index_col=0, parse_dates=True)
    panel = pd.read_csv(outdir / "analysis_panel.csv", index_col=0, parse_dates=True)
    idx = panel.index

    exp_short = ar1_expected_avg_short(short.reindex(idx).ffill())

    sig = {}  # per market signal frames
    for mkt, cc in MKT2CC.items():
        y = yds[cc].reindex(idx).ffill()
        y5, y10, y30 = y[f"{cc}_5y"], y[f"{cc}_10y"], y[f"{cc}_30y"]
        md = MODDUR[mkt]
        slope = local_slope_10y(y5, y10, y30)                 # %/yr
        sh = short.reindex(idx).ffill()
        carry_roll = ((y10 - sh) + md * slope) / 100.0 / 12.0  # monthly, decimal
        tp = (y10 - exp_short)                                  # term premium, %
        # term-premium scaling: full long when TP>0.5%, taper to 0.2 when TP<-0.5%
        tp_factor = ((tp + 0.5) / 1.0).clip(0.2, 1.0)
        carry_adj = carry_roll * tp_factor

        # trend: 6m (126d) return of the long-duration position, vol-scaled (causal)
        r = rets[f"{mkt}_ret"].reindex(idx)
        trend = r.rolling(126).sum() / (r.rolling(126).std() * np.sqrt(126))
        # value: y10 vs trailing 5y mean, vol-scaled (high yield = cheap = long ok)
        value = (y10 - y10.rolling(1260, min_periods=252).mean()) / \
                y10.rolling(1260, min_periods=252).std()

        vol = panel[f"{mkt}_rvol21"].shift(1).clip(lower=0.02)
        sig[mkt] = pd.DataFrame({
            "carry": scale_only(carry_adj), "trend": trend.shift(1),
            "value": value.shift(1), "vol": vol, "ret": r,
            # policy-tightening signal: 6m change in the 3m rate (+ = hiking).
            # The best crash-avoider found — but it LAGS the inflation shock.
            "short_mom6": short.reindex(idx).ffill().diff(126).shift(1),
        }, index=idx)
    return sig


def backtest(sig, position_fn, inst_vol=0.04, wcap=4.0, tc=2e-4):
    """position_fn(df)->per-market signal series; returns daily strategy return."""
    rets = pd.DataFrame({m: sig[m]["ret"] for m in sig})
    W = {}
    for m, df in sig.items():
        s = position_fn(df)
        w = (s * (inst_vol / df["vol"])).clip(-wcap, wcap)
        W[m] = w.where(df["ret"].notna())
    W = pd.DataFrame(W)
    pnl = pd.Series(0.0, index=W.index)
    for m in W.columns:
        pnl = pnl.add((W[m].shift(1) * rets[m]).fillna(0.0), fill_value=0.0)
    cost = tc * W.diff().abs().sum(axis=1).fillna(0.0)
    return pnl - cost


def metrics(r, name):
    r = r.dropna()
    ann = (1 + r).prod() ** (TD / len(r)) - 1
    vol = r.std() * np.sqrt(TD)
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1)
    w1m = (1 + r).rolling(21).apply(np.prod, raw=True).min() - 1
    return dict(strategy=name, ann_ret=ann * 100, ann_vol=vol * 100,
                sharpe=ann / vol if vol else np.nan, maxDD=dd.min() * 100,
                skew=r.skew(), worst1m=w1m * 100)


def dd_window(r, lo, hi):
    w = r.loc[lo:hi].dropna(); e = (1 + w).cumprod()
    return (e / e.cummax() - 1).min() * 100 if len(w) else np.nan


def run(outdir: Path):
    sig = build(outdir)

    naive = lambda df: df["carry"]
    trend_only = lambda df: df["trend"].clip(-3, 3)
    # trend/value gate (cuts DD but kills return — reactive de-risking)
    def gated_trend(df, lam_t=0.6, lam_v=0.4, g_min=-0.3):
        prot = (lam_t * df["trend"].clip(-3, 3) + lam_v * df["value"].clip(-3, 3))
        g = (0.5 + 0.5 * np.tanh(prot)).clip(0, 1)
        return df["carry"].clip(lower=0) * (g_min + (1 - g_min) * g)
    # MACRO gate: cut the long as the ECB tightens (short rate rising). The only
    # signal that lowers DD AND raises return — but it lags the inflation shock.
    def gated_macro(df, lo=0.0, hi=1.5):
        g = (1 - (df["short_mom6"] - lo) / (hi - lo)).clip(0, 1)
        return df["carry"].clip(lower=0) * g

    strat = {
        "carry_naive":  backtest(sig, naive),
        "trend_only":   backtest(sig, trend_only),
        "carry_gated_trend": backtest(sig, gated_trend),
        "carry_gated_macro": backtest(sig, gated_macro),
    }

    rows = [metrics(strat[k], k) for k in strat]
    tbl = pd.DataFrame(rows).set_index("strategy").round(2)
    print("\n" + "=" * 74)
    print(" Rebuilt carry — trend gate (fails) vs macro policy gate (works)")
    print("=" * 74)
    print(tbl.to_string())
    print("\n  Drawdown in key crash windows:")
    for k in strat:
        print(f"   {k:20s}  2022-24: {dd_window(strat[k],'2022-01-01','2024-12-31'):6.1f}%"
              f"   Feb-Apr26: {dd_window(strat[k],'2026-02-01','2026-04-30'):6.1f}%")
    out = pd.DataFrame(strat)
    out.to_csv(outdir / "defensive_carry_returns.csv")
    print(f"\n  wrote defensive_carry_returns.csv to {outdir}/")
    print("\n  NB: research simulation, not advice. The macro gate only PARTIALLY"
          " avoids 2022\n  because the realised policy rate lags the inflation"
          " shock — see the data ask.")
    return strat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-dir", default="outputs")
    run(Path(p.parse_args().outputs_dir))


if __name__ == "__main__":
    main()
