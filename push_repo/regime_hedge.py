"""
regime_hedge.py
===============
Make the global min-variance carry hedge regime-aware.

The min-variance hedge from minvar_hedge.py is ALWAYS ON — it shorts global
duration every day. That's wrong in a flight-to-quality risk-off (2008, 2020),
where bonds RALLY and the carry book wants to be long: the always-on short
fights the book's own gains. It's only right in the inflation/tightening
selloff (2022) where stocks AND bonds fall together.

Equity markets tell you which regime you're in. The danger regime for a carry
book is NOT risk-off per se — it's POSITIVE stock-bond correlation + equity
stress (no safe-haven bid). Flight-to-quality is NEGATIVE stock-bond corr +
equity stress (bonds rally → hedge off).

  >>> We don't yet have equity index / equity-vol series staged (only Brent &
      EURUSD). So this demonstrates the MECHANISM with the proxies we have —
      oil momentum is the inflation axis that separates the two risk-offs
      (2022 = oil spike → hedge on; 2008/2020 = oil collapse → hedge off) —
      and exposes a clean hook (`danger_from_signals`) where real equity-vol +
      stock-bond-correlation slot in for the superior version. RICs to pull are
      listed at the bottom.

  carry_mvhedge          : always-on global hedge (baseline to beat)
  carry_mvhedge_regime   : hedge scaled by the RORO/inflation regime
  *_voltgt               : + vol-target on the residual

Research simulation, not investment advice.

USAGE
    python regime_hedge.py --outputs-dir outputs
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from longshort_duration import build_features, size, positions_to_returns, scale_only, TD
from minvar_hedge import closed_form_hedge, GLOBAL_DUR

TD_ = TD


def zexp(s, minp=252):
    return ((s - s.expanding(minp).mean()) / s.expanding(minp).std().replace(0, np.nan)).clip(-3, 3)


def metrics(r, name):
    r = r.dropna()
    if len(r) < 60:
        return {"strategy": name, "n": len(r)}
    ann = (1 + r).prod() ** (TD / len(r)) - 1
    vol = r.std() * np.sqrt(TD)
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1)
    w1m = (1 + r).rolling(21).apply(np.prod, raw=True).min() - 1
    return {"strategy": name, "ann%": ann * 100, "vol%": vol * 100,
            "sharpe": ann / vol if vol else np.nan, "maxDD%": dd.min() * 100,
            "skew": r.skew(), "worst1m%": w1m * 100}


def crash(r, lo, hi):
    w = r.loc[lo:hi].dropna(); e = (1 + w).cumprod()
    return ((e.iloc[-1] - 1) * 100, (e / e.cummax() - 1).min() * 100) if len(w) else (np.nan, np.nan)


def danger_from_signals(risk: pd.DataFrame, idx, bond_ret: pd.Series | None = None):
    """Map cross-asset signals → a hedge-intensity multiplier in [0, 2].

    PROXY inputs available now: oil (inflation axis) + USD (tightening axis).
      multiplier ≈ 0  when oil collapsing / deflationary  → flight-to-quality,
                                                            hedge OFF (bonds rally)
      multiplier ≈ 1  normal
      multiplier ≈ 2  oil spiking / USD surging           → inflation selloff,
                                                            hedge ON hard (2022)

    REAL equity version (when staged) would replace `z` with, e.g.,
      z = 0.5*zexp(equity_vol) + 0.5*sign(stock_bond_corr)*zexp(equity_drawdown)
    so the hedge ramps up only when bonds have lost their safe-haven bid.
    """
    oil = np.log(risk["brent"].reindex(idx).ffill())
    usd = np.log(risk["eurusd"].reindex(idx).ffill())
    oil_mom = oil.diff(63)
    usd_str = -usd.diff(63)          # EURUSD down ⇒ USD up ⇒ tightening/risk-off
    z = 0.7 * zexp(oil_mom) + 0.3 * zexp(usd_str)
    mult = (1.0 + 0.6 * z).clip(0.0, 2.0)
    return mult.shift(1)             # causal


def apply_scaled_hedge(book, H, W, mult, tc=2e-4):
    Ws = W.mul(mult, axis=0)
    hedge_ret = (H.fillna(0.0) * Ws.shift(1)).sum(axis=1)
    cost = tc * Ws.diff().abs().sum(axis=1).fillna(0.0)
    return book + hedge_ret - cost


def vol_target(r, tgt=0.06, win=63, maxlev=2.0):
    rv = (r.rolling(win).std() * np.sqrt(TD)).shift(1)
    return r * (tgt / rv).clip(0, maxlev).fillna(0.0)


def run(outdir: Path):
    feat, fwd, raw, rets, idx = build_features(outdir)
    base = positions_to_returns(size({m: scale_only(raw[m]["carry"]) for m in raw}, raw), rets)

    yg = pd.read_csv(outdir / "yields_global.csv", index_col=0, parse_dates=True).reindex(idx).ffill()
    H = pd.DataFrame({c.replace("_10y", ""): -GLOBAL_DUR * yg[c].diff() / 100.0
                      for c in yg.columns}, index=idx)
    risk = pd.read_csv(outdir / "risk_proxies.csv", index_col=0, parse_dates=True)

    W = closed_form_hedge(base, H)
    ones = pd.Series(1.0, index=idx)
    mult = danger_from_signals(risk, idx, bond_ret=rets["bund_ret"])

    strat = {
        "carry_naive":            base,
        "carry_voltgt":           vol_target(base),
        "carry_mvhedge":          apply_scaled_hedge(base, H, W, ones),    # always-on
        "carry_mvhedge_regime":   apply_scaled_hedge(base, H, W, mult),   # RORO-conditioned
        "carry_mvhedge_rgm_vtgt": vol_target(apply_scaled_hedge(base, H, W, mult)),
    }
    order = list(strat)
    print("\n" + "=" * 82)
    print(" Regime-conditioned global hedge  (oil/USD proxy for risk-on/risk-off)")
    print("=" * 82)
    tbl = pd.DataFrame([metrics(strat[k], k) for k in order]).set_index("strategy").round(2)
    print(tbl.to_string())
    print("\n  Crash / flight-to-quality windows  (cumulative return, max drawdown):")
    wins = {"2008 GFC (FTQ)": ("2008-01-01", "2008-12-31"),
            "2020 COVID (FTQ)": ("2020-02-01", "2020-05-31"),
            "2022-24 inflation": ("2022-01-01", "2024-12-31"),
            "Feb-Apr26 spike": ("2026-02-01", "2026-04-30")}
    for k in order:
        cells = "  ".join(f"{nm}: {crash(strat[k], lo, hi)[0]:+5.1f}%" for nm, (lo, hi) in wins.items())
        print(f"   {k:22s} {cells}")

    print(f"\n  regime multiplier: mean {mult.mean():.2f}, "
          f"% time hedge≈off (<0.25): {(mult < 0.25).mean():.0%}, "
          f"% time hedge boosted (>1.5): {(mult > 1.5).mean():.0%}")
    pd.DataFrame(strat).to_csv(outdir / "regime_hedge_returns.csv")
    print(f"  wrote regime_hedge_returns.csv to {outdir}/  (research sim, not advice)")
    return strat, mult


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-dir", default="outputs")
    run(Path(p.parse_args().outputs_dir))


if __name__ == "__main__":
    main()

# ── To upgrade from oil/USD proxy to a true equity RORO signal, pull from
#    Refinitiv and stage as risk_proxies columns, then swap into
#    danger_from_signals():
#      .STOXX50E / .SPX            equity level   → drawdown & momentum
#      .V2TX  (VSTOXX) / .VIX      equity vol     → stress level
#    and compute a rolling stock-bond correlation corr(equity_ret, bund_ret):
#      danger HIGH  when corr > 0 AND equity stressed   (2022, no haven bid)
#      danger LOW   when corr < 0 AND equity stressed   (FTQ, bonds rally)
