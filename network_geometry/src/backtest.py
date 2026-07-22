"""
Weekly bund-futures duration overlay backtest.

Inputs (daily):
  - de10, de02 : German 10y / 2y yields (percent)
  - optional extra yield/asset series for the cross-asset network
  - FX gravity-radius factor, VIX

Synthetic bund futures weekly excess return (validated against actual FGBL
when available):
    r_t = -D * dy10_t + (y10_{t-1} - r_short_{t-1}) / 52
with D the modified duration of the CTD-equivalent (default 8.5).

Signals (computed on Friday, positions held the following week):
  MOM   : sign of 13-week change in y10 (yield down -> +1 long duration)
  CARRY : sign of (y10 - y2) slope z-score vs its own history (steep -> long)
  GR    : gravity-radius z-score (high FX turbulence -> flight-to-quality long)
  NET   : network regime (bund eigenvector centrality / density z-scores)

Combination: equal-risk average of active signals, optionally gated by regime.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

DUR = 8.5           # modified duration of 10y bund exposure
TC_BP = 1.0         # round-turn transaction cost, bp of notional per unit turnover
ANN = 52


def weekly_last(s):
    return s.resample("W-FRI").last()


def synthetic_bund_futures_weekly(y10: pd.Series, rshort: pd.Series, dur: float = DUR):
    """Weekly excess returns of a synthetic 10y bund futures position."""
    yw = weekly_last(y10.dropna()) / 100.0
    rw = weekly_last(rshort.reindex(y10.index).ffill().dropna()) / 100.0
    dy = yw.diff()
    carry = (yw.shift(1) - rw.shift(1)) / ANN
    r = -dur * dy + carry
    return r.dropna()


def perf_stats(returns: pd.Series, positions: pd.Series = None, freq: int = ANN):
    r = returns.dropna()
    mu, sd = r.mean() * freq, r.std() * np.sqrt(freq)
    sharpe = mu / sd if sd > 0 else np.nan
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    # Newey-West t-stat of the mean (lag 4)
    T = len(r)
    e = r - r.mean()
    g0 = (e @ e) / T
    nw = g0
    for l in range(1, 5):
        w = 1 - l / 5
        nw += 2 * w * (e[l:] @ e.shift(l).dropna()) / T
    tstat = r.mean() / np.sqrt(nw / T) if nw > 0 else np.nan
    out = {"AnnRet%": 100 * mu, "AnnVol%": 100 * sd, "Sharpe": sharpe,
           "MaxDD%": 100 * dd, "t(NW)": tstat, "N": T,
           "Skew": r.skew(), "HitRate%": 100 * (r > 0).mean()}
    if positions is not None:
        out["AvgTurnover"] = positions.diff().abs().mean()
        out["AvgPos"] = positions.mean()
    return pd.Series(out)


def run_overlay(signals: pd.DataFrame, fut_ret: pd.Series, tc_bp: float = TC_BP,
                clip: float = 1.0):
    """signals: weekly DataFrame of position signals in [-1, 1]; combined equally."""
    pos = signals.mean(axis=1).clip(-clip, clip)
    pos = pos.reindex(fut_ret.index).shift(1).fillna(0)     # trade with 1-week lag
    gross = pos * fut_ret
    costs = pos.diff().abs().fillna(0) * tc_bp / 1e4 * DUR / 8.5
    net = gross - costs
    return pos, net
