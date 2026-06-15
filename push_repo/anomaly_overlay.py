"""
anomaly_overlay.py
=================
Adapt the firm's anomaly-detection overlay (Romano's Volatility-Strategy
scripts) to the EUR duration book.

THE IDEA (why it's better than predicting returns)
--------------------------------------------------
Returns aren't predictable here (the regression ML went flat). But CRASHES are
a risk-regime question, and risk clusters. So instead of predicting the carry
return, we predict P(anomaly) — the probability the next day is a crash day for
carry (return below the training-window 10th percentile) — with a classifier,
and scale exposure by it:  weight = 1 − P(anomaly)   (de-risk), or
                            weight = 1 − 2·P(anomaly) (allow a SHORT when crash
                                                       probability is high).

Faithful to the source where it matters, fixed where it was unsafe:
  * WALK-FORWARD (their ROLLING variant), retrained annually, strictly OOS.
  * anomaly threshold computed on the TRAINING window only (no look-ahead).
  * class imbalance handled with balanced sample weights (crashes are ~10%).
  * features lagged, weights applied with a 1-day execution lag.
  * sane GB regularisation (NOT n_estimators=100000).

Compared against the mechanical vol-target, the honest question: does a learned
crash classifier beat simply de-grossing when realised vol is high?

Research simulation, not investment advice.

USAGE
    python anomaly_overlay.py --outputs-dir outputs
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingClassifier

from longshort_duration import (build_features, size, positions_to_returns,
                                 scale_only, MKT2CC, TD)

ANOM_FEATURES = ["carry", "level", "slope", "curv", "short_mom6", "short_mom12",
                 "value", "rvol", "trend63", "trend252", "strat_ret21", "strat_vol21"]


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


def portfolio_features(feat, base_ret):
    """Aggregate per-market features to portfolio level + add the carry book's
    own trailing return and vol (its drawdown/vol state)."""
    mkts = list(feat.keys())
    agg = {}
    for col in ["carry", "level", "slope", "curv", "short_mom6", "short_mom12",
                "value", "rvol", "trend63", "trend252"]:
        agg[col] = pd.concat([feat[m][col] for m in mkts], axis=1).mean(axis=1)
    F = pd.DataFrame(agg)
    F["strat_ret21"] = base_ret.rolling(21).sum().shift(1)
    F["strat_vol21"] = (base_ret.rolling(21).std() * np.sqrt(TD)).shift(1)
    return F


def walk_forward_anomaly(F, base_ret, pct=10, start_year=2007, balanced=True):
    """Annual-retrain GB classifier → OOS P(anomaly). Anomaly = base return
    below the `pct`-th percentile of the training window."""
    idx = F.index
    P = pd.Series(index=idx, dtype=float)
    data = F.dropna()
    for yr in range(start_year, idx[-1].year + 1):
        ts = pd.Timestamp(f"{yr}-01-01"); te = pd.Timestamp(f"{yr}-12-31")
        tr_mask = data.index < ts
        if tr_mask.sum() < 750:
            continue
        Xtr = data[tr_mask]
        thr = base_ret.reindex(Xtr.index).quantile(pct / 100.0)
        ytr = (base_ret.reindex(Xtr.index) <= thr).astype(int).values
        if ytr.sum() < 10:
            continue
        # balanced sample weights sharpen sensitivity but inflate P (over-derisk);
        # raw weights keep P calibrated to the true ~10% base rate.
        if balanced:
            w = np.where(ytr == 1, (len(ytr) - ytr.sum()) / max(ytr.sum(), 1), 1.0)
        else:
            w = None
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=2,
                                         learning_rate=0.03, subsample=0.7,
                                         random_state=0)
        clf.fit(Xtr.values, ytr, sample_weight=w)
        te_mask = (data.index >= ts) & (data.index <= te)
        if te_mask.any():
            P.loc[data.index[te_mask]] = clf.predict_proba(data[te_mask].values)[:, 1]
    return P.clip(0, 1)


def run(outdir: Path):
    feat, fwd, raw, rets, idx = build_features(outdir)

    # base = long-only carry
    carry_sig = {m: scale_only(raw[m]["carry"]) for m in raw}
    carry_pos = size(carry_sig, raw)
    base = positions_to_returns(carry_pos, rets)

    # reference: mechanical portfolio vol-target
    def vol_target(r, tgt=0.06, win=63, maxlev=2.0):
        rv = (r.rolling(win).std() * np.sqrt(TD)).shift(1)
        return r * (tgt / rv).clip(0, maxlev).fillna(0.0)

    # anomaly overlay (walk-forward GB crash classifier)
    print("  training walk-forward anomaly classifier (annual retrain)…")
    F = portfolio_features(feat, base)
    Pb = walk_forward_anomaly(F, base, balanced=True)    # sharp but trigger-happy
    Pr = walk_forward_anomaly(F, base, balanced=False)   # calibrated to ~10% base rate
    Pb_l, Pr_l = Pb.shift(1).reindex(base.index), Pr.shift(1).reindex(base.index)

    # high-confidence de-risk: act only when the (balanced) signal is strong
    hi = (1 - ((Pb_l - 0.5) / 0.5).clip(0, 1)).fillna(1.0)

    strat = {
        "carry_naive":        base,
        "carry_voltgt":       vol_target(base),                                  # mechanical benchmark
        "carry_anom_derisk":  base * (1 - Pr_l).clip(0, 1).fillna(1.0),          # calibrated de-risk
        "carry_anom_hiconf":  base * hi,                                         # only act on strong signal
        "carry_anom_ls":      base * (1 - 2 * Pb_l).clip(-1, 1).fillna(1.0),     # can flip short
        "carry_anom_voltgt":  vol_target(base * (1 - Pr_l).clip(0, 1).fillna(1.0)),  # de-risk + vol-target
    }
    P = Pb  # for the de-risk-frequency printout

    order = list(strat)
    tbl = pd.DataFrame([metrics(strat[k], k) for k in order]).set_index("strategy").round(2)
    print("\n" + "=" * 80)
    print(" Anomaly-detection overlay vs mechanical vol-target (carry book)")
    print("=" * 80)
    print(tbl.to_string())
    print("\n  Crash windows  (cumulative return, max drawdown):")
    for k in order:
        c22 = crash(strat[k], "2022-01-01", "2024-12-31")
        cmar = crash(strat[k], "2026-02-01", "2026-04-30")
        print(f"   {k:20s}  2022-24: {c22[0]:+6.1f}% (DD {c22[1]:5.1f}%)   "
              f"Feb-Apr26: {cmar[0]:+5.1f}% (DD {cmar[1]:5.1f}%)")

    # how often is the overlay actually de-risking?
    print(f"\n  mean P(anomaly) OOS: {P.mean():.2f}   "
          f"days de-risked >25%: {(P > 0.25).mean():.0%}   "
          f">50% (short in _ls): {(P > 0.5).mean():.0%}")
    out = pd.DataFrame(strat); out["P_anomaly"] = P
    out.to_csv(outdir / "anomaly_overlay_returns.csv")
    print(f"\n  wrote anomaly_overlay_returns.csv to {outdir}/  (research sim, not advice)")
    return strat, P


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-dir", default="outputs")
    run(Path(p.parse_args().outputs_dir))


if __name__ == "__main__":
    main()
