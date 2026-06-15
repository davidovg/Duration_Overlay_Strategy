"""
minvar_hedge.py
===============
Adapt Romano's minimum-variance hedge engine (the cov_vola_following /
MIN-VARIANCE family) to the EUR carry book, and use it to answer one question:
is the ceiling we keep hitting a METHOD problem or a BREADTH problem?

His objective, faithfully: choose hedge weights w on a basket of instruments to
minimise   mean over two windows of [ std(book + H·w) − ratio·worst_loss ]
i.e. volatility PLUS a downside-tail penalty (better than plain vol-targeting),
re-solved on a rolling basis with a gross cap.

We protect the carry book with a basket the optimiser is otherwise blind to —
GLOBAL duration (US/GB/CA/AU 10y, synthetic returns −D·Δy from the yields we
already pulled). If global rates carry a second factor, the hedge has something
real to exploit and should beat the EUR-only / vol-target ceiling. If not, the
problem is breadth and no amount of cleverness fixes it.

  carry_voltgt        : mechanical benchmark
  carry_mvhedge       : closed-form rolling min-variance hedge (global basket)
  carry_mvhedge_dn    : Romano's vol+downside-tail objective (global basket)

Research simulation, not investment advice.

USAGE
    python minvar_hedge.py --outputs-dir outputs
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
from longshort_duration import build_features, size, positions_to_returns, scale_only, TD

GLOBAL_DUR = 8.0   # approx modified duration of a 10y for synthetic returns


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


def breadth(rets, glob):
    def pc1(df):
        d = df.dropna()
        C = np.corrcoef(d.values.T); ev = np.linalg.eigvalsh(C)[::-1]
        return ev[0] / ev.sum(), (ev[0] + ev[1]) / ev.sum()
    eur = rets[["bund_ret", "oat_ret", "btp_ret"]]
    allm = pd.concat([eur, glob], axis=1)
    p1e, p2e = pc1(eur); p1a, p2a = pc1(allm)
    print("  BREADTH (variance explained by leading factors):")
    print(f"    EUR-only (3 instr):      PC1 {p1e:.0%}   PC1+PC2 {p2e:.0%}")
    print(f"    EUR+global (7 instr):    PC1 {p1a:.0%}   PC1+PC2 {p2a:.0%}")


def closed_form_hedge(book, H, lookback=252, gross_cap=3.0, rebal=21, lam=1e-6):
    """Rolling min-variance hedge: w = -Σ_hh^{-1} cov(h, book). Monthly rebalance."""
    idx = book.index
    W = pd.DataFrame(0.0, index=idx, columns=H.columns)
    cols = list(H.columns)
    last_w = np.zeros(len(cols))
    reb_points = range(lookback, len(idx), rebal)
    for i in reb_points:
        win = slice(i - lookback, i)
        Hw = H.iloc[win]; bw = book.iloc[win]
        ok = Hw.notna().all() & (Hw.std() > 0)
        use = [c for c in cols if ok.get(c, False)]
        if len(use) >= 2 and bw.notna().sum() > lookback * 0.8:
            Hu = Hw[use].fillna(0.0).values
            Sigma = np.cov(Hu.T) + lam * np.eye(len(use))
            cov_hp = np.array([np.cov(Hu[:, j], bw.fillna(0.0).values)[0, 1] for j in range(len(use))])
            w = -np.linalg.solve(Sigma, cov_hp)
            g = np.abs(w).sum()
            if g > gross_cap:
                w *= gross_cap / g
            wfull = np.zeros(len(cols))
            for j, c in enumerate(use):
                wfull[cols.index(c)] = w[j]
            last_w = wfull
        end = min(i + rebal, len(idx))
        W.iloc[i:end] = last_w
    return W


def downside_hedge(book, H, lookback=252, lb2=180, ratio=1.5, gross_cap=3.0, rebal=21):
    """Romano's objective: minimise mean over two windows of
    [std(book+H·w) − ratio·worst_loss], gross(w) ≤ cap. Monthly Nelder-Mead."""
    idx = book.index
    cols = list(H.columns)
    np.random.seed(0)  # the objective's random tie-break makes NM non-deterministic otherwise
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    last_w = np.zeros(len(cols))

    def obj(w, Hu, bv):
        if np.abs(w).sum() > gross_cap:
            return 1e4 * (1 + np.random.random())
        c = bv + Hu.dot(w)
        n1 = np.std(c[-lookback:]) - ratio * min(c[-lookback:].min(), 0)
        n2 = np.std(c[-lb2:]) - ratio * min(c[-lb2:].min(), 0)
        return (n1 + n2) / 2

    for i in range(lookback, len(idx), rebal):
        win = slice(i - lookback, i)
        Hw = H.iloc[win]; bw = book.iloc[win]
        ok = Hw.notna().all() & (Hw.std() > 0)
        use = [c for c in cols if ok.get(c, False)]
        if len(use) >= 2 and bw.notna().sum() > lookback * 0.8:
            Hu = Hw[use].fillna(0.0).values; bv = bw.fillna(0.0).values
            w0 = np.array([last_w[cols.index(c)] for c in use])
            if np.abs(w0).sum() == 0:
                w0 = -np.ones(len(use)) / len(use)
            res = minimize(obj, w0, args=(Hu, bv), method="Nelder-Mead",
                           options={"fatol": 1e-7, "xatol": 1e-6, "maxiter": 4000, "maxfev": 4000})
            wfull = np.zeros(len(cols))
            for j, c in enumerate(use):
                wfull[cols.index(c)] = res.x[j]
            last_w = wfull
        end = min(i + rebal, len(idx))
        W.iloc[i:end] = last_w
    return W


def apply_hedge(book, H, W, tc=2e-4):
    hedge_ret = (H.fillna(0.0) * W.shift(1)).sum(axis=1)
    cost = tc * W.diff().abs().sum(axis=1).fillna(0.0)
    return book + hedge_ret - cost


def run(outdir: Path):
    feat, fwd, raw, rets, idx = build_features(outdir)
    base = positions_to_returns(size({m: scale_only(raw[m]["carry"]) for m in raw}, raw), rets)

    # synthetic global duration returns from yields we already pulled
    yg = pd.read_csv(outdir / "yields_global.csv", index_col=0, parse_dates=True).reindex(idx).ffill()
    H = pd.DataFrame({c.replace("_10y", ""): -GLOBAL_DUR * yg[c].diff() / 100.0
                      for c in yg.columns}, index=idx)

    print("\n" + "=" * 80)
    print(" Min-variance hedge of the carry book — method vs breadth")
    print("=" * 80)
    breadth(rets, H)

    def vol_target(r, tgt=0.06, win=63, maxlev=2.0):
        rv = (r.rolling(win).std() * np.sqrt(TD)).shift(1)
        return r * (tgt / rv).clip(0, maxlev).fillna(0.0)

    print("\n  solving rolling hedges (monthly rebalance)…")
    Wcf = closed_form_hedge(base, H)
    Wdn = downside_hedge(base, H)

    strat = {
        "carry_naive":        base,
        "carry_voltgt":       vol_target(base),
        "carry_mvhedge":      apply_hedge(base, H, Wcf),
        "carry_mvhedge_dn":   apply_hedge(base, H, Wdn),
        # best-of-both: hedge the common global factor, then vol-target the residual
        "carry_mvhedge_vtgt": vol_target(apply_hedge(base, H, Wcf)),
    }
    order = list(strat)
    tbl = pd.DataFrame([metrics(strat[k], k) for k in order]).set_index("strategy").round(2)
    print("\n" + tbl.to_string())
    print("\n  Crash windows  (cumulative return, max drawdown):")
    for k in order:
        c22 = crash(strat[k], "2022-01-01", "2024-12-31")
        cmar = crash(strat[k], "2026-02-01", "2026-04-30")
        print(f"   {k:18s}  2022-24: {c22[0]:+6.1f}% (DD {c22[1]:5.1f}%)   "
              f"Feb-Apr26: {cmar[0]:+5.1f}% (DD {cmar[1]:5.1f}%)")

    avg_gross = Wcf.abs().sum(axis=1).replace(0, np.nan).mean()
    print(f"\n  avg hedge gross (closed-form): {avg_gross:.2f} units of global duration")
    out = pd.DataFrame(strat); out.to_csv(outdir / "minvar_hedge_returns.csv")
    print(f"  wrote minvar_hedge_returns.csv to {outdir}/  (research sim, not advice)")
    return strat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-dir", default="outputs")
    run(Path(p.parse_args().outputs_dir))


if __name__ == "__main__":
    main()
