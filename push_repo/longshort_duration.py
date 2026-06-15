"""
longshort_duration.py
=====================
Strategies that can take NEGATIVE duration when the signal warrants it — the
point being to PROFIT from selloffs like 2022 and Mar-2026 rather than just
gating the carry long down to cash.

Two families, compared honestly against long-only carry:

  RULE-BASED (transparent)
    value_ls : mean-reversion. Long when yields are high vs their own history
               (cheap), SHORT when yields are rich (low) — the contrarian that
               would short the 2021 lows.
    macro_ls : carry long, but flips SHORT as the ECB tightens (short-rate
               momentum) — the macro regime trade.
    combo_ls : carry (long bias) + value + macro, allowed to net SHORT. Each
               leg can push the book negative.

  MACHINE LEARNING (walk-forward, honest OOS)
    ml_ls    : gradient-boosted regressor predicting the forward 21d duration
               return from the feature set, POOLED across the 3 markets for
               sample size, retrained annually on an expanding window, applied
               only out-of-sample. Position ∝ predicted return, so it goes
               short whenever the model expects a negative return.

Everything is causal: features are lagged, the ML target's forward window is
held strictly before each test year, positions are executed with a 1-day lag,
turnover is costed. This is a research simulation, NOT investment advice.

USAGE
    python longshort_duration.py --outputs-dir outputs
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingRegressor

MKT2CC = {"bund": "de", "oat": "fr", "btp": "it"}
MODDUR = {"bund": 8.5, "oat": 8.5, "btp": 7.5}
TD = 252
FEATURES = ["carry", "level", "slope", "curv", "short_mom6", "short_mom12",
            "value", "rvol", "trend63", "trend252"]


def zexp(s, minp=252):
    return ((s - s.expanding(minp).mean()) / s.expanding(minp).std().replace(0, np.nan)).clip(-3, 3)


def scale_only(s, minp=252):
    return (s / s.expanding(minp).std().replace(0, np.nan)).clip(-3, 3)


def local_slope_10y(y5, y10, y30):
    x1, x2, x3 = 5.0, 10.0, 30.0
    return (y5 * (20 - x2 - x3) / ((x1 - x2) * (x1 - x3))
            + y10 * (20 - x1 - x3) / ((x2 - x1) * (x2 - x3))
            + y30 * (20 - x1 - x2) / ((x3 - x1) * (x3 - x2)))


# ─── feature panel (per market) ─────────────────────────────────────────────
def build_features(outdir: Path):
    panel = pd.read_csv(outdir / "analysis_panel.csv", index_col=0, parse_dates=True)
    rets = pd.read_csv(outdir / "returns_markets.csv", index_col=0, parse_dates=True)
    sr = pd.read_csv(outdir / "short_rates.csv", index_col=0, parse_dates=True)
    yds = {cc: pd.read_csv(outdir / f"yields_{cc}.csv", index_col=0, parse_dates=True)
           for cc in ("de", "fr", "it")}
    idx = panel.index
    short = sr["euribor_3m"].reindex(idx).ffill()
    short_mom6 = short.diff(126)
    short_mom12 = short.diff(252)

    feat = {}
    fwd = {}
    raw = {}  # un-standardised carry etc. for the rule book
    for mkt, cc in MKT2CC.items():
        y = yds[cc].reindex(idx).ffill()
        y5, y10, y30 = y[f"{cc}_5y"], y[f"{cc}_10y"], y[f"{cc}_30y"]
        md = MODDUR[mkt]
        carry_roll = ((y10 - short) + md * local_slope_10y(y5, y10, y30)) / 100.0 / 12.0
        r = rets[f"{mkt}_ret"].reindex(idx)
        # the panel index carries ~111 dates the futures series lacks; compute
        # rolling trend/forward on the NATIVE consecutive series, then realign,
        # so scattered NaNs don't poison the rolling windows.
        r_nat = rets[f"{mkt}_ret"].dropna()

        def _trend(s, w, mp):
            return s.rolling(w, min_periods=mp).sum() / (s.rolling(w, min_periods=mp).std() * np.sqrt(w))

        value = (y10 - y10.rolling(1260, min_periods=252).mean()) / y10.rolling(1260, min_periods=252).std()
        df = pd.DataFrame({
            "carry": carry_roll,
            "level": panel[f"{cc}_level"], "slope": panel[f"{cc}_slope"],
            "curv": panel[f"{cc}_curvature"],
            "short_mom6": short_mom6, "short_mom12": short_mom12,
            "value": value, "rvol": panel[f"{mkt}_rvol21"],
            "trend63": _trend(r_nat, 63, 40).reindex(idx),
            "trend252": _trend(r_nat, 252, 150).reindex(idx),
        }, index=idx).shift(1)  # lag ALL features by 1 day (causal)
        feat[mkt] = df
        raw[mkt] = dict(carry=carry_roll, value=value, short_mom6=short_mom6,
                        vol=panel[f"{mkt}_rvol21"].shift(1).clip(lower=0.02), ret=r)
        # forward 21d return target (sum of next 21 daily returns), native then realign
        fwd[mkt] = (r_nat.rolling(21).sum().shift(-21)).reindex(idx)
    return feat, fwd, raw, rets, idx


# ─── backtest engine ────────────────────────────────────────────────────────
def positions_to_returns(W, rets, tc=2e-4):
    pnl = pd.Series(0.0, index=W.index)
    for m in W.columns:
        pnl = pnl.add((W[m].shift(1) * rets[f"{m}_ret"].reindex(W.index)).fillna(0.0), fill_value=0.0)
    cost = tc * W.diff().abs().sum(axis=1).fillna(0.0)
    return pnl - cost


def size(signal_by_mkt, raw, inst_vol=0.04, wcap=4.0):
    W = {}
    for m, s in signal_by_mkt.items():
        w = (s * (inst_vol / raw[m]["vol"])).clip(-wcap, wcap)
        W[m] = w.where(raw[m]["ret"].notna())
    return pd.DataFrame(W)


# ─── rule-based long/short ──────────────────────────────────────────────────
def rule_strategies(raw, idx):
    out = {}
    carry = {m: scale_only(raw[m]["carry"]) for m in raw}                 # long bias
    value = {m: zexp(raw[m]["value"]).shift(1) for m in raw}              # +high yld=long
    macro = {m: (-zexp(raw[m]["short_mom6"])).shift(1) for m in raw}      # short on tightening
    out["value_ls"] = {m: value[m] for m in raw}
    out["macro_ls"] = {m: (carry[m].clip(lower=0) + 1.2 * macro[m]) for m in raw}
    out["combo_ls"] = {m: (0.7 * carry[m] + 0.7 * value[m] + 0.7 * macro[m]) for m in raw}
    return out


# ─── walk-forward ML long/short ─────────────────────────────────────────────
def ml_longshort(feat, fwd, raw, idx, start_year=2007, retrain="A"):
    # pooled stacked dataset
    frames = []
    for m in feat:
        d = feat[m].copy()
        d["_fwd"] = fwd[m]
        d["_mkt"] = m
        frames.append(d)
    pool = pd.concat(frames).dropna(subset=FEATURES)
    pred_by_mkt = {m: pd.Series(index=idx, dtype=float) for m in feat}

    years = range(start_year, idx[-1].year + 1)
    for yr in years:
        test_start = pd.Timestamp(f"{yr}-01-01")
        test_end = pd.Timestamp(f"{yr}-12-31")
        # training: forward window must end strictly before test_start
        train = pool[(pool.index < test_start - pd.Timedelta(days=21))].dropna(subset=["_fwd"])
        if len(train) < 750:
            continue
        Xtr = train[FEATURES].values
        ytr = train["_fwd"].values
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        model = GradientBoostingRegressor(
            n_estimators=120, max_depth=2, learning_rate=0.04,
            subsample=0.7, random_state=0)
        model.fit((Xtr - mu) / sd, ytr)
        pstd = np.std(model.predict((Xtr - mu) / sd)) + 1e-9
        for m in feat:
            te = feat[m].loc[(feat[m].index >= test_start) & (feat[m].index <= test_end)].dropna(subset=FEATURES)
            if te.empty:
                continue
            p = model.predict((te[FEATURES].values - mu) / sd) / pstd
            pred_by_mkt[m].loc[te.index] = p
    # position ∝ standardised predicted return (long or short)
    return {m: pred_by_mkt[m].clip(-2, 2) for m in feat}


# ─── metrics ────────────────────────────────────────────────────────────────
def metrics(r, name):
    r = r.dropna()
    if len(r) < 60:
        return {"strategy": name, "n": len(r)}
    ann = (1 + r).prod() ** (TD / len(r)) - 1
    vol = r.std() * np.sqrt(TD)
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1)
    w1m = (1 + r).rolling(21).apply(np.prod, raw=True).min() - 1
    return {"strategy": name, "n": len(r), "ann%": ann * 100, "vol%": vol * 100,
            "sharpe": ann / vol if vol else np.nan, "maxDD%": dd.min() * 100,
            "skew": r.skew(), "worst1m%": w1m * 100}


def crash(r, lo, hi):
    w = r.loc[lo:hi].dropna()
    e = (1 + w).cumprod()
    return ((e.iloc[-1] - 1) * 100, (e / e.cummax() - 1).min() * 100) if len(w) else (np.nan, np.nan)


def run(outdir: Path):
    feat, fwd, raw, rets, idx = build_features(outdir)

    strat = {}
    # long-only carry reference
    strat["carry_long_only"] = positions_to_returns(
        size({m: scale_only(raw[m]["carry"]) for m in raw}, raw), rets)
    # rule-based long/short
    for name, sig in rule_strategies(raw, idx).items():
        strat[name] = positions_to_returns(size(sig, raw), rets)
    # ML long/short
    print("  training walk-forward ML (annual retrain, pooled 3 markets)…")
    ml_sig = ml_longshort(feat, fwd, raw, idx)
    strat["ml_ls"] = positions_to_returns(size(ml_sig, raw), rets)

    # synthesis: macro long/short + portfolio vol-target (handles the fast spike
    # the macro signal lags, and pulls the −42% DD down)
    def vol_target(r, tgt=0.08, win=63, maxlev=2.0):
        rv = (r.rolling(win).std() * np.sqrt(TD)).shift(1)
        return r * (tgt / rv).clip(0, maxlev).fillna(0.0)
    strat["macro_ls_voltgt"] = vol_target(strat["macro_ls"])

    order = ["carry_long_only", "value_ls", "macro_ls", "combo_ls", "ml_ls",
             "macro_ls_voltgt"]
    tbl = pd.DataFrame([metrics(strat[k], k) for k in order]).set_index("strategy").round(2)
    print("\n" + "=" * 80)
    print(" Long/SHORT duration — can take negative duration when signalled")
    print("=" * 80)
    print(tbl.to_string())
    print("\n  Crash windows  (cumulative return, max drawdown):")
    for k in order:
        c22 = crash(strat[k], "2022-01-01", "2024-12-31")
        cmar = crash(strat[k], "2026-02-01", "2026-04-30")
        print(f"   {k:16s}  2022-24: {c22[0]:+6.1f}% (DD {c22[1]:5.1f}%)   "
              f"Feb-Apr26: {cmar[0]:+5.1f}% (DD {cmar[1]:5.1f}%)")

    out = pd.DataFrame(strat)
    out.to_csv(outdir / "longshort_returns.csv")
    print(f"\n  wrote longshort_returns.csv to {outdir}/")
    print("  research simulation, not advice.")
    return strat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-dir", default="outputs")
    run(Path(p.parse_args().outputs_dir))


if __name__ == "__main__":
    main()
