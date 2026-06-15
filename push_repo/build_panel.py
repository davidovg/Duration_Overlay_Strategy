"""
build_panel.py
==============
Join the per-instrument CSVs from download_rates.py into the master analysis
panel for the EUR duration overlay (Bund / OAT / BTP).

DESIGN PRINCIPLE — two return spaces, one signal space
------------------------------------------------------
  * EXECUTION / P&L truth  = roll-adjusted FUTURES returns. Futures prices
    already embed carry + roll via the basis/CTD, so the strategy P&L is
    simply  Σ_i  active_position_i · futures_return_i. This is the backbone.
  * SIGNAL generation      = the cash YIELD CURVE (clean, roll-free). Carry,
    Mean-Reversion and Momentum are all derived here.
  * We also compute a yield-implied return purely for ATTRIBUTION (the per-
    country contribution decomposition, à la deck p.11/13), never for P&L.

LOOK-AHEAD IS THE WHOLE GAME. The three traps, and how we avoid them:
  1. Curve PCA   -> fit loadings on an EXPANDING window, refit monthly, apply
                    only to the following month. Day t uses loadings estimated
                    strictly before t. Signs/order re-aligned each refit.
  2. Mean-Rev    -> AR(1) on the factors, same expanding/monthly cadence.
  3. Roll        -> continuous returns are stitched WITHIN each contract using
                    the contract overlap, so no synthetic price-gap return ever
                    enters the series.

CDS HOOK
--------
sovereign CDS (IT/FR) maps to the BTP-Bund / OAT-Bund SPREAD leg, and the
CDS-bond basis is itself a signal. If outputs/cds_sovereign.csv exists (cols
it_5y, fr_5y, de_5y CDS in bp) it is merged and the cash-vs-CDS basis computed.
Broad credit indices (iTraxx) come in via outputs/credit_indices.csv as REGIME
features only — co-movers, not yield drivers.

INPUT  (outputs/ from download_rates.py)
    yields_de.csv yields_fr.csv yields_it.csv  yields_global.csv
    fut_bund.csv ... fut_btp.csv                fut_chain_<root>.csv
    short_rates.csv  risk_proxies.csv
    [optional] cds_sovereign.csv  credit_indices.csv

OUTPUT
    outputs/analysis_panel.csv        date × all features
    outputs/returns_markets.csv       date × {bund,oat,btp} roll-adj returns

USAGE
    python build_panel.py
    python build_panel.py --outputs-dir outputs --roll-offset 5 --pca-burnin 504
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# market -> (chain root, continuation file, cont col, 10Y cash-yield col, modified duration)
# Mod. duration of the CTD basket is ~ the future's; ~8-9y for the 10Y contracts.
MARKETS = {
    "bund": dict(root="FGBL", cont="fut_bund.csv", cont_col="bund_c1",
                 y10="de_10y", yfile="yields_de.csv", mod_dur=8.5),
    "oat":  dict(root="FOAT", cont="fut_oat.csv",  cont_col="oat_c1",
                 y10="fr_10y", yfile="yields_fr.csv", mod_dur=8.5),
    "btp":  dict(root="FBTP", cont="fut_btp.csv",  cont_col="btp_c1",
                 y10="it_10y", yfile="yields_it.csv", mod_dur=7.5),
}
CURVE_TENORS = {"de": ["de_2y", "de_5y", "de_10y", "de_30y"],
                "fr": ["fr_2y", "fr_5y", "fr_10y", "fr_30y"],
                "it": ["it_2y", "it_5y", "it_10y", "it_30y"]}
TENOR_YEARS = np.array([2.0, 5.0, 10.0, 30.0])

BDAY = pd.tseries.offsets.BusinessDay()


# ─── IO ─────────────────────────────────────────────────────────────────────
def load_wide(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  · missing {path.name} (skipping)")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ─── Roll-adjusted continuous futures returns ───────────────────────────────
def _delivery_date(year: int, month: int) -> pd.Timestamp:
    """Eurex fixed-income delivery = 10th calendar day, rolled to next BDay."""
    d = pd.Timestamp(year=year, month=month, day=10)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def continuous_returns(chain: Optional[pd.DataFrame],
                       cont: Optional[pd.DataFrame],
                       cont_col: str,
                       roll_offset: int,
                       roll_dates: Optional[pd.Series] = None) -> pd.DataFrame:
    """Return DataFrame indexed by date with columns ['ret', 'adj_price'].

    Preferred path: stitch the quarterly chain, rolling `roll_offset` business
    days before delivery. The return on the roll day is the NEW contract's
    own return (valid because contracts overlap), so no price-gap return leaks
    in.

    Fallback (no chain — e.g. dated RICs absent on the feed): use the c1
    continuation, but CLEAN the roll-day basis jump. On each roll window the
    front contract switches and c1 prints an artificial jump while c2 (one
    contract further out) does not; we locate the jump day as the max |r1-r2|
    inside the window and replace c1's return there with c2's (the clean
    new-front return). This removes the ~4 spurious jumps/year without needing
    the dated contracts.
    """
    if chain is not None and not chain.empty:
        chain = chain.copy()
        chain["date"] = pd.to_datetime(chain["date"])
        # delivery + roll date per contract
        meta = (chain[["ric", "year", "month"]].drop_duplicates()
                .reset_index(drop=True))
        meta["delivery"] = [
            _delivery_date(int(y), int(m)) for y, m in zip(meta.year, meta.month)
        ]
        meta["roll"] = meta["delivery"] - roll_offset * BDAY
        meta = meta.sort_values("delivery").reset_index(drop=True)

        # wide close per contract
        wide = chain.pivot_table(index="date", columns="ric", values="close",
                                 aggfunc="last").sort_index()
        cret = wide.pct_change()

        # active contract per date: nearest contract whose roll date is still ahead
        rolls = meta[["ric", "roll", "delivery"]].sort_values("roll")
        dates = wide.index
        active = pd.Series(index=dates, dtype="object")
        ric_list = rolls["ric"].tolist()
        roll_list = rolls["roll"].tolist()
        for d in dates:
            chosen = None
            for ric, r in zip(ric_list, roll_list):
                if r > d and ric in wide.columns:
                    chosen = ric
                    break
            active[d] = chosen
        # daily return = active contract's own return that day
        ret = pd.Series(index=dates, dtype=float)
        for d in dates:
            a = active[d]
            if a is not None and a in cret.columns:
                v = cret.at[d, a]
                ret[d] = v if pd.notna(v) else np.nan
        ret = ret.dropna()
        chain_ret = ret if len(ret) > 50 else None
    else:
        chain_ret = None

    # continuation (cleaned) candidate
    cont_ret = None
    cleaned = 0
    if cont is not None and cont_col in cont.columns:
        c1 = cont[cont_col].dropna()
        r1 = c1.pct_change()
        c2_col = cont_col.replace("_c1", "_c2")
        if c2_col in cont.columns and roll_dates is not None:
            c2 = cont[c2_col].reindex(c1.index)
            r2 = c2.pct_change()
            rw = roll_dates.reindex(c1.index).fillna(False).astype(bool)
            if rw.any():
                diff = (r1 - r2).abs()
                block = (rw != rw.shift()).cumsum()
                for _, sub in rw.groupby(block):
                    if not bool(sub.iloc[0]):       # non-roll block
                        continue
                    win = sub.index
                    d = diff.reindex(win).idxmax()  # the actual front-switch day
                    if pd.notna(d) and pd.notna(r2.get(d, np.nan)):
                        r1.loc[d] = r2.loc[d]
                        cleaned += 1
        cont_ret = r1.dropna()

    # CHOOSE: use the chain ONLY if it actually covers the history. On feeds
    # that carry only the currently-live dated contracts, the chain spans ~1yr
    # while the continuation spans decades — in that case use the continuation.
    use_chain = (chain_ret is not None and
                 (cont_ret is None or len(chain_ret) >= 0.8 * len(cont_ret)))
    if use_chain:
        ret = chain_ret
        print(f"      roll-adjusted from chain: {len(ret)} obs, "
              f"{ret.index[0].date()} → {ret.index[-1].date()}")
    elif cont_ret is not None:
        if chain_ret is not None:
            print(f"      chain covers only {len(chain_ret)} days vs "
                  f"{len(cont_ret)} continuation — using cleaned continuation")
        ret = cont_ret
        print(f"      continuation ({cleaned} roll jumps cleaned via c2): "
              f"{len(ret)} obs, {ret.index[0].date()} → {ret.index[-1].date()}")
    else:
        return pd.DataFrame(columns=["ret", "adj_price"])

    adj = (1.0 + ret).cumprod() * 100.0
    return pd.concat({"ret": ret, "adj_price": adj}, axis=1)


# ─── Look-ahead-safe curve PCA (level / slope / curvature) ──────────────────
def _align_pc(loadings: np.ndarray, scores_proxy_corr) -> np.ndarray:
    return loadings  # placeholder; sign handled in rolling_curve_pca


def rolling_curve_pca(yields: pd.DataFrame, tenors: list[str],
                      burnin: int, refit_freq: str = "ME") -> pd.DataFrame:
    """Expanding-window PCA, refit monthly, applied only to the FOLLOWING
    period. Returns DataFrame [level, slope, curvature] of projected scores.

    Sign/order convention re-imposed each refit (PCA components can flip):
        level     ~ +mean across tenors
        slope     ~ +(long - short)
        curvature ~ +(2*mid - short - long)
    """
    Y = yields[tenors].dropna(how="any")
    if len(Y) < burnin + 21:
        return pd.DataFrame(index=yields.index,
                            columns=["level", "slope", "curvature"], dtype=float)

    refit_dates = Y.iloc[burnin:].resample(refit_freq).last().index
    proxies = {
        "level": Y.mean(axis=1),
        "slope": Y[tenors[-1]] - Y[tenors[0]],
        "curvature": 2 * Y[tenors[len(tenors) // 2]] - Y[tenors[0]] - Y[tenors[-1]],
    }

    out = pd.DataFrame(index=Y.index, columns=["level", "slope", "curvature"], dtype=float)
    # for each refit, fit on data UP TO that date, apply to next refit window
    refit_dates = [d for d in refit_dates if d in Y.index or True]
    bounds = list(refit_dates) + [Y.index[-1] + pd.Timedelta(days=1)]
    for i, refit_d in enumerate(refit_dates):
        train = Y.loc[:refit_d]
        if len(train) < burnin:
            continue
        mu = train.mean().values
        Xc = train.values - mu
        # SVD -> principal directions
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:3]  # 3 × n_tenors
        # apply to the window (refit_d, next_refit]
        nxt = bounds[i + 1]
        mask = (Y.index > refit_d) & (Y.index < nxt)
        win = Y.loc[mask]
        if win.empty:
            continue
        sc = (win.values - mu) @ comps.T  # rows × 3
        sc = pd.DataFrame(sc, index=win.index, columns=["level", "slope", "curvature"])
        # sign-align each component to its economic proxy on the training span
        train_sc = (train.values - mu) @ comps.T
        for k, name in enumerate(["level", "slope", "curvature"]):
            p = proxies[name].reindex(train.index).values
            c = np.corrcoef(train_sc[:, k], np.nan_to_num(p))[0, 1]
            if c < 0:
                sc[name] = -sc[name]
        out.loc[mask, ["level", "slope", "curvature"]] = sc.values
    return out.reindex(yields.index)


# ─── Signals ────────────────────────────────────────────────────────────────
def carry_rolldown(yields: pd.DataFrame, cc: str, short: Optional[pd.Series],
                   mod_dur: float) -> pd.DataFrame:
    """Carry+roll proxy for the 10Y point, in expected-monthly-return units.
       carry = (y10 - short)/12 ;  roll ≈ mod_dur * (y10 - y5)/5 / 12
       (roll-down as a 10Y ages toward 5Y over a month, linearised on the seg).
    """
    y10 = yields[f"{cc}_10y"]
    y5 = yields[f"{cc}_5y"]
    sr = short.reindex(yields.index).ffill() if short is not None else y5
    carry = (y10 - sr) / 100.0 / 12.0
    roll = mod_dur * ((y10 - y5) / 5.0) / 100.0 / 12.0
    sig = (carry + roll)
    return pd.concat({f"{cc}_carry_roll": sig,
                      f"{cc}_slope_10_2": (y10 - yields[f"{cc}_2y"])}, axis=1)


def _ar1_forecast(x: pd.Series, h: int) -> float:
    """Closed-form AR(1) h-step forecast from an expanding window."""
    x = x.dropna()
    if len(x) < 30:
        return np.nan
    x0, x1 = x.shift(1).dropna(), x.iloc[1:]
    x0 = x0.loc[x1.index]
    b = np.polyfit(x0.values, x1.values, 1)  # slope, intercept
    phi, c = float(np.clip(b[0], -0.999, 0.999)), float(b[1])
    mean = c / (1 - phi)
    return mean + (phi ** h) * (x.iloc[-1] - mean)


def mean_reversion(pca: pd.DataFrame, yields: pd.DataFrame, cc: str,
                   mod_dur: float, burnin: int, h: int = 21,
                   refit_freq: str = "ME") -> pd.Series:
    """Expected 10Y return from AR(1)-forecasting the curve factors and
       reconstructing the expected yield change. Expanding/monthly, no leak."""
    fac = pca.dropna(how="any")
    if fac.empty:
        return pd.Series(index=yields.index, dtype=float, name=f"{cc}_meanrev")
    # we need loadings to map factor change -> Δy10. Approximate via regression
    # of y10 on the three factors on an expanding window (same cadence).
    refit_dates = fac.iloc[burnin:].resample(refit_freq).last().index if len(fac) > burnin else []
    bounds = list(refit_dates) + [fac.index[-1] + pd.Timedelta(days=1)]
    y10 = yields[f"{cc}_10y"]
    out = pd.Series(index=yields.index, dtype=float, name=f"{cc}_meanrev")
    for i, rd in enumerate(refit_dates):
        tr_f = fac.loc[:rd]
        tr_y = y10.reindex(tr_f.index)
        valid = tr_y.notna()
        if valid.sum() < burnin:
            continue
        beta = np.linalg.lstsq(
            np.c_[np.ones(valid.sum()), tr_f.values[valid.values]],
            tr_y.values[valid.values], rcond=None)[0]  # [const, bL, bS, bC]
        # h-step factor forecasts
        f_fc = np.array([_ar1_forecast(tr_f[col], h) for col in tr_f.columns])
        f_now = tr_f.iloc[-1].values
        d_y10 = float(beta[1:] @ (f_fc - f_now))  # expected Δy10 over h
        exp_ret = -mod_dur * (d_y10 / 100.0)      # falling yield -> positive
        nxt = bounds[i + 1]
        mask = (yields.index > rd) & (yields.index < nxt)
        out.loc[mask] = exp_ret
    return out


def momentum(adj_price: pd.Series, windows=(5, 21, 63, 95, 126),
             vol_win: int = 63, name: str = "mom") -> pd.Series:
    """Ensemble fast trend on the tradeable (roll-adjusted) price. Positive =
       price rising = yields falling = long-duration signal. Vol-scaled."""
    px = adj_price.dropna()
    if len(px) < max(windows) + vol_win:
        return pd.Series(index=adj_price.index, dtype=float, name=name)
    ret = px.pct_change()
    vol = ret.rolling(vol_win).std()
    sigs = []
    for w in windows:
        ma = px.rolling(w).mean()
        sigs.append(((px - ma) / (px * vol.clip(lower=1e-6))))
    s = pd.concat(sigs, axis=1).mean(axis=1)
    return s.reindex(adj_price.index).rename(name)


def realized_vol(ret: pd.Series, win: int = 21, name="rvol") -> pd.Series:
    return (ret.rolling(win).std() * np.sqrt(252)).rename(name)


def term_premium_proxy(yields: pd.DataFrame, cc: str,
                       short: Optional[pd.Series]) -> pd.Series:
    """PLACEHOLDER term-premium proxy: y10 minus a trailing average of the
       short rate as a crude expected-average-short. Replace with ACM/KW for
       production — flagged, not load-bearing."""
    y10 = yields[f"{cc}_10y"]
    base = short.reindex(yields.index).ffill() if short is not None else yields[f"{cc}_2y"]
    exp_short = base.rolling(252, min_periods=63).mean()
    return (y10 - exp_short).rename(f"{cc}_tp_proxy")


# ─── Assemble ───────────────────────────────────────────────────────────────
def build(outdir: Path, roll_offset: int, burnin: int) -> None:
    print("=" * 72)
    print(" Build analysis panel — EUR duration overlay")
    print("=" * 72)

    yde, yfr, yit = (load_wide(outdir / f"yields_{c}.csv") for c in ("de", "fr", "it"))
    yields = pd.concat([d for d in (yde, yfr, yit) if d is not None], axis=1, sort=True)
    if yields.empty:
        raise SystemExit("No yield files found in outputs/. Run download_rates.py first.")
    yields = yields.sort_index().ffill(limit=3)

    short_df = load_wide(outdir / "short_rates.csv")
    short = short_df["euribor_3m"] if short_df is not None and "euribor_3m" in short_df else None
    risk = load_wide(outdir / "risk_proxies.csv")

    # roll flags (Bund delivery calendar applies to all Eurex bond futures)
    roll_df = load_wide(outdir / "fut_roll_dates.csv")
    roll_dates = (roll_df.iloc[:, 0] if roll_df is not None and roll_df.shape[1]
                  else None)

    feats: dict[str, pd.Series] = {}
    rets: dict[str, pd.Series] = {}

    # 1) roll-adjusted futures returns (the P&L backbone)
    print("\n──── Roll-adjusted futures returns ────")
    for mkt, cfg in MARKETS.items():
        print(f"  {mkt}")
        chain = None
        cf = outdir / f"fut_chain_{mkt}.csv"
        if cf.exists():
            chain = pd.read_csv(cf)
        cont = load_wide(outdir / cfg["cont"])
        cr = continuous_returns(chain, cont, cfg["cont_col"], roll_offset,
                                roll_dates=roll_dates)
        if not cr.empty:
            rets[f"{mkt}_ret"] = cr["ret"]
            feats[f"{mkt}_adjpx"] = cr["adj_price"]
            feats[f"{mkt}_rvol21"] = realized_vol(cr["ret"], 21, f"{mkt}_rvol21")
            feats[f"{mkt}_mom"] = momentum(cr["adj_price"], name=f"{mkt}_mom")

    # 2) curve factors + carry + mean-reversion + term premium (signal space)
    print("\n──── Curve factors & signals (expanding-window, no look-ahead) ────")
    for cc, tenors in CURVE_TENORS.items():
        present = [t for t in tenors if t in yields.columns]
        if len(present) < 3:
            print(f"  {cc}: <3 tenors present, skipping curve PCA")
            continue
        print(f"  {cc}: PCA + carry + mean-rev")
        pca = rolling_curve_pca(yields, present, burnin)
        for col in ["level", "slope", "curvature"]:
            feats[f"{cc}_{col}"] = pca[col].rename(f"{cc}_{col}")
        mkt = {"de": "bund", "fr": "oat", "it": "btp"}[cc]
        md = MARKETS[mkt]["mod_dur"]
        cr_df = carry_rolldown(yields, cc, short, md)
        for c in cr_df.columns:
            feats[c] = cr_df[c]
        feats[f"{cc}_meanrev"] = mean_reversion(pca, yields, cc, md, burnin)
        feats[f"{cc}_tp_proxy"] = term_premium_proxy(yields, cc, short)

    # 3) sovereign spreads (RV leg) + CDS hook
    print("\n──── Sovereign spreads & CDS basis ────")
    if {"it_10y", "de_10y"}.issubset(yields.columns):
        feats["btp_bund_10y"] = (yields["it_10y"] - yields["de_10y"]).rename("btp_bund_10y")
    if {"fr_10y", "de_10y"}.issubset(yields.columns):
        feats["oat_bund_10y"] = (yields["fr_10y"] - yields["de_10y"]).rename("oat_bund_10y")
    cds = load_wide(outdir / "cds_sovereign.csv")
    if cds is not None:
        # cash spread in bp vs CDS spread in bp -> basis. CDS leads in stress.
        if "it_5y" in cds and "btp_bund_10y" in feats:
            feats["it_cds_5y"] = cds["it_5y"].rename("it_cds_5y")
            feats["btp_cds_basis"] = (feats["btp_bund_10y"] * 100 - cds["it_5y"]
                                      ).rename("btp_cds_basis")
        if "fr_5y" in cds and "oat_bund_10y" in feats:
            feats["fr_cds_5y"] = cds["fr_5y"].rename("fr_cds_5y")
            feats["oat_cds_basis"] = (feats["oat_bund_10y"] * 100 - cds["fr_5y"]
                                      ).rename("oat_cds_basis")
    else:
        print("  · no cds_sovereign.csv — basis features skipped (optional)")

    # 4) regime / risk-appetite features (co-movers, NOT yield drivers)
    print("\n──── Regime features ────")
    if risk is not None:
        for col in ("vdax", "vstoxx", "move"):
            if col in risk.columns:
                feats[f"regime_{col}"] = risk[col].rename(f"regime_{col}")
    credit = load_wide(outdir / "credit_indices.csv")
    if credit is not None:
        for col in credit.columns:
            feats[f"regime_credit_{col}"] = credit[col].rename(f"regime_credit_{col}")

    # ── merge & save ────────────────────────────────────────────────────────
    panel = pd.concat(feats, axis=1, sort=True).sort_index()
    panel.columns = [c if isinstance(c, str) else c[0] for c in panel.columns]
    ret_panel = pd.concat(rets, axis=1, sort=True).sort_index()
    ret_panel.columns = [c if isinstance(c, str) else c[0] for c in ret_panel.columns]

    outdir.mkdir(exist_ok=True)
    panel.to_csv(outdir / "analysis_panel.csv")
    ret_panel.to_csv(outdir / "returns_markets.csv")

    print("\n" + "=" * 72)
    print(f"  analysis_panel.csv : {panel.shape[0]} rows × {panel.shape[1]} cols")
    print(f"  returns_markets.csv: {ret_panel.shape[0]} rows × {ret_panel.shape[1]} cols")
    if not panel.empty:
        span = f"{panel.index[0].date()} → {panel.index[-1].date()}"
        print(f"  span: {span}")
        nn = panel.notna().mean().sort_values()
        print("  least-populated columns (coverage):")
        for c, v in nn.head(6).items():
            print(f"    {c:22s} {v:5.0%}")
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser(description="Build the duration-overlay analysis panel")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--roll-offset", type=int, default=5,
                   help="business days before delivery to roll the future")
    p.add_argument("--pca-burnin", type=int, default=504,
                   help="min obs (≈2y) before curve PCA/AR1 produce output")
    args = p.parse_args()
    build(Path(args.outputs_dir), args.roll_offset, args.pca_burnin)


if __name__ == "__main__":
    main()
