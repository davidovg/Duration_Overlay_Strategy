"""
Core factor library: gravity-radius FX turbulence factor (Konstantinov & Fabozzi 2022)
and cross-asset network measures (Konstantinov & Sadeghi 2025; Konstantinov, Aldridge
& Kazemi 2023).

All functions operate on pandas objects indexed by date.
"""

import numpy as np
import pandas as pd
import networkx as nx

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 1. Gravity-radius factor (geometry of currency volatilities)
# ---------------------------------------------------------------------------
# Following Konstantinov & Fabozzi (2022), each pair of currencies (i, j)
# measured against a common base is represented as a triangle in Euclidean
# space: the two volatility vectors sigma_i, sigma_j span an angle theta_ij
# with cos(theta_ij) = rho_ij, and the law of cosines yields the cross-rate
# volatility as the third side:
#     sigma_cross^2 = sigma_i^2 + sigma_j^2 - 2 rho_ij sigma_i sigma_j.
# Rotating the triangle about the base-currency origin, Pappus-Guldin links
# the generated volume to the distance of the triangle's centroid from the
# axis of rotation.  The centroid of the triangle with vertices
# {0, sigma_i e_1, sigma_j (cos theta, sin theta)} sits at
#     C_ij = ( (sigma_i + sigma_j rho_ij)/3 , sigma_j sqrt(1-rho^2)/3 ),
# whose distance from the origin (the gravity radius) is
#     R_ij = (1/3) sqrt( sigma_i^2 + sigma_j^2 + 2 rho_ij sigma_i sigma_j ).
# R_ij increases in both volatilities AND in correlation, which is what makes
# the aggregate factor behave like a turbulence index (vol-up + corr-up
# regimes score highest).

def gravity_radius_pair(sig_i, sig_j, rho_ij):
    """Gravity radius of one currency-pair triangle."""
    return np.sqrt(np.maximum(sig_i**2 + sig_j**2 + 2.0 * rho_ij * sig_i * sig_j, 0.0)) / 3.0


def gravity_radius_index(returns: pd.DataFrame, vol_window: int = 63,
                         corr_window: int = 63, weights=None) -> pd.Series:
    """
    Aggregate gravity-radius factor across all currency pairs.

    returns : DataFrame of daily log returns of currencies vs. common base (EUR).
    weights : optional dict/Series of currency weights (defaults to equal).
    Output  : daily Series (annualised vol units).
    """
    vols = returns.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    cols = list(returns.columns)
    n = len(cols)
    if weights is None:
        w = pd.Series(1.0, index=cols)
    else:
        w = pd.Series(weights).reindex(cols).fillna(0.0)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    corr = returns.rolling(corr_window).corr()  # MultiIndex (date, col) x col

    out = pd.Series(index=returns.index, dtype=float)
    # vectorised: build per-pair series
    acc = None
    wsum = 0.0
    for i, j in pairs:
        ci, cj = cols[i], cols[j]
        rho = corr.xs(ci, level=1)[cj]
        r = gravity_radius_pair(vols[ci], vols[cj], rho)
        pw = w[ci] * w[cj]
        acc = r * pw if acc is None else acc + r * pw
        wsum += pw
    out = acc / wsum
    return out


def mahalanobis_turbulence(returns: pd.DataFrame, window: int = 252) -> pd.Series:
    """Classic Chow et al. (1999) / Kritzman-Li (2010) turbulence, for comparison."""
    out = pd.Series(index=returns.index, dtype=float)
    vals = returns.values
    for t in range(window, len(returns)):
        hist = vals[t - window:t]
        mu = np.nanmean(hist, axis=0)
        cov = np.cov(hist, rowvar=False)
        try:
            inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            continue
        d = vals[t] - mu
        out.iloc[t] = float(d @ inv @ d)
    return out


# ---------------------------------------------------------------------------
# 2. Cross-asset correlation network measures
# ---------------------------------------------------------------------------

def _corr_to_adjacency(corr: pd.DataFrame, threshold: float = 0.0,
                       absolute: bool = True) -> pd.DataFrame:
    """Weighted adjacency from a correlation matrix (zero diagonal)."""
    a = corr.abs() if absolute else corr.clip(lower=0.0)
    a = a.where(a >= threshold, 0.0)
    np.fill_diagonal(a.values, 0.0)
    return a


def network_snapshot(returns_window: pd.DataFrame, threshold: float = 0.3):
    """
    Compute network measures from one window of returns.
    Returns dict with density, eigenvector centrality (per node),
    degree (per node), mean correlation.
    """
    corr = returns_window.corr()
    a = _corr_to_adjacency(corr, threshold=threshold)
    g = nx.from_pandas_adjacency(a)
    n = len(corr)
    m_possible = n * (n - 1) / 2.0
    m_actual = (a.values > 0).sum() / 2.0
    density = m_actual / m_possible

    try:
        eig = nx.eigenvector_centrality_numpy(g, weight="weight")
    except Exception:
        eig = {c: np.nan for c in corr.columns}
    deg = dict(g.degree(weight="weight"))
    return {
        "density": density,
        "mean_corr": corr.values[np.triu_indices(n, 1)].mean(),
        "eigenvector": eig,
        "degree": deg,
    }


def rolling_network_measures(returns: pd.DataFrame, target: str,
                             window: int = 126, step: int = 5,
                             threshold: float = 0.3) -> pd.DataFrame:
    """
    Rolling network measures: overall density, mean correlation, and the
    target node's eigenvector centrality & weighted degree.
    Computed every `step` days and forward-filled to daily.
    """
    idx, rows = [], []
    for t in range(window, len(returns), step):
        win = returns.iloc[t - window:t].dropna(axis=1, how="any")
        if target not in win.columns or win.shape[1] < 5:
            continue
        snap = network_snapshot(win, threshold=threshold)
        idx.append(returns.index[t - 1])
        rows.append({
            "net_density": snap["density"],
            "mean_corr": snap["mean_corr"],
            "target_eigc": snap["eigenvector"].get(target, np.nan),
            "target_degree": snap["degree"].get(target, np.nan),
        })
    out = pd.DataFrame(rows, index=idx)
    return out.reindex(returns.index).ffill()


# ---------------------------------------------------------------------------
# 3. Signal utilities
# ---------------------------------------------------------------------------

def zscore(s: pd.Series, window: int = 156) -> pd.Series:
    """Rolling z-score (expanding fallback for early sample)."""
    m = s.rolling(window, min_periods=window // 3).mean()
    v = s.rolling(window, min_periods=window // 3).std()
    return (s - m) / v
