# Network & Geometry Duration Overlay for Bund Futures

Systematic duration overlay for Euro-Bund futures (FGBL) combining two strands of
Gueorgui S. Konstantinov's research:

1. **Gravity-radius FX turbulence factor** — Konstantinov & Fabozzi (2022),
   *The Geometry of the World of Currency Volatilities*, Computational Economics
   60:125–145. Each EUR currency pair is a triangle in volatility space
   (law of cosines: sides = vols, angle = correlation). The Pappus–Guldin centroid
   distance `R = (1/3)·sqrt(σi² + σj² + 2ρσiσj)` rises when vols spike *and*
   correlations rise — a turbulence gauge (corr ≈ 0.58 with VIX, peaks Jan 2009).
2. **Cross-asset network state variables** — Konstantinov & Sadeghi (2025, CFA RF
   monograph ch. 2), Konstantinov, Aldridge & Kazemi (2023, JPM). Rolling
   correlation networks over global rates / FX / vol nodes; regime signals from
   network **density** (Erdős–Rényi–Bollobás giant-component threshold) and the
   Bund node's **eigenvector centrality**.

These gate/condition baseline **momentum** and **carry** duration signals in a
weekly-rebalanced overlay (1-week execution lag, 1 bp turnover cost, Newey–West
inference). A one-layer GCN benchmark (Kipf–Welling) on the weekly network
snapshots is included as an AI comparison per Labonne (2023).

## Layout

```
src/factors.py        gravity radius, Mahalanobis turbulence, network measures
src/backtest.py       weekly overlay engine, synthetic FGBL returns, perf stats
src/make_data.py      public data pipeline (Fed H.10 FX, CBOE VIX)
download_refinitiv.py LSEG Workspace downloader (full universe, 2004→)
download_bloomberg.py Bloomberg blpapi downloader (same universe)
data/                 processed public data (fx crosses, VIX, gravity radius)
paper/                LaTeX draft for The Journal of Fixed Income + compiled PDF
```

## Reproduce

```bash
pip install -r requirements.txt
python src/make_data.py            # rebuilds FX/VIX panels from public sources
# run one of the downloaders on a terminal/Workspace machine for yield data
latexmk -pdf paper/main.tex        # compile the paper
```

## Status

- [x] Gravity-radius factor validated on public FX data (2005–2026)
- [x] Network machinery + backtest engine
- [x] Paper skeleton (methodology, lit review) — compiles
- [ ] Yield/futures data via Refinitiv (blocked on download run)
- [ ] Full backtest results + exhibits
- [ ] GCN benchmark results
