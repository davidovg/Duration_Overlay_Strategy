# Duration Overlay Strategy

Systematic EUR-rates **tactical duration overlay** on Bund / OAT / BTP futures (1999–2026), built as a research programme to keep the carry premium while removing the crash. Overlay P&L ≈ Σ active_DV01 · (−Δy): signals are constructed in clean yield/DV01 space and executed in futures space.

> ⚠️ **Research simulation — not investment advice.** Every figure here is a backtest on historical data. It is not live trading, not net of all real-world costs, and not a recommendation.

> **Data is not included.** The scripts read/write an `outputs/` directory of Refinitiv/LSEG data that is **excluded from this repo** (licensed). Regenerate it with `download_rates.py` + `build_panel.py` against your own Workspace session.

## The arc

1. **Baselines & vol-targeting** — confirm the structural-long carry crash; a portfolio vol-target contains it.
2. **Long/short duration** (`longshort_duration.py`) — rule-based + walk-forward ML that can flip net short. A disciplined return-regressor finds no exploitable signal out-of-sample.
3. **Anomaly-detection overlay** (`anomaly_overlay.py`) — predict P(crash day) and de-risk; walk-forward, class-balanced. Detects crashes but only ties mechanical vol-targeting on this book.
4. **Global minimum-variance hedge** (`minvar_hedge.py`) — hedge the carry book with a global duration basket. First approach to raise return **and** Sharpe; turns 2022 positive.
5. **Regime-conditioned global hedge** (`regime_hedge.py`) — scale the hedge by a risk-on/risk-off / inflation regime. **Leading approach.**

**Headline finding:** the ceiling is *breadth*, not method. EUR duration is ~one factor (level = 73% of Bund/OAT/BTP variance), so single-universe overlays top out near Sharpe 0.2. Adding global duration drops the dominant factor to 56% and is what finally moves the needle.

## Selected results (research simulation)

| Strategy | Sharpe | Max DD | 2022–24 |
|---|---:|---:|---:|
| carry (long-only) | 0.15 | −50% | −38% |
| carry + vol-target | 0.16 | −19% | −8% |
| macro long/short + vol-target | 0.21 | −25% | −7% |
| global min-variance hedge | 0.22 | −42% | **+2%** |
| **regime-conditioned global hedge** | **0.27** | −41% | **+2%** |

See `docs/RESEARCH_SUMMARY.md` for the full table, findings, open items, and an annotated reading list.

## Pipeline

```bash
pip install -r requirements.txt

# 1. pull data (needs a running Refinitiv/LSEG Workspace + entitlements)
python download_rates.py
python build_panel.py            # -> outputs/analysis_panel.csv, returns_markets.csv

# 2. run any strategy stage (each reads outputs/, writes returns + a chart)
python longshort_duration.py --outputs-dir outputs
python anomaly_overlay.py     --outputs-dir outputs
python minvar_hedge.py        --outputs-dir outputs
python regime_hedge.py        --outputs-dir outputs
```

## Layout

```
.
├── download_rates.py        # Refinitiv/LSEG downloader
├── build_panel.py           # look-ahead-safe analysis panel
├── backtest.py              # baseline harness
├── defensive_carry.py       # carry crash-avoidance study
├── longshort_duration.py    # long/short + walk-forward ML
├── anomaly_overlay.py       # crash-classifier de-risk overlay
├── minvar_hedge.py          # global minimum-variance hedge
├── regime_hedge.py          # regime-conditioned global hedge (leading)
├── docs/RESEARCH_SUMMARY.md # full write-up + references
└── figures/                 # equity/drawdown charts
```

## Notes

- Scripts import from one another (e.g. `regime_hedge` imports `minvar_hedge` and `longshort_duration`), so keep them in the same directory / run from the repo root.
- The minimum-variance hedge and anomaly-detection components adapt methodology from a colleague's volatility-strategy framework, re-implemented here with strict walk-forward discipline.
