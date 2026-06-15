# EUR Rates Duration Overlay — Research Summary

**Prepared for:** internal review
**Status:** research simulation on historical data — **NOT investment advice or a live recommendation**
**Universe:** Bund / OAT / BTP futures (EUR tactical duration), 1999–2026
**Frame:** overlay P&L ≈ Σ active_DV01 · (−Δy) ≈ Σ notional · futures_return; signals built in clean yield/DV01 space, executed in futures space.

> Note on provenance: the anomaly-detection and minimum-variance hedge components adapt methodology from F. Romano's volatility-strategy framework, re-implemented with walk-forward discipline and applied to the duration book. Treat accordingly re: internal sharing.

---

## 1. Objective

Build a systematic EUR-rates duration overlay with a materially better drawdown profile than a FINCA-style three-signal (carry + mean-reversion + momentum) construction, which lost ~12% in a single month (Mar-2026) on a structural duration-long carrying into a selloff. The working question throughout: *can we keep the carry premium while removing the crash?*

## 2. Data

Pulled via Refinitiv/LSEG and assembled into a look-ahead-safe panel:
- Benchmark yields DE/FR/IT (2/5/10/30y, ~7,000 obs each) → curve PCA (level/slope/curvature, expanding-window refit), carry/roll-down, value, momentum, sovereign spreads.
- Futures continuations c1+c2 (roll-adjusted) for Bund/Bobl/Schatz/Buxl/OAT/BTP.
- Global 10y yields US/GB/CA/AU (used for the hedge basket).
- Short rates (Euribor 3m, €STR); risk proxies (Brent, EURUSD).
- Gaps: equity index / equity-vol (VDAX, MOVE) and CDS came back empty on the feed — see open items.

## 3. What was built (in order)

1. **Baselines & vol-targeting** — carry, carry+momentum, FINCA-static, and a vol-targeted/drawdown-managed variant. Confirms the structural-long carry crash and that a portfolio vol-target contains it but a naive drawdown de-gross worsens whipsaw.
2. **Long/short duration** — strategies that can flip *net short*: rule-based (value, macro-regime, combo) and a walk-forward ML return-regressor (`longshort_duration.py`).
3. **Anomaly-detection overlay** — Romano's reframe: predict P(crash day) as a classifier and de-risk by it, walk-forward, class-balanced (`anomaly_overlay.py`).
4. **Global minimum-variance hedge** — closed-form and downside-tail (Romano's `cov_vola_following`) hedge of the carry book using a global duration basket (`minvar_hedge.py`).
5. **Regime-conditioned global hedge** — scale the hedge by a risk-on/risk-off / inflation regime so it stays on in inflation selloffs and turns off in flight-to-quality (`regime_hedge.py`). **Current leading approach.**

## 4. Results (research simulation; benchmark is a passive-long proxy)

| Strategy | Ann % | Vol % | Sharpe | Max DD % | 2022–24 | Mar-26 |
|---|---:|---:|---:|---:|---:|---:|
| carry (long-only) | 2.27 | 15.3 | 0.15 | −50.5 | −37.6% | −7.2% |
| carry + vol-target | 0.98 | 6.2 | 0.16 | −18.7 | −7.6% | −1.4% |
| macro long/short | 2.83 | — | 0.17 | −42 | −24% | — |
| macro long/short + vol-target | — | — | 0.21 | −25 | −6.6% | −1.7% |
| ML return-regressor (long/short) | ~0 | 1.6 | ~0 | −8 | flat | flat |
| anomaly de-risk overlay (best variant) | 1.05 | 6.3 | 0.17 | −19 | +crash cut | — |
| **global min-variance hedge** | **2.84** | 13.2 | **0.22** | −42 | **+1.6%** | −0.6% |
| global hedge + vol-target | 0.91 | 6.3 | 0.14 | −20 | +1.0% | +0.1% |
| **regime-conditioned global hedge** | **3.66** | 13.4 | **0.27** | −41 | **+2.0%** | **+5.6%** |
| regime hedge + vol-target | 1.44 | 6.3 | 0.23 | −19 | +3.2% | +1.8% |

## 5. Key findings (the honest read)

- **Breadth, not method, is the ceiling.** EUR duration is essentially one risk factor: the level PC explains **73%** of Bund/OAT/BTP return variance (top-two = 94%). Every single-universe overlay — vol-target, anomaly classifier, macro gate — tops out near Sharpe 0.2 because it can only rearrange exposure to that one factor.
- **Adding global duration is the unlock.** It drops the dominant factor from 73% → **56%** (top-two 94% → 71%): real orthogonal risk appears. The min-variance hedge monetises it — Sharpe 0.22 with *higher* return than naive carry and a **positive 2022** (short global duration profits in the common selloff that drives most of the carry drawdown).
- **Regime-conditioning is additive.** Scaling the hedge by a risk-on/off signal — on in inflation selloffs, off in flight-to-quality — lifts Sharpe to **0.27**, the best result here. Economic logic: don't short duration when bonds are rallying as a haven (2008/2020); short hard when they've lost the haven bid (2022).
- **Trend-following is negative on this universe** at every horizon — it cannot serve as the carry-crash hedge here (contrast with the diversified-futures TSM literature).
- **Predicting returns fails; predicting risk regime is the right target.** A disciplined walk-forward return-regressor finds no exploitable signal (consistent with the robustness literature). Romano's crash-classifier reframe is the correct idea, but on *this* book it only ties mechanical vol-targeting, because the predictable part of "crash risk" here is largely volatility — which vol-targeting already exploits. A crash classifier earns its keep only with a *leading* feature (inflation expectations, equity-vol) that volatility itself lacks.

## 6. Open items / next steps (priority order)

1. **Real global government futures** (TY, Gilt, CGB, ACGB, poss. JGB) to replace the synthetic −D·Δy global returns and make the hedge tradeable. *Highest-value single pull.*
2. **Equity / equity-vol series** (`.STOXX50E`/`.SPX`, `.V2TX`/`.VIX`) to replace the oil/USD proxy in the regime signal with a true stock-bond-correlation classifier — measures directly whether bonds still have a haven bid (would have called COVID correctly where oil did not). Hook already built (`danger_from_signals`).
3. **EUR 5y5y inflation forward** to *lead* the macro short (the realized-policy gate lags the inflation shock by construction).
4. **Curve trades** (2s10s steepeners/flatteners) — orthogonal to the level factor by construction; further breadth.
5. **Convexity sleeve** (payer swaptions) for the fast spikes the level hedge cannot time (Mar-2026 type).
6. **Validation before trusting the 0.27**: walk-forward selection of the regime-multiplier parameters + block-bootstrap significance test on the leading Sharpe. The regime mapping is currently a sensible a-priori choice, not an out-of-sample-fit result.
7. Residual tail: the global hedge removes only the *common* factor; a BTP/sovereign-specific blowup is unhedged (hence −41% DD on the un-vol-targeted hedge). Vol-targeting the residual pulls this to ~−19%.

## 7. Code inventory

| File | Purpose |
|---|---|
| `download_rates.py` | Refinitiv/LSEG downloader (dual backend, field cascade, roll handling) |
| `build_panel.py` | Look-ahead-safe analysis panel + market returns |
| `backtest.py` | Baseline harness (carry / FINCA-static / vol-targeted) |
| `defensive_carry.py` | Carry crash-avoidance study (trend vs macro gating) |
| `longshort_duration.py` | Long/short duration + walk-forward return-regressor ML |
| `anomaly_overlay.py` | Walk-forward crash-classifier de-risk overlay |
| `minvar_hedge.py` | Global minimum-variance hedge (closed-form + downside-tail) |
| `regime_hedge.py` | Risk-on/off regime-conditioned global hedge **(leading)** |
| `*.png` | Equity/drawdown charts for each stage |

---

## 8. Further reading (annotated)

### Bond return predictability & robustness
- **Bauer & Hamilton (2018), "Robust Bond Risk Premia," *Review of Financial Studies*.** The cautionary anchor: once you correct the inference, only the level and slope of the curve are robust out-of-sample predictors; most macro/unspanned predictability is fragile. Why we stayed parsimonious. https://www.researchgate.net/publication/345699693_Robust_Bond_Risk_Premia
- **Bianchi, Büchner & Tamoni (2021), "Bond Risk Premiums with Machine Learning," *RFS* 34(2):1046–1089** — and crucially the **Corrigendum (2021)**. ML (trees/NNs) on yields + macro shows bond predictability and economic gains; the corrigendum, after removing look-ahead, finds a *lower* but still significant OOS R². A direct lesson in walk-forward hygiene. https://doi.org/10.1093/rfs/hhaa062
- **Brooks & Moskowitz, "Yield Curve Premia."** Value/momentum/carry applied to the curve, internationally — the style-premia lens on duration. https://spinup-000d1a-wp-offload-media.s3.amazonaws.com/faculty/wp-content/uploads/sites/3/2021/08/Yield-Curve-Premia.pdf

### Carry & trend (style premia in fixed income)
- **Koijen, Moskowitz, Pedersen & Vrugt (2018), "Carry," *JFE*.** FI carry as a predictor; importantly, fixed-income carry does *not* carry FX-carry's extreme negative skew — relevant to how we framed the carry-crash.
- **Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum," *JFE* 104(2).** The canonical trend paper; a diversified TSM book "performs best during extreme markets" — yet it is *negative* on our narrow 3-instrument EUR universe, which is the breadth point in miniature. https://www.sciencedirect.com/science/article/pii/S0304405X11002613

### Stock–bond correlation & risk-on/risk-off regimes (the regime conditioner)
- **Campbell, Pflueger & Viceira (2020), "Macroeconomic Drivers of Bond and Equity Risks," *Journal of Political Economy* 128(8):3148–3185.** The theoretical backbone for the regime hedge: the sign of the stock-bond correlation flips with the inflation–output-gap correlation; bonds switch between safe-haven and risky. Exactly the 2022 vs 2008/2020 distinction we condition on. https://www.journals.uchicago.edu/doi/abs/10.1086/707766
- **Pflueger (2025), "Back to the 1980s or not? The drivers of inflation and real risks in Treasury bonds," *JFE* 167.** Recent and on-point for the post-2022 regime.
- **Campbell, Sunderam & Viceira (2017), "Inflation Bets or Deflation Hedges? The Changing Risks of Nominal Bonds," *Critical Finance Review* 6(2).** Time-variation in whether nominal bonds hedge or amplify equity risk.
- Accessible primer: Campbell, Viceira & Pflueger, "When Do Stocks and Bonds Move Together, and Why Does it Matter?" *Econofact* (2023). https://econofact.org/when-do-stocks-and-bonds-move-together-and-why-does-it-matter

### Anomaly detection & crash prediction (Romano-style methodology)
- **"Explainable Ensemble Learning for Predicting Stock Market Crises: Calibration, Threshold Optimization, and Robustness Analysis," *Information* (MDPI), 2026.** The most directly useful upgrade path: calibrated tree-ensembles for crash prediction, explicitly handling class imbalance, probability *calibration*, and threshold selection, reporting early-warning lead time. Addresses precisely the miscalibration ("trigger-happy" P) we hit in `anomaly_overlay.py`. https://www.mdpi.com/2078-2489/17/2/114
- **Chatzis, Siakoulis, Petropoulos, Stavroulakis & Vlachogiannakis (2018), "Forecasting stock market crisis events using deep and statistical machine learning techniques," *Expert Systems with Applications*.** Daily stock/bond/currency data across 39 countries; classification trees, SVM, random forests, NNs, XGBoost and DNNs, with bootstrap resampling for class imbalance and a long OOS window — the closest published analog to the anomaly-detection overlay. https://www.sciencedirect.com/science/article/abs/pii/S0957417418303798
- Survey context: ensemble methods (RF / gradient boosting) for rare-event/anomaly detection and the recurring practical issues — class imbalance, false-positive control, calibration — that dominate this problem.

---
*All figures are backtested research simulations on historical data and do not represent live trading, returns net of all real-world costs, or any recommendation. Not investment advice.*
