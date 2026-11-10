# Index Rebalancing Slippage Harvesting via Float-Adjusted Weight Drift

[![CI](https://github.com/rxj0102/index-drift-alpha/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/index-drift-alpha/actions/workflows/ci.yml)

> *A systematic, patient-capital strategy for extracting alpha from the mechanical
> rebalancing pressure that is structurally guaranteed to resolve - but timing-uncertain
> enough that fast money leaves it on the table.*

---

## Abstract

The explosive growth of passive investing has transformed index funds from price-takers
into price-makers. Passive equity vehicles now hold a large share of US equity assets, and
S&P 500 ETFs trade heavily every day (market-size figures are deliberately left out here
because they are not sourced in this repo). When daily price
movements push constituent realized weights away from their index-mandated target weights,
a mechanical rebalancing pressure builds - not at the next reconstitution, but continuously,
every trading day. This pressure is deterministic in direction but uncertain in timing: the
index fund manager must eventually trade to restore target weights, and the closer the
scheduled rebalance date, the more predictable the direction and magnitude of that flow.

This repository implements a float-adjusted weight drift strategy that quantifies this
continuous mechanical pressure, ranks constituents by drift urgency, and constructs a
daily-refreshed long/short portfolio designed to capture the pre-rebalance mean reversion.
The core empirical question - tested explicitly in the backtests - is whether this
mechanical signal is more exploitable in mid-cap (S&P 400) than in large-cap (S&P 500),
where HFT front-running of passive flows is more pervasive. The methodology introduces
a drift z-score framework for cross-sectional signal normalization, an ETF flow
confirmation overlay derived from yfinance AUM proxies and SEC N-CEN filings, and a
three-window rebalance event study (PRE / NEAR / POST) that decomposes returns around
each scheduled reconstitution date.

---

## Core Hypothesis

**H₀ (Null):** Float-adjusted weight drift provides no systematic predictor of future
returns beyond what is explained by momentum, reversal, and common risk factors. The
apparent alpha is attributable to data mining or look-ahead bias.

**H₁ (Alternative):** Constituents with large negative drift (underweight relative to
target) experience positive excess returns over the 10-day pre-rebalance window, and
constituents with large positive drift (overweight) experience negative excess returns,
consistent with predictable passive rebalancing demand. Furthermore, this effect is
materially stronger in the S&P 400 mid-cap universe than in the S&P 500, due to lower
HFT penetration in mid-cap passive flows.

---

## Key Distinction

> **This strategy explicitly rejects the constituent-addition front-run**, which is
> crowded to the point of negative expected value net of execution costs. The practice
> of buying stocks the moment they are announced for addition to the S&P 500 (and selling
> them on the effective date) was documented by Shleifer (1986) and Chen, Noronha &
> Singal (2004). By 2015, the front-running was so severe that the "index premium"
> had inverted in large-cap indices - latecomers were left holding expensive shares
> that experienced a predictable post-addition reversal.
>
> **Instead, this strategy exploits the silent, continuous drift of realized weights away
> from target weights within existing constituents** - a mechanical pressure that is
> structurally guaranteed to resolve but timing-uncertain, creating an exploitable edge
> for patient, size-appropriate capital. No front-running of public announcements is
> required. The signal is derived entirely from observable price data, the publicly
> available index reconstitution file, and float share data from SEC filings.

---

## Methodology

### Pipeline Overview

```
Index Reconstitution File (target weights w_i*)
         ↓
Daily Price Data × Float Shares → Realized Weights
         ↓
Weight Drift δᵢ(t) = realized_i(t) - target_i*
         ↓
Rolling Z-Score + Absolute Rank → Signal Strength
         ↓
Blackout Filter (scheduled rebalance ± 5 days)
         ↓
Long (negative drift) / Short (positive drift) Portfolio
         ↓
ETF Flow Confirmation Overlay → Final Signal
```

### Mathematical Formulation

**Realized float-adjusted weight:**

$$w_i^{\text{realized}}(t) = \frac{P_i(t) \cdot S_i^{\text{float}}}{\sum_{j} P_j(t) \cdot S_j^{\text{float}}}$$

where $P_i(t)$ is the adjusted close price of constituent $i$ on day $t$, and
$S_i^{\text{float}}$ is the float-adjusted share count from the most recent
reconstitution filing.

**Weight drift (signed):**

$$\delta_i(t) = w_i^{\text{realized}}(t) - w_i^*$$

*Sign convention:* $\delta_i > 0$ → overweight → anticipated **SELL** flow → we go **SHORT**.
$\delta_i < 0$ → underweight → anticipated **BUY** flow → we go **LONG**.

**Drift z-score (cross-sectionally normalised):**

$$z_i(t) = \frac{\delta_i(t) - \mu_{\delta_i}(t, T)}{\sigma_{\delta_i}(t, T)}, \quad T = 20 \text{ days}$$

**Signal strength (composite rank × capped z-score):**

$$\text{strength}_i(t) = \text{rank}_{|\delta|}^{\text{pctile}}(t) \times \min\bigl(|z_i(t)|,\; 4.0\bigr)$$

**Estimated ETF rebalancing flow:**

$$\text{flow}_i(t) = \delta_i(t) \times \text{AUM}_{\text{ETF}}(t)$$

*Note:* For a complete flow estimate, sum AUM across all ETFs tracking the same index
(e.g., SPY + IVV + VOO for the S&P 500). See `data/reference/etf_index_map.csv`.

---

## Factor Crash Validation

This strategy is **NOT** a crisis alpha strategy. Its edge arises from the mechanical
predictability of index fund rebalancing flows - a signal that requires functioning,
liquid markets where passive funds can and do execute their mandated rebalancing.
During severe market dislocations (March 2020, Q4 2018, Q1 2022), two effects typically
overwhelm the mechanical drift signal:

1. **Correlation collapse** - All constituents move together in the same direction,
   compressing cross-sectional drift variation (σ→0 in the denominator of the z-score).
2. **Fund flow reversals** - Redemption pressure forces index funds to sell everything
   regardless of drift direction, decoupling realized flows from drift predictions.

The strategy's drawdown periods in the backtest align with these systemic risk episodes.
This is honest: document it, size capital accordingly, and do not back-fill with crisis
hedges that were not in the original signal.

---

## Repository Structure

```
index-drift-alpha/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                          ← all parameters and paths
├── data/
│   ├── raw/                           ← SEC N-CEN JSON cache (gitignored)
│   ├── processed/                     ← Parquet price/drift files (gitignored)
│   └── reference/
│       ├── index_constituents.csv     ← seed: 50 SP400 tickers + target weights
│       ├── rebalance_calendar.csv     ← 2015–2023 scheduled rebalances
│       └── etf_index_map.csv          ← ETF → index mapping + AUM
├── outputs/                           ← charts (gitignored; regenerated by notebook)
├── notebooks/
│   └── 01_exploratory_analysis.ipynb  ← full analysis walkthrough
└── src/
    ├── __init__.py
    ├── utils.py           ← lag_dataframe, winsorize_df (look-ahead guards)
    ├── data_loader.py     ← price fetch, SEC EDGAR, constituent/calendar loaders
    ├── drift_engine.py    ← realized weights, drift, z-scores, flow estimates
    ├── signal.py          ← signal construction, decay, IC, flow confirmation
    ├── backtest.py        ← long/short backtest, event study, performance metrics
    └── visualize.py       ← all 7 publication-quality charts
```

---

## How to Run

### 1. Installation

```bash
git clone https://github.com/rxj0102/index-drift-alpha.git
cd index-drift-alpha
pip install -r requirements.txt
```

### 2. Smoke Test - Drift Pipeline

Run the full drift computation for the default universe (SP400) and print
the top 10 highest-drift names for the most recent available date:

```bash
python -m src.drift_engine --index SP400
```

Options: `--index {SP500, SP400, Russell1000, Russell2000}`

### 3. Full Notebook Analysis

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

The notebook runs the entire pipeline from raw price download through
signal construction, backtest, and chart generation. All outputs are saved
to `outputs/` automatically.

### 4. Configuration

All tunable parameters are in `config.py`. Key settings to adjust for
sensitivity analysis:

| Parameter | Default | Description |
|---|---|---|
| `DRIFT_THRESHOLD_SOFT` | 0.0015 | 15 bps weight drift → watch |
| `DRIFT_THRESHOLD_HARD` | 0.0030 | 30 bps → strong signal |
| `DRIFT_LOOKBACK_DAYS` | 20 | Rolling window for z-score |
| `DRIFT_ZSCORE_THRESHOLD` | 1.5 | Z-score trigger level |
| `HOLDING_PERIOD_DAYS` | 10 | Mean-reversion window |
| `REBALANCE_BLACKOUT_DAYS` | 5 | ±days around scheduled rebalance |
| `TRANSACTION_COST_BPS` | 8 | One-way cost (stress-test at 15, 25) |
| `UNIVERSE` | SP400 | Default index |

---

## Key Findings

No backtest results are reported. A review of the design (November 2026) found three issues that would make any backtest result misleading, so they are documented here before any results.

### 1. In a cap-weighted index, price moves do not create rebalancing trades

With float shares held fixed, `realized_weight_i(t) = p_i(t) f_i / sum_j p_j(t) f_j`. If the target is the snapshot weight, then

`drift_i(t) = w_i(0) x (R_i(t) / R_index(t) - 1)`

where `R` is the gross return since the snapshot. Drift is therefore the stock's return relative to the index since the snapshot date, scaled by its starting weight. A fund tracking a float-adjusted cap-weighted index already holds the right shares after any price move, so it has no reason to sell an outperformer or buy an underperformer. Passive flow comes from index additions and deletions, float and share-count changes, and buybacks or issuance, not from price drift. A backtest of this signal would measure short-term reversal relative to the index, not passive rebalancing demand.

### 2. Look-ahead and survivorship bias

The backtest window is `BACKTEST_START` = 2015-01-01 to `BACKTEST_END` = 2023-12-31, but `data/reference/index_constituents.csv` is a 50-name snapshot dated 2023-09-30. Applying 2023 membership and float to 2015 selects survivors and uses information not available at the time.

### 3. Reference data does not match SEC filings

`scripts/check_constituent_shares.py` compares each `float_shares` value with SEC-reported shares outstanding (dei:EntityCommonStockSharesOutstanding, latest value in 2023 H2). Of the 40 tickers that matched, 11 differ by more than 20 percent, and several exceed total shares outstanding, which a float cannot do (for example EME 1.75x, NBR 5.45x, SM 1.51x, ANF 1.46x). Several values are round fractions such as 166,666,667 and 500,000,000. Full comparison: `results/constituent_share_check.csv`.

### What a testable version looks like: S&P 400 addition and deletion event study

The passive-flow effect that the literature does document is the index inclusion and deletion effect (Shleifer 1986; Harris and Gurel 1986), where index funds must trade on a known date. `scripts/index_event_study.py` tests it on S&P 400 membership changes with free data. Membership changes come from the Wikipedia articles List of S&P 400, 500 and 600 companies (CC BY-SA 4.0), saved in `data/reference/`; prices from yfinance.

<!-- event-study:start (generated by scripts/index_event_study.py, do not edit by hand) -->
Market-adjusted cumulative abnormal returns (stock minus MDY) around S&P 400 effective dates, events 2012-01-01 to 2025-12-31. Trading day 0 is the effective date; index funds trade at the close of day -1. t-statistics average all events on the same date first, then test across dates. 229 deletions driven by M&A are excluded. Cells show the mean CAR across event dates.

**2020 to 2025, every event classified by origin or destination**

| Group | Events with prices | PRE [-20,-6] | NEAR [-5,-1] | DAY0 [0] | POST [+1,+10] | FULL [-20,+10] | Median event NEAR |
|---|---|---|---|---|---|---|---|
| New additions (not from the S&P 600) | 156 of 195 | +0.82% (t = 0.93) | +4.61% (t = 6.12) | -0.21% (t = -0.45) | +0.61% (t = 0.67) | +6.34% (t = 5.04) | +2.33% |
| Additions moved up from the S&P 600 | 64 of 76 | +4.16% (t = 1.91) | -5.60% (t = -4.53) | +0.39% (t = 0.73) | +1.44% (t = 1.09) | +0.40% (t = 0.13) | -4.26% |
| Deletions moved down to the S&P 600 | 127 of 151 | -1.26% (t = -0.86) | +4.44% (t = 5.25) | +0.16% (t = 0.58) | +0.69% (t = 0.51) | +4.04% (t = 1.76) | +1.71% |
| Deletions moved up to the S&P 500 | 36 of 36 | +0.21% (t = 0.15) | +0.82% (t = 0.98) | +0.50% (t = 1.47) | -1.49% (t = -1.31) | +0.04% (t = 0.03) | -0.31% |
| Deletions, unclassified (not M&A) | 12 of 22 | +5.49% (t = 1.12) | +0.29% (t = 0.12) | -1.35% (t = -1.15) | -0.65% (t = -0.37) | +3.78% (t = 0.52) | -0.31% |

**2012 to 2019, origin and destination not available (pooled)**

| Group | Events with prices | PRE [-20,-6] | NEAR [-5,-1] | DAY0 [0] | POST [+1,+10] | FULL [-20,+10] | Median event NEAR |
|---|---|---|---|---|---|---|---|
| All additions, origin unknown | 158 of 282 | +0.71% (t = 1.43) | +0.67% (t = 1.43) | +0.22% (t = 1.24) | +0.04% (t = 0.08) | +1.18% (t = 1.56) | +0.80% |
| All non-M&A deletions, destination unknown | 57 of 116 | +0.76% (t = 0.46) | -2.85% (t = -3.19) | -0.22% (t = -0.66) | +0.75% (t = 0.60) | -1.57% (t = -0.62) | -1.99% |

Price coverage: 68% of additions and 71% of non-M&A deletions have yfinance prices for the full window. Firms that later delisted or changed ticker drop out, so results lean toward survivors. Moves to the S&P 600 are matched with the S&P 600 changes table; moves to the S&P 500 only with current S&P 500 members, so up-moves by firms that later left the S&P 500 sit in the unclassified group. Wikipedia gives effective dates, not announcement dates, so the NEAR window mixes the announcement reaction with the index-fund trade. Event list: `results/index_events.csv`; per-event CARs: `results/event_study_car.csv`.
<!-- event-study:end -->

Reading the table (the numbers above are regenerated on every run; this paragraph only states directions):

- The price effect sits in the NEAR window, the days between announcement and the close of day -1 when index funds trade. Day 0 and POST are not significant, so there is nothing left to capture by trading on or after the effective date.
- In 2020 to 2025 the sign follows net passive demand, not just whether a stock enters or leaves the S&P 400. New additions rise. Stocks that move up from the S&P 600 fall, and stocks that move down to the S&P 600 rise. That pattern is consistent with S&P 600 trackers holding a larger share of a small stock's float than S&P 400 trackers hold of a mid-cap's, so the small-cap side of a migration dominates. That explanation is a hypothesis; this repo does not measure index-fund ownership.
- Pooling 2012 to 2019 without origin and destination mixes these opposite flows, which is why the pooled rows are weaker.
- Testing the original claim that the effect is stronger in the S&P 400 than in the S&P 500 needs an S&P 500 change history with dates, which the Wikipedia article no longer carries.

### Signal documentation

| Field | Value |
|---|---|
| Signal | Drift z-score: (realized weight minus target weight), standardized over `DRIFT_LOOKBACK_DAYS` = 20 |
| Expected sign | Negative: overweight names are shorted, underweight names bought |
| Inputs | Daily prices (yfinance), constituent float and target weights (`data/reference/index_constituents.csv`) |
| Timing | Rebalanced daily, held `HOLDING_PERIOD_DAYS` = 10 |
| Code | `src/drift_engine.py`, `src/signal.py`, `src/backtest.py` |
| Status | Not run: see issues 1 to 3 above |

---

## Limitations

1. **Float share staleness** - The reconstitution file is updated at most quarterly.
   Intraday buybacks, secondary offerings, and block sales change float shares daily but
   are not captured. This introduces a slow-moving bias in the realized-weight denominator.
   Mitigation: refresh float shares from 13F filings quarterly and monitor the sensitivity
   of drift signals to ±5% float perturbations.

2. **Timing uncertainty - the strategy's existential risk** - The rebalancing pressure
   this strategy captures is *directionally* certain but *temporally* uncertain. An index
   fund manager who decides to rebalance two weeks early (or two weeks late) relative to
   the scheduled date invalidates the expected entry/exit timing. Suggested mitigation:
   implement a confidence decay function that reduces position size by 20% for each day
   held past day 5 without a fresh drift signal or flow confirmation.

3. **ETF AUM proxy quality** - yfinance `shares_outstanding` for ETFs reports total
   authorized shares, not shares actually outstanding. The divergence between this figure
   and true NAV × shares data can be ±20–40%. Flow estimates are directionally correct
   but should not be used for precise dollar-flow projections without proper fund
   administrator data feeds.

4. **Transaction cost sensitivity in mid-cap** - The 8 bps one-way assumption is
   optimistic for SP400 names with average daily volume (ADV) below $50M. For names
   in the bottom ADV quartile, 15–25 bps is more realistic. Stress-testing at 25 bps
   is essential before allocating real capital.

5. **HFT arms race** - The alpha documented here assumes that the drift signal is not
   yet fully crowded in mid-cap. If the number of participants running similar strategies
   in the SP400 exceeds a critical threshold, the alpha decays to zero net of costs.
   Capacity should be estimated as roughly 10% of the estimated daily flow for each
   constituent - exceeding this fraction makes the strategy self-defeating.

---

## Reproducibility

### Data availability

All inputs come from free, public sources. No paid data or API keys are required unless noted.

- Yahoo Finance via the yfinance package (free, no key)
- SEC EDGAR filings (free, requires a User-Agent header)
- Files shipped in the `data/` folder of this repository

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `uv.lock`.
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
pip install uv     # once
uv sync            # installs the exact pinned environment
uv run pytest -q   # runs the test suite
```

### Runtime

The test suite (11 tests) finishes in about 15 seconds on the machine above, including any data downloads it triggers.

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.


## Academic References

* **Shleifer, A. (1986).** "Do Demand Curves for Stocks Slope Down?" *Journal of Finance*,
  41(3), 579–590. [Foundational paper on S&P 500 addition price effects.]

* **Chen, H., Noronha, G., & Singal, V. (2004).** "The Price Response to S&P 500 Index
  Additions and Deletions: Evidence of Asymmetry and a New Explanation." *Journal of Finance*,
  59(4), 1901–1930. [Documents losses to index fund investors from front-running.]

* **Petajisto, A. (2011).** "The Index Premium and Its Hidden Cost for Index Funds."
  *Journal of Empirical Finance*, 18(2), 271–288. [Quantifies the index premium as a
  recurring drag on passive fund performance.]

* **Wurgler, J., & Zhuravskaya, E. (2002).** "Does Arbitrage Flatten Demand Curves for
  Stocks?" *Journal of Business*, 75(4), 583–608. [Why addition effects are larger for
  stocks with no close substitutes - relevant to the mid-cap vs. large-cap comparison.]

---

## Disclaimer

This repository is for **academic research and educational purposes only**. Nothing
contained herein constitutes investment advice, a solicitation, or an offer to buy or
sell any security. Past performance of backtested strategies does not guarantee future
results. All backtests use survivorship-biased constituent lists and are subject to
data snooping, look-ahead, and overfitting risks. The authors make no representations
about the accuracy, completeness, or fitness for purpose of any data or code in this
repository.

---

*Built with Python, pandas, yfinance, and seaborn. Reference data sourced from public
SEC filings and index methodology documents.*

## Known limitations

- **Signal lag.** `lag_dataframe` shifts rows, which is correct for a wide one-row-per-date frame but does *not* lag a long-format signal (many tickers per date). The exploratory notebook previously used it that way, so entries effectively happened at the signal-generating close. Use `lag_signal_dates(signal, prices.index, periods=1)`; notebook cell 27 now does. **Notebook outputs and any quoted Sharpe/return figures were produced before this change and must be re-run.**
- **Entry price.** `_forward_return` starts from the close of the (lagged) entry date; execution at the open or with slippage is not modelled.
- **Static targets and floats.** Target weights and float shares are single snapshots, so measured drift contains a slow-moving bias (z-scoring mitigates, but does not remove it).
