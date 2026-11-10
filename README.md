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

**Status:** no results are published yet. They depend on live data that was not available when this repository was last updated. Run `notebooks/01_exploratory_analysis.ipynb` to generate them. Results will be added here only from that run. Pre-stated threshold: an SP400/SP500 Sharpe ratio of at least 1.30x supports the thesis.

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
