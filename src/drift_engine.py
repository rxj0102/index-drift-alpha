"""
drift_engine.py — Float-adjusted weight drift: computation and signal flagging.

Module-level sign convention (read this before modifying any formula)
----------------------------------------------------------------------
For constituent *i* at time *t*::

    drift_i(t) = realized_weight_i(t) − target_weight_i

Numerical example
~~~~~~~~~~~~~~~~~
Suppose ticker XYZ has a target weight of 2.00 % in the index.
Over the past month XYZ outperformed the index by +8 %.  Its price rose
from $100 to $108 while the average index price rose only 2 %.

If float_shares_XYZ = 50,000,000 and total float market cap of the index
= $250 billion::

    realized_weight_XYZ = (108 × 50_000_000) / 250_000_000_000
                        = 5_400_000_000 / 250_000_000_000
                        = 0.0216  →  2.16 %

    drift_XYZ = 0.0216 − 0.0200 = +0.0016  (+16 bps)

**Interpretation**: XYZ is *overweight* relative to its target.
The index fund manager must *sell* XYZ at the next rebalance to bring it
back to 2.00 %.  We therefore take a SHORT position in XYZ, anticipating
that the passive selling will create downward price pressure.

Sign summary:
    drift > 0  →  overweight  →  expected SELL flow  →  we go SHORT
    drift < 0  →  underweight →  expected BUY flow   →  we go LONG

This is the opposite of momentum — we are trading *against* the recent
winner (which is the index fund doing its mechanical job of rebalancing).

Usage
-----
    python -m src.drift_engine --index SP400

    # or from Python:
    from src.drift_engine import compute_realized_weights, compute_weight_drift
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import config
from src.data_loader import (
    fetch_constituent_prices,
    load_index_constituents,
    load_rebalance_calendar,
    is_blackout_date,
)
from src.utils import winsorize_df

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. compute_realized_weights
# ---------------------------------------------------------------------------

def compute_realized_weights(
    price_df: pd.DataFrame,
    constituents_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute daily float-adjusted realized weights for all constituents.

    For each date *t* and each constituent *i*::

        realized_weight_i(t) = price_i(t) × float_shares_i
                               / Σ_j (price_j(t) × float_shares_j)

    Parameters
    ----------
    price_df : pd.DataFrame
        Wide-format adjusted close prices: index = trading dates, columns =
        tickers, values = USD price.  Must contain all tickers in
        ``constituents_df``.
    constituents_df : pd.DataFrame
        Output of :func:`data_loader.load_index_constituents`.  Must contain
        ``ticker`` and ``float_shares`` columns.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame: index = trading dates, columns = tickers,
        values = float-adjusted realized weight (0–1, sums to 1 per row).

    Notes
    -----
    **Float share staleness** — ``float_shares_i`` is held constant at the
    last reconstitution snapshot value loaded from
    ``data/reference/index_constituents.csv``.  In production, float share
    updates from quarterly SEC 13F filings, Form 144 sales, and corporate
    buyback announcements would refresh this figure.  Stale float shares
    cause the denominator of the formula to diverge from the true index
    float market cap, introducing a slow-moving bias in measured drift.
    For the purposes of this research prototype the bias is tolerable
    because: (a) share counts change slowly relative to price changes, and
    (b) we use cross-sectional drift *z-scores*, which largely cancel out
    level biases that affect all names uniformly.

    **Missing prices** — tickers missing a price on date *t* contribute
    ``NaN`` to that day's float market cap sum.  This understates the
    denominator and inflates the weights of all present names.  The
    :func:`data_loader.fetch_constituent_prices` function already warns
    about tickers with >5 % missing data.
    """
    # Align float shares to the price DataFrame column order
    float_map = constituents_df.set_index("ticker")["float_shares"]
    common_tickers = [t for t in price_df.columns if t in float_map.index]

    if len(common_tickers) < len(price_df.columns):
        missing = set(price_df.columns) - set(float_map.index)
        logger.warning(
            "Tickers in price_df but not in constituents_df (excluded from weights): %s",
            sorted(missing),
        )

    price_subset = price_df[common_tickers]
    float_arr = np.array([float_map[t] for t in common_tickers], dtype=float)

    # float_market_cap[t, i] = price_i(t) × float_shares_i
    float_mkt_cap = price_subset.values * float_arr  # broadcast along columns

    # Denominator: total float market cap per day
    total_float_mkt_cap = np.nansum(float_mkt_cap, axis=1, keepdims=True)

    # Avoid division by zero on all-NaN rows
    total_float_mkt_cap[total_float_mkt_cap == 0] = np.nan

    realized_w = float_mkt_cap / total_float_mkt_cap

    realized_df = pd.DataFrame(
        realized_w, index=price_subset.index, columns=common_tickers
    )
    logger.info(
        "Realized weights computed: %d dates × %d tickers.",
        realized_df.shape[0], realized_df.shape[1],
    )
    return realized_df


# ---------------------------------------------------------------------------
# 2. compute_weight_drift
# ---------------------------------------------------------------------------

def compute_weight_drift(
    realized_weights: pd.DataFrame,
    constituents_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute signed weight drift: realized − target, for every date and ticker.

    Parameters
    ----------
    realized_weights : pd.DataFrame
        Output of :func:`compute_realized_weights`.
    constituents_df : pd.DataFrame
        Must contain ``ticker`` and ``target_weight`` columns.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with identical structure to ``realized_weights``:
        index = trading dates, columns = tickers, values = drift (signed float).

    Notes
    -----
    **Sign convention** — positive drift means the stock has *outperformed*
    and is currently *overweight* relative to its target.  An index fund must
    **sell** it at the next rebalance.  We therefore anticipate sell pressure
    and take a **short** position.  Negative drift implies *underweight*:
    the fund must **buy**, so we go **long**.  See the module docstring for
    a worked numerical example.

    **Drift units** — drift is expressed in weight-point terms (e.g. 0.0016
    means 16 basis points of weight drift), not in return space.  This makes
    the signal comparable across names with different volatility profiles but
    requires setting thresholds (``DRIFT_THRESHOLD_SOFT``,
    ``DRIFT_THRESHOLD_HARD``) in the same weight-point units.

    **Secular drift** — large-cap growth stocks persistently accumulate
    positive drift between rebalances because their prices tend to rise faster
    than the index average (momentum effect).  This secular component is
    removed by z-scoring in :func:`compute_drift_zscore`, which is why
    the z-score is preferable to raw drift for cross-sectional ranking.
    """
    target_map = constituents_df.set_index("ticker")["target_weight"]
    common = [t for t in realized_weights.columns if t in target_map.index]

    if not common:
        raise ValueError("No common tickers between realized_weights and constituents_df.")

    target_arr = np.array([target_map[t] for t in common])
    drift = realized_weights[common].values - target_arr  # broadcast

    drift_df = pd.DataFrame(
        drift, index=realized_weights.index, columns=common
    )
    logger.info(
        "Weight drift computed: mean |drift| = %.4f, max |drift| = %.4f.",
        np.nanmean(np.abs(drift)), np.nanmax(np.abs(drift)),
    )
    return drift_df


# ---------------------------------------------------------------------------
# 3. compute_drift_zscore
# ---------------------------------------------------------------------------

def compute_drift_zscore(
    drift_df: pd.DataFrame,
    window: int = config.DRIFT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Compute rolling z-score of each ticker's drift time series.

    For each ticker *i* at date *t*::

        z_i(t) = (drift_i(t) − rolling_mean_i(t, window))
                 / rolling_std_i(t, window)

    Parameters
    ----------
    drift_df : pd.DataFrame
        Output of :func:`compute_weight_drift`.
    window : int, optional
        Rolling window length in trading days.  Default is
        ``config.DRIFT_LOOKBACK_DAYS`` (20 days).

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame of drift z-scores, winsorised at
        ±``config.DRIFT_ZSCORE_WINSOR_CAP`` (default ±4.0).

    Notes
    -----
    **Why z-score instead of raw drift?**

    1. **Cross-sectional comparability** — A 20 bps drift for a high-
       volatility small-cap is less remarkable than the same drift for a
       low-volatility large-cap.  Z-scoring normalises for each name's
       historical drift distribution, making the signal comparable across
       names with different size and beta profiles.

    2. **Secular trend removal** — Large-cap growth stocks accumulate
       persistent positive drift as momentum compounds between rebalances.
       Rolling z-scoring removes this level shift, keeping the signal
       stationary and preventing the long/short book from being persistently
       skewed toward one end of the size spectrum.

    3. **Outlier robustness** — Corporate actions (splits, spin-offs, large
       secondaries) cause abrupt float share changes that appear as extreme
       drift spikes.  Winsorisation at ±4.0 caps these events, preventing
       them from dominating the cross-sectional signal on an affected date.

    **Minimum window** — The z-score is NaN for the first ``window`` rows
    (insufficient history).  Backtest functions must handle this by dropping
    NaN-dominated dates before evaluating signals.
    """
    roll_mean = drift_df.rolling(window=window, min_periods=max(5, window // 2)).mean()
    roll_std = drift_df.rolling(window=window, min_periods=max(5, window // 2)).std()

    # Avoid division by zero when std is zero (constant drift series)
    roll_std = roll_std.replace(0, np.nan)

    zscore_df = (drift_df - roll_mean) / roll_std

    # Winsorise at ±DRIFT_ZSCORE_WINSOR_CAP
    cap = config.DRIFT_ZSCORE_WINSOR_CAP
    zscore_df = zscore_df.clip(lower=-cap, upper=cap)

    logger.info(
        "Drift z-scores computed (window=%d): mean=%.3f, std=%.3f.",
        window,
        float(np.nanmean(zscore_df.values)),
        float(np.nanstd(zscore_df.values)),
    )
    return zscore_df


# ---------------------------------------------------------------------------
# 4. compute_absolute_drift_rank
# ---------------------------------------------------------------------------

def compute_absolute_drift_rank(drift_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank of |drift| for each date.

    Parameters
    ----------
    drift_df : pd.DataFrame
        Output of :func:`compute_weight_drift`.

    Returns
    -------
    pd.DataFrame
        Percentile ranks (0.0 to 1.0 scale): same index and columns as
        ``drift_df``.  Rank 1.0 means highest absolute drift on that date.

    Notes
    -----
    A name ranked near 1.0 has the largest absolute drift relative to all
    other constituents on that date.  This means it has the *most pressing*
    mechanical rebalancing need and should therefore receive the strongest
    signal weight.

    The rank is based on absolute drift (|drift|), not signed drift, because
    both overweight (sell pressure) and underweight (buy pressure) extremes
    are of interest.  Direction is captured separately via the sign of the
    raw drift and z-score.
    """
    abs_rank = drift_df.abs().rank(axis=1, pct=True, na_option="keep")
    logger.info(
        "Absolute drift percentile ranks computed for %d dates.", abs_rank.shape[0]
    )
    return abs_rank


# ---------------------------------------------------------------------------
# 5. compute_flow_size_estimate
# ---------------------------------------------------------------------------

def compute_flow_size_estimate(
    drift_df: pd.DataFrame,
    etf_aum_df: pd.DataFrame,
    etf_ticker: str,
) -> pd.DataFrame:
    """Estimate the dollar flow needed to correct each constituent's drift.

    For each date *t* and ticker *i*::

        flow_i(t) = drift_i(t) × AUM(t)

    Negative flow = need to BUY; positive flow = need to SELL.

    Parameters
    ----------
    drift_df : pd.DataFrame
        Output of :func:`compute_weight_drift`.
    etf_aum_df : pd.DataFrame
        Output of :func:`data_loader.fetch_etf_aum_history` (long format
        with ``date``, ``etf_ticker``, ``aum_usd`` columns).
    etf_ticker : str
        Which ETF's AUM to use as the multiplier.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ``[date, ticker, drift, flow_estimate_usd, flow_direction]``.

    Notes
    -----
    **Single-ETF scope** — This estimates the flow from ONE ETF only.  In
    practice, multiple ETFs track the same index (e.g. SPY + IVV + VOO all
    track the S&P 500).  The ``data/reference/etf_index_map.csv`` file lists
    all relevant vehicles.  For a complete flow estimate, sum AUM across all
    ETFs tracking the same index::

        total_index_aum = SPY_aum + IVV_aum + VOO_aum  # for SP500
        flow_i(t) = drift_i(t) × total_index_aum

    This would roughly triple the SP500 flow estimate relative to SPY alone.
    For SP400, the primary ETFs are MDY ($18.6B) and IJH ($62.4B); using
    only MDY understates total flow pressure by ~77 %.

    **AUM proxy quality** — As noted in :func:`data_loader.fetch_etf_aum_history`,
    the yfinance AUM is a directional proxy.  The error on absolute dollar
    flow estimates may be ±20–40 % relative to true NAV × shares data.
    """
    # Extract AUM series for the given ETF
    etf_aum_series = (
        etf_aum_df[etf_aum_df["etf_ticker"] == etf_ticker]
        .set_index("date")["aum_usd"]
        .sort_index()
    )

    records = []
    for date_idx in drift_df.index:
        # Find the closest available AUM for this date (forward-fill)
        aum_at_date = etf_aum_series.asof(date_idx)
        if pd.isna(aum_at_date):
            continue

        for ticker in drift_df.columns:
            drift_val = drift_df.at[date_idx, ticker]
            if pd.isna(drift_val):
                continue
            flow_usd = drift_val * aum_at_date
            direction = "SELL" if flow_usd > 0 else "BUY"
            records.append(
                {
                    "date": date_idx,
                    "ticker": ticker,
                    "drift": drift_val,
                    "flow_estimate_usd": flow_usd,
                    "flow_direction": direction,
                }
            )

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    logger.info(
        "Flow size estimates computed: %d records for ETF '%s'.",
        len(df), etf_ticker,
    )
    return df


# ---------------------------------------------------------------------------
# 6. flag_actionable_drift
# ---------------------------------------------------------------------------

def flag_actionable_drift(
    drift_df: pd.DataFrame,
    drift_zscore_df: pd.DataFrame,
    rebalance_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Combine drift, z-score, rank, and blackout filter into action flags.

    Parameters
    ----------
    drift_df : pd.DataFrame
        Output of :func:`compute_weight_drift`.
    drift_zscore_df : pd.DataFrame
        Output of :func:`compute_drift_zscore`.
    rebalance_calendar : pd.DataFrame
        Output of :func:`data_loader.load_rebalance_calendar`.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ``[date, ticker, drift, drift_zscore, abs_drift_rank,
        signal_strength, blackout_flag, action]``.

    Notes
    -----
    **Action logic**:

    * ``"BUY"``  — drift < −SOFT_THRESHOLD  AND  zscore < −ZSCORE_THRESHOLD
    * ``"SELL"`` — drift > +SOFT_THRESHOLD  AND  zscore > +ZSCORE_THRESHOLD
    * ``"WATCH"``— threshold crossed (soft or hard) but zscore below trigger
    * ``"NONE"`` — no material drift

    ``"BUY"`` and ``"SELL"`` are suppressed to ``"NONE"`` during blackout
    windows (within ±REBALANCE_BLACKOUT_DAYS of a scheduled rebalance).

    **Signal strength formula**::

        signal_strength_i(t) = abs_drift_rank_i(t) × min(|zscore_i(t)|, 4.0)

    This weights names that are both at an extreme z-score AND ranked high
    in cross-sectional absolute drift.  The min(|z|, 4.0) cap prevents a
    single outlier event from dominating the composite score.

    Results are saved to ``data/processed/actionable_drift.csv``.
    """
    abs_rank_df = compute_absolute_drift_rank(drift_df)

    soft_t = config.DRIFT_THRESHOLD_SOFT
    z_t = config.DRIFT_ZSCORE_THRESHOLD
    cap = config.DRIFT_ZSCORE_WINSOR_CAP

    records = []
    all_dates = drift_df.index.intersection(drift_zscore_df.index)

    logger.info("Flagging actionable drift across %d dates...", len(all_dates))

    for date_idx in all_dates:
        blackout = is_blackout_date(date_idx, rebalance_calendar)

        for ticker in drift_df.columns:
            drift_val = drift_df.at[date_idx, ticker]
            zscore_val = drift_zscore_df.at[date_idx, ticker]
            rank_val = abs_rank_df.at[date_idx, ticker]

            if pd.isna(drift_val) or pd.isna(zscore_val):
                continue

            signal_strength = rank_val * min(abs(zscore_val), cap)

            # Determine base action
            if drift_val < -soft_t and zscore_val < -z_t:
                action = "BUY"
            elif drift_val > soft_t and zscore_val > z_t:
                action = "SELL"
            elif abs(drift_val) > soft_t or abs(zscore_val) > z_t:
                action = "WATCH"
            else:
                action = "NONE"

            # Suppress in blackout
            if blackout and action in ("BUY", "SELL"):
                action = "NONE"

            records.append(
                {
                    "date": date_idx,
                    "ticker": ticker,
                    "drift": drift_val,
                    "drift_zscore": zscore_val,
                    "abs_drift_rank": rank_val,
                    "signal_strength": signal_strength,
                    "blackout_flag": blackout,
                    "action": action,
                }
            )

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Persist
    out_path = config.PATHS["actionable_drift"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(
        "Actionable drift saved to %s (%d rows; BUY=%d, SELL=%d, WATCH=%d).",
        out_path, len(df),
        (df["action"] == "BUY").sum(),
        (df["action"] == "SELL").sum(),
        (df["action"] == "WATCH").sum(),
    )
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _run_pipeline(index_name: str) -> None:
    """End-to-end drift computation pipeline (smoke-test entry point)."""
    import sys

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    logger.info("=== Index Drift Pipeline | index=%s ===", index_name)

    # Load constituents
    constituents = load_index_constituents(index_name)
    tickers = constituents["ticker"].tolist()
    logger.info("Universe: %d tickers", len(tickers))

    # Load rebalance calendar
    rebalance_cal = load_rebalance_calendar(index_name)

    # Fetch prices (uses cache if available)
    prices = fetch_constituent_prices(
        tickers, config.BACKTEST_START, config.BACKTEST_END
    )

    # Compute realized weights and drift
    realized_w = compute_realized_weights(prices, constituents)
    drift = compute_weight_drift(realized_w, constituents)
    zscore = compute_drift_zscore(drift)

    # Flag actionable drift
    flags = flag_actionable_drift(drift, zscore, rebalance_cal)

    # ---- Summary table for the most recent date with data ----------------
    latest_date = flags["date"].max()
    latest = flags[flags["date"] == latest_date].copy()
    latest = latest[latest["action"].isin(["BUY", "SELL", "WATCH"])].copy()
    latest = latest.sort_values("signal_strength", ascending=False).head(10)

    print(f"\n{'='*72}")
    print(f"  Drift Summary — {latest_date.date()} | Index: {index_name}")
    print(f"{'='*72}")
    if latest.empty:
        print("  No actionable drift signals on this date.")
    else:
        cols = ["ticker", "drift", "drift_zscore", "signal_strength", "action"]
        print(
            latest[cols]
            .rename(columns={
                "drift": "drift (wt-pts)",
                "drift_zscore": "z-score",
                "signal_strength": "signal_str",
            })
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
    print(f"{'='*72}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index drift computation pipeline smoke test."
    )
    parser.add_argument(
        "--index",
        type=str,
        default=config.UNIVERSE,
        choices=list(config.INDEX_ETF_MAP.keys()),
        help="Index to run the drift pipeline on.",
    )
    args = parser.parse_args()
    _run_pipeline(args.index)
