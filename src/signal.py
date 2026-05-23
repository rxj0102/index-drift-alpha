"""
signal.py — Signal construction, decay analysis, IC, and ETF flow confirmation.

This module transforms raw actionable drift flags into a portfolio-ready
daily signal DataFrame, evaluates signal quality (decay, IC), and overlays
ETF flow confirmation to test the strategy's secondary hypothesis.

Signal construction pipeline
-----------------------------
1. :func:`build_daily_signal`     — Filter + quintile trim → long/short books
2. :func:`compute_signal_turnover` — Measure daily book churn
3. :func:`compute_signal_decay`   — Event study over 1–20 day horizons
4. :func:`compute_information_coefficient` — Spearman IC vs. forward returns
5. :func:`overlay_etf_flow_confirmation`  — Flag flow-confirmed signals

Usage
-----
    from src.signal import build_daily_signal, compute_information_coefficient
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.utils import lag_dataframe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. build_daily_signal
# ---------------------------------------------------------------------------

def build_daily_signal(actionable_drift_df: pd.DataFrame) -> pd.DataFrame:
    """Construct daily long/short books from actionable drift flags.

    Parameters
    ----------
    actionable_drift_df : pd.DataFrame
        Output of :func:`drift_engine.flag_actionable_drift`.  Must contain
        columns ``[date, ticker, action, signal_strength, drift, drift_zscore]``.

    Returns
    -------
    pd.DataFrame
        Long-format signal DataFrame with columns:
        ``[date, ticker, book_side, signal_strength, portfolio_weight,
        drift, drift_zscore]``.

    Notes
    -----
    **Quintile filter** — Within each book side on each date, only the top
    ``SIGNAL_QUINTILE`` (default 80th percentile) by signal_strength are
    kept.  This concentrates the portfolio on the highest-conviction names
    and avoids diluting returns with weak signals.

    **Minimum book size** — If fewer than ``MIN_BOOK_SIZE`` (default 5) names
    qualify for either the long or the short book on a given date, the entire
    day's signal is set to FLAT (no position).  A thin book indicates either
    a calm rebalancing environment (no real drift) or a data quality issue;
    trading on <5 names creates idiosyncratic risk that overwhelms the alpha.

    **Portfolio weights** — Equal-weight within each book side (not weighted
    by signal strength).  Signal strength is used only for rank-filtering;
    equal weighting is more robust out-of-sample because signal_strength
    estimates degrade quickly beyond the top quintile.

    **Look-ahead prevention** — This function does NOT apply the date lag.
    The caller must run :func:`utils.lag_dataframe` on the output before
    passing it to any backtest function.
    """
    signal_records = []

    for date_val, group in actionable_drift_df.groupby("date"):
        long_cands = group[group["action"] == "BUY"].copy()
        short_cands = group[group["action"] == "SELL"].copy()

        # Minimum book size check — applied to the pre-quintile-filter pool.
        # "Qualify" means: pass the BUY/SELL drift+z-score threshold.  If
        # fewer than MIN_BOOK_SIZE names qualify on either side, the day's
        # signal is FLAT regardless of how extreme the qualifying names are.
        # A thin qualifying pool (e.g. only 2 BUY candidates) suggests
        # idiosyncratic noise rather than a broad mechanical drift wave.
        if len(long_cands) < config.MIN_BOOK_SIZE or len(short_cands) < config.MIN_BOOK_SIZE:
            logger.debug(
                "Date %s: thin pre-filter pool (long=%d, short=%d) → FLAT signal.",
                date_val.date() if hasattr(date_val, "date") else date_val,
                len(long_cands), len(short_cands),
            )
            continue

        # Apply quintile filter within each book — keep top SIGNAL_QUINTILE
        # fraction by signal_strength.  This concentrates the book on the
        # highest-conviction names within the qualifying pool.
        q = config.SIGNAL_QUINTILE
        threshold_l = long_cands["signal_strength"].quantile(q)
        threshold_s = short_cands["signal_strength"].quantile(q)
        long_cands = long_cands[long_cands["signal_strength"] >= threshold_l]
        short_cands = short_cands[short_cands["signal_strength"] >= threshold_s]

        # Safety: ensure at least 1 name survives the quintile filter per side
        if long_cands.empty or short_cands.empty:
            continue

        n_long = len(long_cands)
        n_short = len(short_cands)

        for _, row in long_cands.iterrows():
            signal_records.append(
                {
                    "date": date_val,
                    "ticker": row["ticker"],
                    "book_side": "LONG",
                    "signal_strength": row["signal_strength"],
                    "portfolio_weight": 1.0 / n_long,
                    "drift": row["drift"],
                    "drift_zscore": row["drift_zscore"],
                }
            )
        for _, row in short_cands.iterrows():
            signal_records.append(
                {
                    "date": date_val,
                    "ticker": row["ticker"],
                    "book_side": "SHORT",
                    "signal_strength": row["signal_strength"],
                    "portfolio_weight": 1.0 / n_short,
                    "drift": row["drift"],
                    "drift_zscore": row["drift_zscore"],
                }
            )

    signal_df = pd.DataFrame(signal_records)
    if not signal_df.empty:
        signal_df["date"] = pd.to_datetime(signal_df["date"])
        signal_df = signal_df.sort_values(["date", "book_side", "signal_strength"],
                                           ascending=[True, True, False]).reset_index(drop=True)

    n_dates = signal_df["date"].nunique() if not signal_df.empty else 0
    logger.info(
        "Signal built: %d active dates, %d LONG, %d SHORT name-days.",
        n_dates,
        (signal_df["book_side"] == "LONG").sum() if not signal_df.empty else 0,
        (signal_df["book_side"] == "SHORT").sum() if not signal_df.empty else 0,
    )
    return signal_df


# ---------------------------------------------------------------------------
# 2. compute_signal_turnover
# ---------------------------------------------------------------------------

def compute_signal_turnover(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily fraction of names that changed in each book.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Output of :func:`build_daily_signal`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ``[date, long_turnover, short_turnover, combined_turnover]``.
        Values are fractions in [0, 1].

    Notes
    -----
    **Turnover definition** — On date *t*, turnover for the long book is::

        long_turnover(t) = |LONG(t) △ LONG(t−1)| / |LONG(t) ∪ LONG(t−1)|

    where △ is symmetric difference (names that entered or exited) and ∪ is
    union (all names that appeared in either day).

    **Red flag** — Daily turnover exceeding ``config.TURNOVER_RED_FLAG``
    (40 %) is logged as a warning.  At 40 % daily churn, a 10-day holding
    period implies the book is nearly fully refreshed twice per holding
    period.  Combined with 8 bps one-way costs, this produces a daily cost
    drag that erodes the gross alpha.  Investigate by:

    * Widening the z-score threshold (reduce DRIFT_ZSCORE_THRESHOLD).
    * Extending the rolling lookback (increase DRIFT_LOOKBACK_DAYS).
    * Requiring persistence: a name must be flagged on N consecutive days
      before entering the book (not currently implemented here — this is a
      natural extension).

    **Cost drag estimate** — Average combined_turnover × TRANSACTION_COST_BPS
    gives the expected daily cost drag in basis points.
    """
    dates = sorted(signal_df["date"].unique())
    records = []

    prev_long: set[str] = set()
    prev_short: set[str] = set()

    for date_val in dates:
        day = signal_df[signal_df["date"] == date_val]
        curr_long = set(day[day["book_side"] == "LONG"]["ticker"])
        curr_short = set(day[day["book_side"] == "SHORT"]["ticker"])

        def _turnover(curr: set, prev: set) -> float:
            union = curr | prev
            if not union:
                return 0.0
            return len(curr.symmetric_difference(prev)) / len(union)

        lt = _turnover(curr_long, prev_long)
        st = _turnover(curr_short, prev_short)
        combined = 0.5 * lt + 0.5 * st

        if combined > config.TURNOVER_RED_FLAG:
            logger.warning(
                "High turnover on %s: long=%.1f%%, short=%.1f%%, combined=%.1f%%.",
                date_val, lt * 100, st * 100, combined * 100,
            )

        records.append(
            {
                "date": date_val,
                "long_turnover": lt,
                "short_turnover": st,
                "combined_turnover": combined,
            }
        )
        prev_long = curr_long
        prev_short = curr_short

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    avg = df["combined_turnover"].mean()
    cost_drag_bps = avg * config.TRANSACTION_COST_BPS
    logger.info(
        "Turnover computed: avg combined=%.1f%%, estimated daily cost drag=%.2f bps.",
        avg * 100, cost_drag_bps,
    )
    return df


# ---------------------------------------------------------------------------
# 3. compute_signal_decay
# ---------------------------------------------------------------------------

def compute_signal_decay(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """Event study: measure mean forward return at multiple horizons.

    For each LONG signal on date *t*, computes the forward return over
    horizons [1, 3, 5, 10, 15, 20] trading days.  SHORT signal returns are
    sign-inverted so that a positive mean return for both books indicates
    the signal is working.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Output of :func:`build_daily_signal`.
    price_df : pd.DataFrame
        Wide-format adjusted close prices.

    Returns
    -------
    pd.DataFrame
        Decay DataFrame with columns:
        ``[horizon_days, mean_return, std_return, t_stat, p_value, n_signals]``.

    Notes
    -----
    **Optimal holding period** — The horizon with the highest mean_return
    is your empirical optimal holding period.  If returns mean-revert past
    day 10, the forward holding period should be capped there.

    **Position sizing implication** — If the signal decays sharply after
    day 5, a confidence-decay approach is warranted: reduce position size
    by 20 % for each day held beyond day 5 without a fresh signal
    confirmation.  This is discussed in Section 10 of the notebook.

    **Event independence caveat** — Signal events are not independent across
    the same ticker on adjacent days (a name can be in the long book for
    many consecutive days).  The t-statistics reported here are for
    *guidance* only; a proper test requires clustering standard errors by
    ticker and quarter.
    """
    horizons = config.SIGNAL_DECAY_HORIZONS
    price_aligned = price_df.sort_index()
    all_dates = price_aligned.index.tolist()

    records = []
    for horizon in horizons:
        returns_h = []

        for _, row in signal_df.iterrows():
            date_val = row["date"]
            ticker = row["ticker"]
            side = row["book_side"]

            if ticker not in price_aligned.columns:
                continue
            if date_val not in price_aligned.index:
                continue

            t_idx = all_dates.index(date_val)
            t_fwd_idx = t_idx + horizon
            if t_fwd_idx >= len(all_dates):
                continue

            p_entry = price_aligned.at[date_val, ticker]
            p_exit = price_aligned.at[all_dates[t_fwd_idx], ticker]

            if pd.isna(p_entry) or pd.isna(p_exit) or p_entry == 0:
                continue

            ret = (p_exit - p_entry) / p_entry
            if side == "SHORT":
                ret = -ret  # invert so positive = signal working
            returns_h.append(ret)

        arr = np.array(returns_h)
        if len(arr) >= 2:
            t_stat, p_val = stats.ttest_1samp(arr, 0.0)
        else:
            t_stat, p_val = np.nan, np.nan

        records.append(
            {
                "horizon_days": horizon,
                "mean_return": float(np.nanmean(arr)) if len(arr) > 0 else np.nan,
                "std_return": float(np.nanstd(arr)) if len(arr) > 0 else np.nan,
                "t_stat": t_stat,
                "p_value": p_val,
                "n_signals": len(arr),
            }
        )

    decay_df = pd.DataFrame(records)
    logger.info("Signal decay computed for horizons: %s", horizons)
    return decay_df


# ---------------------------------------------------------------------------
# 4. compute_information_coefficient
# ---------------------------------------------------------------------------

def compute_information_coefficient(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizon: int = config.HOLDING_PERIOD_DAYS,
) -> dict:
    """Compute Spearman rank IC between signal_strength and forward returns.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Output of :func:`build_daily_signal`.
    price_df : pd.DataFrame
        Wide-format adjusted close prices.
    horizon : int, optional
        Forward return horizon in trading days.  Default is
        ``config.HOLDING_PERIOD_DAYS`` (10 days).

    Returns
    -------
    dict
        Keys: ``ic_long``, ``ic_short``, ``ic_combined``, ``p_value``,
        ``n_obs``.

    Notes
    -----
    **IC benchmark** — A Spearman IC above ``config.IC_MEANINGFUL_THRESHOLD``
    (0.04, or 4 %) is considered material for a microstructure signal.
    Justification: A pure microstructure signal with 10-day mean reversion
    is noisier than a fundamental factor signal; ICs in the 0.03–0.08 range
    are typical for validated anomaly papers.  An IC < 0.02 that is not
    statistically significant (p > 0.05) should be treated as noise.

    **Spearman vs. Pearson** — Spearman rank correlation is preferred because
    it is robust to the heavy-tailed forward return distribution and to
    outlier signal events caused by corporate actions.

    **Separate long/short ICs** — Computing IC for the long book separately
    from the short book is important: the mechanism (passive buy pressure vs.
    passive sell pressure) is asymmetric in practice.  In liquid large-cap
    markets, short selling costs and borrow availability can impair the short
    book's realised IC relative to the long book's.
    """
    price_aligned = price_df.sort_index()
    all_dates = price_aligned.index.tolist()

    def _compute_ic(df: pd.DataFrame) -> tuple[float, float, int]:
        signals, fwd_rets = [], []
        for _, row in df.iterrows():
            date_val = row["date"]
            ticker = row["ticker"]
            side = row["book_side"]

            if ticker not in price_aligned.columns:
                continue
            if date_val not in price_aligned.index:
                continue

            t_idx = all_dates.index(date_val)
            t_fwd = t_idx + horizon
            if t_fwd >= len(all_dates):
                continue

            p0 = price_aligned.at[date_val, ticker]
            p1 = price_aligned.at[all_dates[t_fwd], ticker]

            if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                continue

            ret = (p1 - p0) / p0
            if side == "SHORT":
                ret = -ret

            signals.append(row["signal_strength"])
            fwd_rets.append(ret)

        if len(signals) < 5:
            return np.nan, np.nan, len(signals)

        ic, pval = stats.spearmanr(signals, fwd_rets)
        return float(ic), float(pval), len(signals)

    long_df = signal_df[signal_df["book_side"] == "LONG"]
    short_df = signal_df[signal_df["book_side"] == "SHORT"]

    ic_long, pval_long, n_long = _compute_ic(long_df)
    ic_short, pval_short, n_short = _compute_ic(short_df)
    ic_combined, pval_combined, n_all = _compute_ic(signal_df)

    result = {
        "ic_long": ic_long,
        "ic_short": ic_short,
        "ic_combined": ic_combined,
        "p_value": pval_combined,
        "n_obs": n_all,
    }

    logger.info(
        "IC | LONG=%.4f | SHORT=%.4f | COMBINED=%.4f (p=%.4f, n=%d).",
        ic_long, ic_short, ic_combined, pval_combined, n_all,
    )
    if not np.isnan(ic_combined) and abs(ic_combined) < config.IC_MEANINGFUL_THRESHOLD:
        logger.warning(
            "Combined IC (%.4f) is below the meaningful threshold (%.2f). "
            "Signal may not be exploitable net of costs.",
            ic_combined, config.IC_MEANINGFUL_THRESHOLD,
        )
    return result


# ---------------------------------------------------------------------------
# 5. overlay_etf_flow_confirmation
# ---------------------------------------------------------------------------

def overlay_etf_flow_confirmation(
    signal_df: pd.DataFrame,
    flow_df: pd.DataFrame,
) -> pd.DataFrame:
    """Flag signals where estimated ETF flow direction confirms the signal.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Output of :func:`build_daily_signal`.
    flow_df : pd.DataFrame
        Output of :func:`drift_engine.compute_flow_size_estimate`.

    Returns
    -------
    pd.DataFrame
        A copy of ``signal_df`` with two additional columns:
        ``flow_confirmed`` (bool) and ``flow_magnitude_usd`` (float).

    Notes
    -----
    **Confirmation logic**:

    * LONG signal + flow_direction == "BUY"  → ``flow_confirmed = True``
    * SHORT signal + flow_direction == "SELL" → ``flow_confirmed = True``
    * Otherwise → ``flow_confirmed = False``

    **Secondary hypothesis** — Signals where estimated ETF flow *confirms*
    the drift direction should have a higher realised IC than unconfirmed
    signals.  The intuition is that flow confirmation acts as a second
    indicator that the passive rebalancing pressure is building.  The
    notebook (Section 5 and Section 7) tests this IC split explicitly.

    If the IC difference (confirmed vs. unconfirmed) is not statistically
    significant, it suggests that:

    1. The AUM proxy is too noisy to provide incremental information, or
    2. The market is already pricing in the anticipated flow, making the
       flow signal redundant given the drift signal.
    """
    if flow_df.empty:
        logger.warning("flow_df is empty — cannot compute flow confirmation.")
        signal_out = signal_df.copy()
        signal_out["flow_confirmed"] = False
        signal_out["flow_magnitude_usd"] = np.nan
        return signal_out

    # Build a lookup: (date, ticker) → (flow_direction, flow_estimate_usd)
    flow_idx = flow_df.set_index(["date", "ticker"])

    confirmed_flags = []
    magnitude_usd = []

    for _, row in signal_df.iterrows():
        key = (row["date"], row["ticker"])
        if key in flow_idx.index:
            flow_row = flow_idx.loc[key]
            flow_dir = flow_row["flow_direction"]
            flow_usd = flow_row["flow_estimate_usd"]
        else:
            flow_dir = None
            flow_usd = np.nan

        if row["book_side"] == "LONG" and flow_dir == "BUY":
            confirmed = True
        elif row["book_side"] == "SHORT" and flow_dir == "SELL":
            confirmed = True
        else:
            confirmed = False

        confirmed_flags.append(confirmed)
        magnitude_usd.append(flow_usd)

    signal_out = signal_df.copy()
    signal_out["flow_confirmed"] = confirmed_flags
    signal_out["flow_magnitude_usd"] = magnitude_usd

    n_conf = sum(confirmed_flags)
    n_total = len(confirmed_flags)
    logger.info(
        "Flow confirmation overlay: %d/%d signals confirmed (%.1f%%).",
        n_conf, n_total, 100 * n_conf / max(n_total, 1),
    )
    return signal_out
