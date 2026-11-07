"""
utils.py — Look-ahead-bias guards and general-purpose helpers.

These two functions are the safety layer between signal generation and
backtest functions.  Every signal DataFrame **must** be lagged before it
touches any backtest or performance evaluation routine.

Design notes
------------
* ``lag_dataframe`` shifts the **entire** DataFrame forward by ``periods``
  trading days.  Day t's signal, which is computed from prices on day t,
  is only available to the backtest on day t+1.  Forgetting this shift
  is the most common source of inflated backtest Sharpe ratios in academic
  replication studies.

* ``winsorize_df`` operates column-by-column so that outlier events in one
  ticker do not contaminate the cross-sectional quantile bounds of others.
  The drift z-score distribution is approximately normal in calm markets but
  develops heavy tails around corporate actions and index events; winsorising
  at ±4σ removes roughly 0.003 % of observations under normality.

Usage
-----
    from src.utils import lag_dataframe, winsorize_df

    signal_clean = lag_dataframe(raw_signal_df, periods=1)
    drift_clipped = winsorize_df(drift_zscore_df, lower=0.001, upper=0.999)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# lag_dataframe
# ---------------------------------------------------------------------------

def lag_dataframe(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Shift all columns forward by *periods* rows to prevent look-ahead bias.

    This is the single most important function in the library.  It **must**
    be called on any signal DataFrame before it is passed to a backtest
    function.

    Why it matters
    --------------
    A signal computed from prices on day *t* is not actionable until the
    open of day *t + 1* at the earliest.  Failing to lag inflates the
    apparent Sharpe ratio because the backtest effectively "knows" the
    return before it has to commit to the trade.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame with a ``DatetimeIndex`` and one column per
        asset.  All columns are shifted uniformly.
    periods : int, optional
        Number of rows to shift forward.  Default is 1 (one trading day).
        Set to 0 only in unit tests where the lag is applied externally.

    Returns
    -------
    pd.DataFrame
        Lagged DataFrame with the same index and columns.  The first
        *periods* rows are NaN (no prior signal exists).

    Notes
    -----
    * The function preserves the original index so that date alignment with
      price DataFrames remains correct after the shift.
    * The lagged DataFrame is a copy — the original ``df`` is unmodified.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"A": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3))
    >>> lag_dataframe(df, periods=1)
                  A
    2020-01-01  NaN
    2020-01-02  1.0
    2020-01-03  2.0
    """
    if periods < 0:
        raise ValueError(f"periods must be >= 0, got {periods}.")
    if periods == 0:
        logger.warning("lag_dataframe called with periods=0 — no lag applied.")
        return df.copy()

    # NOTE: shifts ROWS. For long-format signals (many tickers per date) use
    # ``lag_signal_dates`` instead, otherwise the signal is effectively unlagged.
    lagged = df.shift(periods)
    logger.debug(
        "lag_dataframe: shifted %d columns by %d period(s); "
        "first %d rows are NaN.",
        df.shape[1], periods, periods,
    )
    return lagged


# ---------------------------------------------------------------------------
# winsorize_df
# ---------------------------------------------------------------------------

def lag_signal_dates(
    signal_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    periods: int = 1,
    date_col: str = "date",
) -> pd.DataFrame:
    """Delay a long-format signal by ``periods`` trading days (look-ahead guard).

    ``lag_dataframe`` shifts *rows*, which is only correct for a wide frame with
    one row per date.  A long-format signal (many tickers per date) must instead
    have its ``date`` column moved forward on the trading calendar, so a signal
    computed from the close of day *t* is first tradable at day *t + periods*.

    Rows whose lagged date falls beyond the last trading date are dropped.
    """
    if periods < 0:
        raise ValueError(f"periods must be >= 0, got {periods}.")
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(trading_dates).unique()))
    out = signal_df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    pos = cal.get_indexer(out[date_col])
    if (pos < 0).any():
        raise ValueError("signal dates not found in trading_dates calendar")
    new_pos = pos + periods
    keep = new_pos < len(cal)
    out = out.loc[keep].copy()
    out[date_col] = cal[new_pos[keep]]
    return out.reset_index(drop=True)


def winsorize_df(
    df: pd.DataFrame,
    lower: float = 0.005,
    upper: float = 0.995,
) -> pd.DataFrame:
    """Clip each column at the *lower* and *upper* quantile bounds.

    Used primarily in :func:`drift_engine.compute_drift_zscore` to remove
    extreme outlier observations caused by corporate actions (stock splits,
    spin-offs, sudden float restatements) or data errors.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame.  Each column is winsorised independently,
        which is correct because different tickers can have very different
        distributional scales.
    lower : float, optional
        Lower quantile bound (0–1).  Values below this quantile are clipped
        up.  Default 0.005 (0.5th percentile).
    upper : float, optional
        Upper quantile bound (0–1).  Values above this quantile are clipped
        down.  Default 0.995 (99.5th percentile).

    Returns
    -------
    pd.DataFrame
        Winsorised copy of ``df`` with identical shape, index, and columns.

    Notes
    -----
    * Quantile bounds are computed **per column** (per ticker), not across
      the entire panel.  This prevents a single volatile ticker from
      distorting the clip bounds for every other name.
    * NaN values are excluded from quantile calculation and preserved in the
      output (``pd.DataFrame.quantile`` uses ``numeric_only=True`` by default
      and ignores NaN).
    * For z-score inputs, the caller in ``drift_engine`` additionally hard-
      caps at ±4.0 after winsorisation as a belt-and-suspenders approach.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"A": list(range(100))})
    >>> clipped = winsorize_df(df, lower=0.05, upper=0.95)
    >>> clipped["A"].min(), clipped["A"].max()
    (4.95, 94.05)
    """
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError(
            f"Require 0 <= lower < upper <= 1; got lower={lower}, upper={upper}."
        )

    lower_bounds = df.quantile(lower)
    upper_bounds = df.quantile(upper)
    clipped = df.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    logger.debug(
        "winsorize_df: clipped %d×%d frame at [%.3f, %.3f] quantiles.",
        df.shape[0], df.shape[1], lower, upper,
    )
    return clipped
