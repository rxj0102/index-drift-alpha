"""
backtest.py — Long/short backtests, event studies, and performance analytics.

All backtest functions are self-contained pandas/numpy — no external
portfolio or backtest libraries are required.

Architecture notes
------------------
* Every function that consumes a ``signal_df`` assumes it has already been
  processed through :func:`utils.lag_dataframe` by the caller.  If you
  see suspiciously high Sharpe ratios, the first thing to check is whether
  the lag was applied.

* Transaction costs are deducted as a one-way cost applied at entry and
  exit of each holding period:
    net_return = gross_return − 2 × (TRANSACTION_COST_BPS / 10_000) × turnover_rate

* Holding period returns are distributed evenly across the holding window
  (simple daily spreading) to avoid return lumping at fixed intervals,
  which would distort the daily Sharpe ratio calculation.

Usage
-----
    from src.backtest import run_long_short_backtest, compute_performance_metrics
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.data_loader import is_blackout_date

logger = logging.getLogger(__name__)

_BPS = 1e-4  # 1 basis point in decimal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _forward_return(
    price_df: pd.DataFrame,
    ticker: str,
    entry_date: pd.Timestamp,
    horizon: int,
) -> Optional[float]:
    """Compute simple buy-and-hold return from entry_date over horizon days."""
    dates = price_df.index.tolist()
    if entry_date not in dates:
        return None
    t0 = dates.index(entry_date)
    t1 = t0 + horizon
    if t1 >= len(dates):
        return None
    p0 = price_df.at[entry_date, ticker]
    p1 = price_df.at[dates[t1], ticker]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return None
    return (p1 - p0) / p0


def _compute_drawdown(cumret_series: pd.Series) -> pd.Series:
    """Drawdown (0 to -1) from an equity curve that starts at 1.0."""
    rolling_max = cumret_series.cummax()
    dd = (cumret_series - rolling_max) / rolling_max.clip(lower=1e-10)
    return dd


# ---------------------------------------------------------------------------
# 1. run_long_short_backtest
# ---------------------------------------------------------------------------

def run_long_short_backtest(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    rebalance_calendar: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Simulate a daily-rebalanced long/short portfolio.

    For each date with an active signal:

    * **LONG book return** = equal-weight mean return of long tickers
      over the next ``HOLDING_PERIOD_DAYS`` days.
    * **SHORT book return** = equal-weight mean return of short tickers
      (sign-inverted).
    * **LS return** = 0.5 × long_return + 0.5 × short_return.
    * **Net return** = gross LS return − transaction cost drag.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Output of :func:`signal.build_daily_signal`, **already lagged**
        via :func:`utils.lag_dataframe`.
    price_df : pd.DataFrame
        Wide-format adjusted close prices.
    rebalance_calendar : pd.DataFrame, optional
        Used to label blackout dates in the output.  Pass ``None`` to skip.

    Returns
    -------
    pd.DataFrame
        Backtest results with columns:
        ``[date, long_return, short_return, ls_return, tc_drag,
        net_ls_return, cumulative_pnl, drawdown, regime_label]``.

    Notes
    -----
    **Holding period return spreading** — Rather than booking the full
    10-day holding return on the entry date and zero on subsequent days,
    returns are spread evenly: ``daily_contribution = period_return / horizon``.
    This produces a smoother PnL series and a more accurate daily Sharpe
    ratio estimate.  It does NOT affect the total return over any full
    holding period.

    **Transaction cost model** — At 8 bps one-way (default):

        tc_drag_per_trade = 2 × 8 bps = 16 bps (round trip)

    If a name persists in the book across consecutive days (zero turnover),
    no cost is incurred for the re-entry.  Cost is only charged on the
    *fraction* of names that entered or exited since the prior day.

    **Gross Sharpe inflation check** — If the reported Sharpe (gross of
    costs) exceeds 2.0 for the SP400 universe over 2015–2023, suspect that
    the signal lag was not applied.  This strategy's theoretical gross
    Sharpe should be in the 0.8–1.4 range for the SP400 universe.
    """
    price_aligned = price_df.sort_index()
    all_dates = price_aligned.index.tolist()

    # Aggregate signal by date
    dates_with_signal = sorted(signal_df["date"].unique())

    # Track daily contributions (key = date, value = list of accumulated returns)
    daily_contributions: dict[pd.Timestamp, list[float]] = {d: [] for d in all_dates}
    daily_tc: dict[pd.Timestamp, float] = {d: 0.0 for d in all_dates}

    prev_long_tickers: set[str] = set()
    prev_short_tickers: set[str] = set()

    for entry_date in dates_with_signal:
        day_sig = signal_df[signal_df["date"] == entry_date]
        long_tickers = list(day_sig[day_sig["book_side"] == "LONG"]["ticker"])
        short_tickers = list(day_sig[day_sig["book_side"] == "SHORT"]["ticker"])

        if not long_tickers and not short_tickers:
            continue

        # ---- Transaction cost based on turnover -------------------------
        def _turnover(curr: set, prev: set) -> float:
            union = set(curr) | prev
            return len(set(curr).symmetric_difference(prev)) / max(len(union), 1)

        lt = _turnover(set(long_tickers), prev_long_tickers)
        st = _turnover(set(short_tickers), prev_short_tickers)
        combined_to = 0.5 * lt + 0.5 * st
        tc = 2 * (config.TRANSACTION_COST_BPS * _BPS) * combined_to

        # ---- Compute forward returns for each name ----------------------
        horizon = config.HOLDING_PERIOD_DAYS

        long_rets = []
        for t in long_tickers:
            if t in price_aligned.columns:
                r = _forward_return(price_aligned, t, entry_date, horizon)
                if r is not None:
                    long_rets.append(r)

        short_rets = []
        for t in short_tickers:
            if t in price_aligned.columns:
                r = _forward_return(price_aligned, t, entry_date, horizon)
                if r is not None:
                    short_rets.append(-r)  # invert for short

        if not long_rets or not short_rets:
            prev_long_tickers = set(long_tickers)
            prev_short_tickers = set(short_tickers)
            continue

        long_ret = float(np.mean(long_rets))
        short_ret = float(np.mean(short_rets))
        ls_ret = 0.5 * long_ret + 0.5 * short_ret
        net_ls_ret = ls_ret - tc

        # Spread the holding period return evenly across horizon days
        entry_t_idx = all_dates.index(entry_date)
        for k in range(horizon):
            fwd_date_idx = entry_t_idx + k
            if fwd_date_idx < len(all_dates):
                fwd_date = all_dates[fwd_date_idx]
                daily_contributions[fwd_date].append(net_ls_ret / horizon)
                if k == 0:
                    daily_tc[fwd_date] += tc

        prev_long_tickers = set(long_tickers)
        prev_short_tickers = set(short_tickers)

    # ---- Aggregate to daily series -------------------------------------
    records = []
    for d in all_dates:
        contribs = daily_contributions.get(d, [])
        net_ret = float(np.sum(contribs)) if contribs else 0.0
        tc_d = daily_tc.get(d, 0.0)

        regime = "ACTIVE"
        if rebalance_calendar is not None and is_blackout_date(d, rebalance_calendar):
            regime = "BLACKOUT"

        records.append(
            {
                "date": d,
                "net_ls_return": net_ret,
                "tc_drag": tc_d,
            }
        )

    bt = pd.DataFrame(records)
    bt["date"] = pd.to_datetime(bt["date"])
    bt = bt.set_index("date").sort_index()

    # For display, approximate long/short split
    bt["ls_return"] = bt["net_ls_return"] + bt["tc_drag"]
    bt["long_return"] = bt["ls_return"] / 2
    bt["short_return"] = bt["ls_return"] / 2

    # Cumulative PnL (assuming 1 dollar NAV)
    bt["cumulative_pnl"] = (1 + bt["net_ls_return"]).cumprod()
    bt["drawdown"] = _compute_drawdown(bt["cumulative_pnl"])

    # Regime label
    if rebalance_calendar is not None:
        bt["regime_label"] = [
            "BLACKOUT" if is_blackout_date(d, rebalance_calendar) else "ACTIVE"
            for d in bt.index
        ]
    else:
        bt["regime_label"] = "ACTIVE"

    bt = bt.reset_index()
    bt = bt[["date", "long_return", "short_return", "ls_return",
             "tc_drag", "net_ls_return", "cumulative_pnl",
             "drawdown", "regime_label"]]

    logger.info(
        "Backtest complete: %d trading days, final cumPnL=%.4f.",
        len(bt), bt["cumulative_pnl"].iloc[-1] if len(bt) > 0 else 0,
    )
    return bt


# ---------------------------------------------------------------------------
# 2. run_event_study_backtest
# ---------------------------------------------------------------------------

def run_event_study_backtest(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """Event-study backtest: independent event × horizon return table.

    Each (ticker, signal_date) pair is treated as an independent event.
    Returns are computed at [1, 3, 5, 10, 20] day horizons.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Output of :func:`signal.build_daily_signal`, lagged.
    price_df : pd.DataFrame
        Wide-format adjusted close prices.

    Returns
    -------
    pd.DataFrame
        Event-level DataFrame with columns:
        ``[event_date, ticker, book_side, signal_strength,
        ret_1d, ret_3d, ret_5d, ret_10d, ret_20d,
        flow_confirmed, index_name, sector]``.

    Notes
    -----
    The event-study format is better suited to academic presentation than
    a time-series backtest because:

    1. It preserves the full cross-sectional distribution of returns, enabling
       quantile analysis (do top-quintile signals outperform bottom-quintile?).
    2. It avoids overlapping return windows (which complicate standard error
       calculation in time-series tests).
    3. It enables sub-group analysis: sector, index, flow-confirmed vs. not.

    Note that observations are **not** independent across adjacent dates for
    the same ticker (a name can be in the long book for many consecutive days).
    Standard errors should be clustered by (ticker, quarter) for rigorous
    inference.
    """
    horizons = [1, 3, 5, 10, 20]
    price_aligned = price_df.sort_index()
    records = []

    for _, row in signal_df.iterrows():
        event_date = row["date"]
        ticker = row["ticker"]
        side = row["book_side"]

        if ticker not in price_aligned.columns:
            continue

        ret_vals = {}
        for h in horizons:
            r = _forward_return(price_aligned, ticker, event_date, h)
            if r is not None and side == "SHORT":
                r = -r
            ret_vals[f"ret_{h}d"] = r

        records.append(
            {
                "event_date": event_date,
                "ticker": ticker,
                "book_side": side,
                "signal_strength": row["signal_strength"],
                **ret_vals,
                "flow_confirmed": row.get("flow_confirmed", False),
                "index_name": config.UNIVERSE,
                "sector": None,   # Populated from constituents in notebook
            }
        )

    event_df = pd.DataFrame(records)
    if not event_df.empty:
        event_df["event_date"] = pd.to_datetime(event_df["event_date"])
    logger.info("Event study: %d events across %d tickers.", len(event_df),
                event_df["ticker"].nunique() if not event_df.empty else 0)
    return event_df


# ---------------------------------------------------------------------------
# 3. run_midcap_vs_largecap_split
# ---------------------------------------------------------------------------

def run_midcap_vs_largecap_split(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    constituents_df: pd.DataFrame,
) -> dict:
    """Run separate backtests for SP500 (large-cap) and SP400 (mid-cap).

    Parameters
    ----------
    signal_df : pd.DataFrame
        Combined signal DataFrame covering both indices, lagged.
    price_df : pd.DataFrame
        Wide-format adjusted close prices.
    constituents_df : pd.DataFrame
        Must contain ``ticker`` and ``index_name`` columns.

    Returns
    -------
    dict
        Keys ``"SP500"`` and ``"SP400"``, each containing a ``metrics``
        sub-dict from :func:`compute_performance_metrics`.

    Notes
    -----
    **Core hypothesis test** — The strategy's central claim is that mid-cap
    (SP400) should generate materially higher risk-adjusted returns than
    large-cap (SP500) because:

    1. HFT front-running of passive ETF flows is more prevalent in the
       SP500, compressing the available alpha window.
    2. SP400 names have lower average daily volume, meaning the passive
       rebalancing flow represents a larger fraction of daily liquidity,
       creating larger price impact that a patient contrarian can capture.
    3. SP400 index funds are fewer and smaller, so the reconstitution
       signal is less widely followed.

    **Interpretation benchmark** — If the SP400 Sharpe is not at least
    30 % higher than the SP500 Sharpe, the HFT crowding thesis is not
    supported by the data.  A plausible alternative interpretation is that
    both universes are equally crowded, or that the 8 bps cost assumption
    is too optimistic for SP400 and understates the true drag.

    If SP500 Sharpe > SP400 Sharpe, one explanation is that larger ETF AUM
    tracking the SP500 creates *more* predictable and *larger* flows,
    producing a stronger mechanical signal despite higher front-running
    intensity.
    """
    # Split signal by index constituent membership
    sp500_tickers = set(
        constituents_df[constituents_df["index_name"] == "SP500"]["ticker"]
    )
    sp400_tickers = set(
        constituents_df[constituents_df["index_name"] == "SP400"]["ticker"]
    )

    sp500_sig = signal_df[signal_df["ticker"].isin(sp500_tickers)].copy()
    sp400_sig = signal_df[signal_df["ticker"].isin(sp400_tickers)].copy()

    results = {}
    for label, subset_sig in [("SP500", sp500_sig), ("SP400", sp400_sig)]:
        if subset_sig.empty:
            logger.warning("No signals for %s — skipping backtest.", label)
            results[label] = {"backtest_df": pd.DataFrame(), "metrics": {}}
            continue
        bt = run_long_short_backtest(subset_sig, price_df)
        metrics = compute_performance_metrics(bt, label=label)
        results[label] = {"backtest_df": bt, "metrics": metrics}

    # Log the Sharpe comparison
    sp500_sharpe = results.get("SP500", {}).get("metrics", {}).get("sharpe_ratio", np.nan)
    sp400_sharpe = results.get("SP400", {}).get("metrics", {}).get("sharpe_ratio", np.nan)
    if not np.isnan(sp500_sharpe) and not np.isnan(sp400_sharpe) and sp500_sharpe != 0:
        ratio = sp400_sharpe / sp500_sharpe
        logger.info(
            "Sharpe ratio — SP500: %.3f | SP400: %.3f | Ratio: %.2f× "
            "(HFT thesis requires ≥1.30×)",
            sp500_sharpe, sp400_sharpe, ratio,
        )
        if ratio < 1.30:
            logger.warning(
                "SP400/SP500 Sharpe ratio %.2f× < 1.30 threshold. "
                "HFT crowding thesis not strongly supported.",
                ratio,
            )
    return results


# ---------------------------------------------------------------------------
# 4. compute_performance_metrics
# ---------------------------------------------------------------------------

def compute_performance_metrics(
    backtest_df: pd.DataFrame,
    label: str = "",
) -> dict:
    """Compute annualised performance metrics for a backtest result.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of :func:`run_long_short_backtest`.
    label : str, optional
        Label for the printed table (e.g., ``"SP400"``).

    Returns
    -------
    dict
        Keys: ``annualised_return``, ``sharpe_ratio``, ``max_drawdown``,
        ``drawdown_peak_date``, ``drawdown_trough_date``, ``calmar_ratio``,
        ``sortino_ratio``, ``win_rate``, ``avg_gross_return``,
        ``avg_net_return``, ``tc_drag_pct_gross``.
    """
    if backtest_df.empty:
        logger.warning("Empty backtest_df — cannot compute metrics.")
        return {}

    ret = backtest_df["net_ls_return"].copy()
    gross_ret = backtest_df["ls_return"].copy()
    cum_pnl = backtest_df["cumulative_pnl"].copy()

    trading_days = 252
    n_days = len(ret)

    annualised_return = float((cum_pnl.iloc[-1]) ** (trading_days / n_days) - 1)
    daily_std = float(ret.std())
    sharpe = float((ret.mean() / daily_std) * np.sqrt(trading_days)) if daily_std > 0 else 0.0

    # Maximum drawdown
    max_dd = float(backtest_df["drawdown"].min())
    trough_date = backtest_df.loc[backtest_df["drawdown"].idxmin(), "date"]
    peak_mask = cum_pnl[:backtest_df["drawdown"].idxmin() + 1]
    peak_date = backtest_df.loc[peak_mask.idxmax(), "date"] if not peak_mask.empty else pd.NaT

    calmar = float(-annualised_return / max_dd) if max_dd < 0 else np.inf

    # Sortino ratio (downside deviation)
    downside = ret[ret < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else np.nan
    sortino = (
        float((ret.mean() / downside_std) * np.sqrt(trading_days))
        if downside_std and downside_std > 0 else np.nan
    )

    # Win rate and average returns
    win_rate = float((ret > 0).mean())
    avg_gross = float(gross_ret.mean())
    avg_net = float(ret.mean())
    tc_pct = float(abs(avg_gross - avg_net) / abs(avg_gross) * 100) if avg_gross != 0 else 0.0

    metrics = {
        "label": label,
        "annualised_return": annualised_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "drawdown_peak_date": peak_date,
        "drawdown_trough_date": trough_date,
        "calmar_ratio": calmar,
        "sortino_ratio": sortino,
        "win_rate": win_rate,
        "avg_gross_return": avg_gross,
        "avg_net_return": avg_net,
        "tc_drag_pct_gross": tc_pct,
        "n_trading_days": n_days,
    }

    # Print formatted table
    header = f"  Performance Metrics — {label}  " if label else "  Performance Metrics  "
    sep = "─" * max(len(header), 52)
    print(f"\n{sep}")
    print(header)
    print(sep)
    print(f"  Annualised Return    : {annualised_return:>8.2%}")
    print(f"  Sharpe Ratio         : {sharpe:>8.3f}")
    print(f"  Maximum Drawdown     : {max_dd:>8.2%}")
    print(f"  Calmar Ratio         : {calmar:>8.3f}")
    print(f"  Sortino Ratio        : {sortino:>8.3f}")
    print(f"  Win Rate             : {win_rate:>8.1%}")
    print(f"  Avg Gross Daily Ret  : {avg_gross:>8.4%}")
    print(f"  Avg Net Daily Ret    : {avg_net:>8.4%}")
    print(f"  TC Drag % of Gross   : {tc_pct:>8.1f}%")
    print(f"  Trading Days         : {n_days:>8d}")
    print(f"{sep}\n")

    return metrics


# ---------------------------------------------------------------------------
# 5. run_rebalance_window_analysis
# ---------------------------------------------------------------------------

def run_rebalance_window_analysis(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    rebalance_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Compute mean LS returns in PRE / NEAR / POST windows around rebalances.

    For each rebalance event, the function assigns each signal date to one
    of three sub-periods defined in ``config.py``:

    * **PRE** :  [−20, −6] trading days before rebalance date
    * **NEAR**:  [−5,  −1] trading days (blackout zone)
    * **POST**:  [+1, +10] trading days after rebalance

    Parameters
    ----------
    signal_df : pd.DataFrame
        Output of :func:`signal.build_daily_signal`, lagged.
    price_df : pd.DataFrame
        Wide-format adjusted close prices.
    rebalance_calendar : pd.DataFrame
        Output of :func:`data_loader.load_rebalance_calendar`.

    Returns
    -------
    pd.DataFrame
        Window analysis DataFrame with columns:
        ``[rebalance_date, period, mean_ls_return, n_signals,
        mean_signal_strength]``.

    Notes
    -----
    **Expected findings**:

    * **PRE window** should show the strongest LS return: drift is building,
      passive funds are not yet trading, and the signal has not yet been
      front-run by every other participant.
    * **NEAR window** (blackout zone) should show attenuated or noisy returns:
      other front-runners are competing for the same flow, increasing execution
      costs and reducing net alpha.
    * **POST window** tests for mean reversion after rebalancing completes.
      A positive POST window return suggests the market *overshoots* on
      rebalance day, creating a continued momentum effect.  A negative POST
      window return suggests rebalancing corrects the drift completely.

    If the PRE window does not show a statistically significant positive
    return, this strategy's core thesis is not supported by the data for that
    index and time period.
    """
    price_aligned = price_df.sort_index()
    all_dates = sorted(price_aligned.index.tolist())

    def _offset_date(rebalance_date: pd.Timestamp, offset_days: int) -> pd.Timestamp:
        """Offset a date by trading days (not calendar days)."""
        if rebalance_date not in price_aligned.index:
            approx = price_aligned.index.asof(rebalance_date)
            if pd.isna(approx):
                return rebalance_date + pd.Timedelta(days=int(offset_days * 7 / 5))
            rebalance_date = approx
        r_idx = all_dates.index(rebalance_date)
        target_idx = max(0, min(len(all_dates) - 1, r_idx + offset_days))
        return all_dates[target_idx]

    windows = {
        "PRE":  (config.WINDOW_PRE_START,  config.WINDOW_PRE_END),
        "NEAR": (config.WINDOW_NEAR_START, config.WINDOW_NEAR_END),
        "POST": (config.WINDOW_POST_START, config.WINDOW_POST_END),
    }

    records = []
    for _, reb_row in rebalance_calendar.iterrows():
        reb_date = reb_row["rebalance_date"]

        for period_name, (start_off, end_off) in windows.items():
            window_start = _offset_date(reb_date, start_off)
            window_end = _offset_date(reb_date, end_off)

            period_signals = signal_df[
                (signal_df["date"] >= window_start) &
                (signal_df["date"] <= window_end)
            ]

            if period_signals.empty:
                records.append(
                    {
                        "rebalance_date": reb_date,
                        "period": period_name,
                        "mean_ls_return": np.nan,
                        "n_signals": 0,
                        "mean_signal_strength": np.nan,
                    }
                )
                continue

            # Compute LS return for the holding period starting at window_start
            ls_returns = []
            for _, sig_row in period_signals.iterrows():
                ticker = sig_row["ticker"]
                side = sig_row["book_side"]
                sig_date = sig_row["date"]

                r = _forward_return(price_aligned, ticker, sig_date,
                                    config.HOLDING_PERIOD_DAYS)
                if r is None:
                    continue
                if side == "SHORT":
                    r = -r
                ls_returns.append(r)

            records.append(
                {
                    "rebalance_date": reb_date,
                    "period": period_name,
                    "mean_ls_return": float(np.mean(ls_returns)) if ls_returns else np.nan,
                    "n_signals": len(ls_returns),
                    "mean_signal_strength": float(period_signals["signal_strength"].mean()),
                }
            )

    window_df = pd.DataFrame(records)
    window_df["rebalance_date"] = pd.to_datetime(window_df["rebalance_date"])
    window_df = window_df.sort_values(["rebalance_date", "period"]).reset_index(drop=True)

    logger.info(
        "Rebalance window analysis: %d events × 3 periods = %d rows.",
        rebalance_calendar.shape[0], len(window_df),
    )
    return window_df
