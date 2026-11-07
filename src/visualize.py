"""
visualize.py — Publication-quality charts for the drift strategy.

All charts are saved to ``outputs/`` at 300 DPI, seaborn-whitegrid style.
All functions return the ``matplotlib.figure.Figure`` object so callers can
further customise or display inline in notebooks.

Chart inventory
---------------
1. :func:`plot_drift_heatmap`                — Figure 1: mechanism overview
2. :func:`plot_drift_timeseries_single`      — Single-ticker dual-axis drill-down
3. :func:`plot_signal_decay_curve`           — Optimal holding period chart
4. :func:`plot_cumulative_pnl_comparison`    — SP400 vs SP500 LS strategy
5. :func:`plot_flow_confirmation_ic_split`   — IC split by flow confirmation
6. :func:`plot_rebalance_window_returns`     — PRE/NEAR/POST window returns
7. :func:`plot_weight_drift_vs_forward_return_scatter` — Regression validation

Usage
-----
    from src.visualize import plot_drift_heatmap, plot_cumulative_pnl_comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import config

matplotlib.rcParams["figure.dpi"] = 150
sns.set_theme(style="whitegrid", palette="muted")

logger = logging.getLogger(__name__)

_OUTPUTS = config.PATHS["outputs"]
_OUTPUTS.mkdir(parents=True, exist_ok=True)

_BLUE   = "#2166AC"
_RED    = "#D6604D"
_GREY   = "#AAAAAA"
_ORANGE = "#F4A582"
_DARK   = "#1a1a2e"


def _save(fig: plt.Figure, filename: str) -> Path:
    out = _OUTPUTS / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    logger.info("Chart saved to %s", out)
    return out


# ---------------------------------------------------------------------------
# 1. plot_drift_heatmap
# ---------------------------------------------------------------------------

def plot_drift_heatmap(
    drift_df: pd.DataFrame,
    index_name: str,
    n_tickers: int = 30,
) -> plt.Figure:
    """Heatmap of weight drift for the top N most-drifting constituents.

    This is Figure 1 — the visual that makes the mechanism immediately
    obvious.  Red cells (positive drift) indicate overweight names facing
    sell pressure; blue cells (negative drift) indicate underweight names
    facing buy pressure.

    Parameters
    ----------
    drift_df : pd.DataFrame
        Wide-format drift DataFrame: index=dates, columns=tickers.
    index_name : str
        Used in the chart title.
    n_tickers : int, optional
        Number of tickers to display (sorted by max |drift| over the period).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Reduce to weekly frequency to avoid overplotting dates
    drift_weekly = drift_df.resample("W").last()

    # Select top N tickers by average absolute drift
    avg_abs_drift = drift_df.abs().mean(axis=0).sort_values(ascending=False)
    top_tickers = avg_abs_drift.head(n_tickers).index.tolist()
    plot_data = drift_weekly[top_tickers].T  # tickers × dates

    # Sort tickers by max |drift| for visual clarity
    sort_order = drift_df[top_tickers].abs().max(axis=0).sort_values(ascending=False).index
    plot_data = plot_data.loc[sort_order]

    fig, ax = plt.subplots(figsize=(18, 10))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)  # blue→white→red

    sns.heatmap(
        plot_data,
        cmap=cmap,
        center=0,
        linewidths=0.2,
        linecolor="#eeeeee",
        ax=ax,
        cbar_kws={
            "label": "Weight Drift (realised − target)",
            "shrink": 0.6,
            "pad": 0.02,
        },
    )

    # Format x-axis date labels
    n_cols = plot_data.shape[1]
    tick_every = max(1, n_cols // 12)
    ax.set_xticks(range(0, n_cols, tick_every))
    date_labels = [
        d.strftime("%b %Y") if hasattr(d, "strftime") else str(d)
        for d in plot_data.columns[::tick_every]
    ]
    ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    ax.set_title(
        f"{index_name} — Float-Adjusted Weight Drift Heatmap\n"
        f"Top {n_tickers} constituents by average |drift|  |  Weekly frequency",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Constituent Ticker", fontsize=11)

    # Annotation
    ax.text(
        0.01, -0.06,
        "Red = overweight (realised > target) → expected SELL pressure | "
        "Blue = underweight → expected BUY pressure",
        transform=ax.transAxes, fontsize=8, color="#555555",
    )

    fig.tight_layout()
    _save(fig, f"fig1_drift_heatmap_{index_name}.png")
    return fig


# ---------------------------------------------------------------------------
# 2. plot_drift_timeseries_single
# ---------------------------------------------------------------------------

def plot_drift_timeseries_single(
    drift_df: pd.DataFrame,
    drift_zscore_df: pd.DataFrame,
    ticker: str,
    actionable_df: Optional[pd.DataFrame] = None,
    rebalance_calendar: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """Dual-axis chart: raw drift and drift z-score for one ticker.

    Parameters
    ----------
    drift_df : pd.DataFrame
        Wide-format drift DataFrame.
    drift_zscore_df : pd.DataFrame
        Wide-format drift z-score DataFrame.
    ticker : str
        The ticker to plot.
    actionable_df : pd.DataFrame, optional
        Output of :func:`drift_engine.flag_actionable_drift` for transition
        annotations.
    rebalance_calendar : pd.DataFrame, optional
        For shading blackout windows.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ticker not in drift_df.columns:
        raise ValueError(f"Ticker '{ticker}' not found in drift_df.")

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    dates = drift_df.index
    raw_drift = drift_df[ticker]
    zscore = drift_zscore_df[ticker]

    # Raw drift (left axis)
    ax1.plot(dates, raw_drift * 100, color=_BLUE, lw=1.5, alpha=0.85, label="Raw drift (bps)")
    ax1.axhline(config.DRIFT_THRESHOLD_SOFT * 100, color=_RED, ls="--", lw=0.9, alpha=0.7,
                label=f"±Soft threshold ({config.DRIFT_THRESHOLD_SOFT * 1e4:.0f} bps)")
    ax1.axhline(-config.DRIFT_THRESHOLD_SOFT * 100, color=_RED, ls="--", lw=0.9, alpha=0.7)
    ax1.axhline(config.DRIFT_THRESHOLD_HARD * 100, color=_RED, ls="-.", lw=0.9, alpha=0.5,
                label=f"±Hard threshold ({config.DRIFT_THRESHOLD_HARD * 1e4:.0f} bps)")
    ax1.axhline(-config.DRIFT_THRESHOLD_HARD * 100, color=_RED, ls="-.", lw=0.9, alpha=0.5)
    ax1.axhline(0, color="#999999", lw=0.7)
    ax1.set_ylabel("Weight Drift (basis points)", color=_BLUE, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=_BLUE)

    # Z-score (right axis)
    ax2.fill_between(dates, zscore, alpha=0.18, color=_ORANGE)
    ax2.plot(dates, zscore, color=_ORANGE, lw=1.2, alpha=0.9, label="Drift z-score")
    ax2.axhline(config.DRIFT_ZSCORE_THRESHOLD, color=_ORANGE, ls=":", lw=1.1,
                label=f"Z-score threshold (±{config.DRIFT_ZSCORE_THRESHOLD})")
    ax2.axhline(-config.DRIFT_ZSCORE_THRESHOLD, color=_ORANGE, ls=":", lw=1.1)
    ax2.axhline(0, color="#cccccc", lw=0.5)
    ax2.set_ylabel("Drift Z-Score", color=_ORANGE, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=_ORANGE)

    # Blackout shading
    if rebalance_calendar is not None:
        for _, row in rebalance_calendar.iterrows():
            ax1.axvspan(row["blackout_start"], row["blackout_end"],
                        alpha=0.08, color="#888888", zorder=0)

    # Action transitions
    if actionable_df is not None:
        ticker_flags = actionable_df[actionable_df["ticker"] == ticker].copy()
        buys = ticker_flags[ticker_flags["action"] == "BUY"]["date"]
        sells = ticker_flags[ticker_flags["action"] == "SELL"]["date"]
        ax1.scatter(buys, raw_drift.reindex(buys) * 100, marker="^",
                    color=_BLUE, s=60, zorder=5, label="BUY signal")
        ax1.scatter(sells, raw_drift.reindex(sells) * 100, marker="v",
                    color=_RED, s=60, zorder=5, label="SELL signal")

    ax1.set_title(
        f"{ticker} — Weight Drift Time Series\n"
        f"Left: raw drift (bps)  |  Right: rolling {config.DRIFT_LOOKBACK_DAYS}-day z-score",
        fontsize=12, fontweight="bold",
    )
    ax1.set_xlabel("Date", fontsize=11)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               fontsize=8, framealpha=0.9)

    fig.tight_layout()
    _save(fig, f"fig2_drift_timeseries_{ticker}.png")
    return fig


# ---------------------------------------------------------------------------
# 3. plot_signal_decay_curve
# ---------------------------------------------------------------------------

def plot_signal_decay_curve(decay_df: pd.DataFrame) -> plt.Figure:
    """Line chart: mean forward return vs. holding horizon for LONG and SHORT.

    Parameters
    ----------
    decay_df : pd.DataFrame
        Output of :func:`signal.compute_signal_decay`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    horizons = decay_df["horizon_days"].values
    mean_ret = decay_df["mean_return"].values * 100
    std_ret = decay_df["std_return"].values * 100
    n = decay_df["n_signals"].values
    se = std_ret / np.sqrt(np.maximum(n, 1))

    ax.plot(horizons, mean_ret, color=_BLUE, lw=2.2, marker="o", ms=7,
            label="Mean forward return (combined L/S)")
    ax.fill_between(horizons, mean_ret - se, mean_ret + se,
                    alpha=0.18, color=_BLUE, label="±1 std error")
    ax.axhline(0, color="#aaaaaa", lw=0.8)

    # Annotate peak
    peak_idx = np.nanargmax(mean_ret)
    peak_h = horizons[peak_idx]
    peak_r = mean_ret[peak_idx]
    ax.axvline(peak_h, color=_RED, ls="--", lw=1.2,
               label=f"Peak horizon: day {peak_h}")
    ax.annotate(
        f"Peak\n{peak_r:.2f}%",
        xy=(peak_h, peak_r), xytext=(peak_h + 1.0, peak_r + 0.05),
        fontsize=9, color=_RED,
        arrowprops=dict(arrowstyle="->", color=_RED, lw=1.0),
    )

    ax.set_xlabel("Holding Horizon (trading days)", fontsize=12)
    ax.set_ylabel("Mean Forward Return (%)", fontsize=12)
    ax.set_title(
        "Signal Decay Curve — Mean Forward Return by Horizon\n"
        "(Combined long + short book; positive = signal working)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(horizons)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "fig3_signal_decay_curve.png")
    return fig


# ---------------------------------------------------------------------------
# 4. plot_cumulative_pnl_comparison
# ---------------------------------------------------------------------------

def plot_cumulative_pnl_comparison(
    sp500_backtest: pd.DataFrame,
    sp400_backtest: pd.DataFrame,
    rebalance_calendar: Optional[pd.DataFrame] = None,
    sp500_sharpe: float = np.nan,
    sp400_sharpe: float = np.nan,
) -> plt.Figure:
    """Overlay SP500 and SP400 LS strategy cumulative PnL.

    Parameters
    ----------
    sp500_backtest : pd.DataFrame
        Output of :func:`backtest.run_long_short_backtest` for SP500.
    sp400_backtest : pd.DataFrame
        Output of :func:`backtest.run_long_short_backtest` for SP400.
    rebalance_calendar : pd.DataFrame, optional
        Used to shade blackout windows.
    sp500_sharpe : float, optional
        Pre-computed Sharpe for legend annotation.
    sp400_sharpe : float, optional
        Pre-computed Sharpe for legend annotation.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    sp500_lbl = f"SP500 LS Strategy (Sharpe={sp500_sharpe:.2f})" if not np.isnan(sp500_sharpe) else "SP500 LS Strategy"
    sp400_lbl = f"SP400 LS Strategy (Sharpe={sp400_sharpe:.2f})" if not np.isnan(sp400_sharpe) else "SP400 LS Strategy"

    if not sp500_backtest.empty:
        ax.plot(sp500_backtest["date"], sp500_backtest["cumulative_pnl"],
                color=_GREY, lw=1.5, ls="--", alpha=0.85, label=sp500_lbl)

    if not sp400_backtest.empty:
        ax.plot(sp400_backtest["date"], sp400_backtest["cumulative_pnl"],
                color=_BLUE, lw=2.2, label=sp400_lbl)

    ax.axhline(1.0, color="#aaaaaa", lw=0.7, ls=":")

    # Shade blackout windows
    if rebalance_calendar is not None:
        for _, row in rebalance_calendar.iterrows():
            ax.axvspan(row["blackout_start"], row["blackout_end"],
                       alpha=0.06, color="#888888", zorder=0)
        grey_patch = mpatches.Patch(color="#888888", alpha=0.15,
                                     label="Rebalance blackout windows")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(grey_patch)
        labels.append("Rebalance blackout windows")
        ax.legend(handles, labels, fontsize=10)
    else:
        ax.legend(fontsize=10)

    ax.set_title(
        "Cumulative PnL: SP400 vs SP500 Long/Short Strategy\n"
        "(Net of 8 bps one-way transaction costs)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative PnL ($1 start)", fontsize=12)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    _save(fig, "fig4_cumulative_pnl_comparison.png")
    return fig


# ---------------------------------------------------------------------------
# 5. plot_flow_confirmation_ic_split
# ---------------------------------------------------------------------------

def plot_flow_confirmation_ic_split(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizon: int = config.HOLDING_PERIOD_DAYS,
) -> plt.Figure:
    """Side-by-side IC bars: flow-confirmed vs. unconfirmed signals.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Must contain ``flow_confirmed`` column from
        :func:`signal.overlay_etf_flow_confirmation`.
    price_df : pd.DataFrame
        Wide-format prices.
    horizon : int, optional
        Forward return horizon.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy import stats as scipy_stats

    price_aligned = price_df.sort_index()
    all_dates = price_aligned.index.tolist()

    def _compute_ic_for_subset(df: pd.DataFrame) -> tuple[float, float]:
        signals, fwds = [], []
        for _, row in df.iterrows():
            t = row["ticker"]
            d = row["date"]
            if t not in price_aligned.columns or d not in price_aligned.index:
                continue
            ti = all_dates.index(d)
            tf = ti + horizon
            if tf >= len(all_dates):
                continue
            p0 = price_aligned.at[d, t]
            p1 = price_aligned.at[all_dates[tf], t]
            if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                continue
            ret = (p1 - p0) / p0
            if row["book_side"] == "SHORT":
                ret = -ret
            signals.append(row["signal_strength"])
            fwds.append(ret)
        if len(signals) < 5:
            return np.nan, np.nan
        ic, pval = scipy_stats.spearmanr(signals, fwds)
        return float(ic), float(pval)

    groups = {
        "Confirmed\nLONG": signal_df[(signal_df["book_side"] == "LONG") & signal_df["flow_confirmed"]],
        "Unconfirmed\nLONG": signal_df[(signal_df["book_side"] == "LONG") & ~signal_df["flow_confirmed"]],
        "Confirmed\nSHORT": signal_df[(signal_df["book_side"] == "SHORT") & signal_df["flow_confirmed"]],
        "Unconfirmed\nSHORT": signal_df[(signal_df["book_side"] == "SHORT") & ~signal_df["flow_confirmed"]],
    }

    ic_vals, colors, labels = [], [], []
    for label, grp in groups.items():
        ic, _ = _compute_ic_for_subset(grp)
        ic_vals.append(ic if not np.isnan(ic) else 0.0)
        colors.append(_BLUE if "Confirmed" in label else _GREY)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, ic_vals, color=colors, edgecolor="white", linewidth=1.2, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(config.IC_MEANINGFUL_THRESHOLD, color=_RED, ls="--", lw=1.0,
               label=f"IC threshold ({config.IC_MEANINGFUL_THRESHOLD:.2f})")
    ax.axhline(-config.IC_MEANINGFUL_THRESHOLD, color=_RED, ls="--", lw=1.0)

    for bar, val in zip(bars, ic_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    confirmed_patch = mpatches.Patch(color=_BLUE, alpha=0.85, label="Flow-Confirmed")
    unconf_patch = mpatches.Patch(color=_GREY, alpha=0.85, label="Unconfirmed")
    ax.legend(handles=[confirmed_patch, unconf_patch,
                         mpatches.Patch(color=_RED, alpha=0.6, label=f"IC threshold ({config.IC_MEANINGFUL_THRESHOLD})")],
              fontsize=10)
    ax.set_title(
        "Information Coefficient Split: Flow-Confirmed vs. Unconfirmed Signals\n"
        f"Spearman IC against {horizon}-day forward return",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Spearman IC", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig5_flow_confirmation_ic_split.png")
    return fig


# ---------------------------------------------------------------------------
# 6. plot_rebalance_window_returns
# ---------------------------------------------------------------------------

def plot_rebalance_window_returns(window_analysis_df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart of mean LS returns by rebalance window period.

    Parameters
    ----------
    window_analysis_df : pd.DataFrame
        Output of :func:`backtest.run_rebalance_window_analysis`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    periods = ["PRE", "NEAR", "POST"]
    period_colors = {"PRE": _BLUE, "NEAR": _GREY, "POST": _ORANGE}

    # Summarise by period
    summary = (
        window_analysis_df.groupby("period")["mean_ls_return"]
        .agg(["mean", "sem", "count"])
        .reindex(periods)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(periods))
    width = 0.55

    bars = ax.bar(
        x, summary["mean"] * 100,
        width=width,
        color=[period_colors[p] for p in periods],
        edgecolor="white", linewidth=1.2, alpha=0.85,
        yerr=summary["sem"] * 100,
        capsize=5,
    )

    # Annotate with signal counts
    for i, (period, row) in enumerate(summary.iterrows()):
        n = int(row["count"]) if not np.isnan(row["count"]) else 0
        y_pos = (row["mean"] * 100 if not np.isnan(row["mean"]) else 0) + \
                (row["sem"] * 100 if not np.isnan(row["sem"]) else 0) + 0.05
        ax.text(i, y_pos, f"n={n}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["PRE (−20 to −6 days)", "NEAR / Blackout\n(−5 to −1 days)",
         "POST (+1 to +10 days)"],
        fontsize=10,
    )
    ax.set_ylabel("Mean LS Return (%)", fontsize=12)
    ax.set_title(
        "Mean Long/Short Return by Rebalance Window Period\n"
        "(Error bars = ±1 standard error; n = number of rebalance events)",
        fontsize=12, fontweight="bold",
    )

    legend_patches = [
        mpatches.Patch(color=_BLUE, alpha=0.85, label="PRE — signal capture window"),
        mpatches.Patch(color=_GREY, alpha=0.85, label="NEAR — blackout zone"),
        mpatches.Patch(color=_ORANGE, alpha=0.85, label="POST — post-rebalance drift"),
    ]
    ax.legend(handles=legend_patches, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig6_rebalance_window_returns.png")
    return fig


# ---------------------------------------------------------------------------
# 7. plot_weight_drift_vs_forward_return_scatter
# ---------------------------------------------------------------------------

def plot_weight_drift_vs_forward_return_scatter(
    actionable_drift_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizon: int = config.HOLDING_PERIOD_DAYS,
) -> plt.Figure:
    """Scatter: drift z-score vs. forward return with OLS trendline.

    Parameters
    ----------
    actionable_drift_df : pd.DataFrame
        Output of :func:`drift_engine.flag_actionable_drift` filtered to
        actionable signals (BUY or SELL).
    price_df : pd.DataFrame
        Wide-format adjusted close prices.
    horizon : int, optional
        Forward return horizon in trading days.

    Returns
    -------
    matplotlib.figure.Figure
    """
    price_aligned = price_df.sort_index()
    all_dates = price_aligned.index.tolist()

    flagged = actionable_drift_df[actionable_drift_df["action"].isin(["BUY", "SELL"])].copy()

    records = []
    for _, row in flagged.iterrows():
        t = row["ticker"]
        d = row["date"]
        side = "LONG" if row["action"] == "BUY" else "SHORT"

        if t not in price_aligned.columns or d not in price_aligned.index:
            continue
        ti = all_dates.index(d)
        tf = ti + horizon
        if tf >= len(all_dates):
            continue
        p0 = price_aligned.at[d, t]
        p1 = price_aligned.at[all_dates[tf], t]
        if pd.isna(p0) or pd.isna(p1) or p0 == 0:
            continue
        ret = (p1 - p0) / p0
        if side == "SHORT":
            ret = -ret
        records.append(
            {
                "drift_zscore": row["drift_zscore"],
                "fwd_return_pct": ret * 100,
                "book_side": side,
            }
        )

    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        logger.warning("No data for scatter plot — returning empty figure.")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(11, 7))

    for side, color, marker in [("LONG", _BLUE, "o"), ("SHORT", _RED, "s")]:
        sub = plot_df[plot_df["book_side"] == side]
        if sub.empty:
            continue
        ax.scatter(
            sub["drift_zscore"], sub["fwd_return_pct"],
            color=color, alpha=0.35, s=28, marker=marker,
            label=f"{side} signals (n={len(sub)})",
        )
        # OLS trendline
        x_vals = sub["drift_zscore"].values
        y_vals = sub["fwd_return_pct"].values
        mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        if mask.sum() >= 5:
            slope, intercept, r_val, p_val, _ = stats.linregress(
                x_vals[mask], y_vals[mask]
            )
            x_line = np.linspace(x_vals[mask].min(), x_vals[mask].max(), 100)
            ax.plot(
                x_line, intercept + slope * x_line,
                color=color, lw=2.0, ls="--",
                label=f"{side} OLS (slope={slope:.3f})",
            )

    # IC annotation
    if len(plot_df) >= 10:
        ic, pval = stats.spearmanr(plot_df["drift_zscore"], plot_df["fwd_return_pct"])
        ax.text(
            0.03, 0.95,
            f"Spearman IC = {ic:.4f}  |  p-value = {pval:.4f}  |  n = {len(plot_df)}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
        )

    ax.axvline(0, color="#aaaaaa", lw=0.8)
    ax.axhline(0, color="#aaaaaa", lw=0.8)
    ax.set_xlabel("Drift Z-Score at Signal Date", fontsize=12)
    ax.set_ylabel(f"{horizon}-Day Forward Return (%; sign-adjusted for short)", fontsize=12)
    ax.set_title(
        "Weight Drift Z-Score vs. Forward Return\n"
        "Regression-free validation: slope direction and magnitude tell the story",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "fig7_drift_vs_return_scatter.png")
    return fig
