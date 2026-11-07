"""Offline tests: drift math, point-in-time z-scores, lagging and backtest mechanics."""

import numpy as np
import pandas as pd
import pytest

import config
from src.backtest import _compute_drawdown, _forward_return, run_long_short_backtest
from src.drift_engine import (
    compute_absolute_drift_rank,
    compute_drift_zscore,
    compute_realized_weights,
    compute_weight_drift,
)
from src.signal import build_daily_signal
from src.utils import lag_dataframe, lag_signal_dates


def _prices(n=120, k=8, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0)),
        index=idx,
        columns=[f"T{i}" for i in range(k)],
    )


def _constituents(px):
    k = px.shape[1]
    return pd.DataFrame(
        {"ticker": px.columns, "float_shares": np.full(k, 1e6), "target_weight": np.full(k, 1.0 / k)}
    )


# ------------------------------------------------------------------ weights
def test_realized_weights_sum_to_one_and_use_same_day_prices_only():
    px = _prices()
    w = compute_realized_weights(px, _constituents(px))
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    # weight on day t is unchanged if future prices are altered
    px2 = px.copy()
    px2.iloc[60:] *= 2.0
    w2 = compute_realized_weights(px2, _constituents(px2))
    pd.testing.assert_frame_equal(w.iloc[:60], w2.iloc[:60])


def test_weight_drift_sign_overweight_is_positive():
    px = _prices()
    cons = _constituents(px)
    px = px.copy()
    px["T0"] *= 3.0  # T0 becomes overweight vs equal target
    drift = compute_weight_drift(compute_realized_weights(px, cons), cons)
    assert (drift["T0"] > 0).all()
    np.testing.assert_allclose(drift.sum(axis=1), 0.0, atol=1e-12)


# ------------------------------------------------------------------- zscore
def test_drift_zscore_is_point_in_time_and_capped():
    px = _prices()
    cons = _constituents(px)
    drift = compute_weight_drift(compute_realized_weights(px, cons), cons)
    full = compute_drift_zscore(drift, window=20)
    for cut in (50, 90):
        trunc = compute_drift_zscore(drift.iloc[:cut], window=20)
        pd.testing.assert_frame_equal(trunc, full.iloc[:cut])
    assert full.abs().max().max() <= config.DRIFT_ZSCORE_WINSOR_CAP + 1e-12
    assert full.iloc[:4].isna().all().all()  # min_periods warm-up


def test_absolute_drift_rank_in_unit_interval_and_orders_by_magnitude():
    drift = pd.DataFrame({"A": [0.01, -0.02], "B": [0.03, 0.001], "C": [-0.005, 0.0]})
    r = compute_absolute_drift_rank(drift)
    assert r.max().max() == 1.0 and r.min().min() > 0
    assert r.iloc[0].idxmax() == "B" and r.iloc[1].idxmax() == "A"


# --------------------------------------------------------------------- lags
def test_row_lag_does_not_lag_long_format_signal_but_date_lag_does():
    """Regression: row-shifting a long frame leaves almost every row on its own date."""
    dates = pd.bdate_range("2021-01-04", periods=5)
    sig = pd.DataFrame(
        [(d, f"T{i}", "LONG") for d in dates for i in range(6)],
        columns=["date", "ticker", "book_side"],
    )
    # the (buggy) row shift keeps the same dates
    row_lagged = lag_dataframe(sig.set_index("date")).reset_index()
    assert (row_lagged["date"] == sig["date"]).all()
    # the date-aware lag moves every signal one trading day later
    out = lag_signal_dates(sig, dates, periods=1)
    assert out["date"].min() == dates[1]
    assert len(out) == len(sig) - 6  # last day falls off the calendar
    merged = sig.iloc[:-6].reset_index(drop=True)
    assert ((out["date"] - merged["date"]).dt.days >= 1).all()


def test_lag_signal_dates_zero_and_unknown_dates():
    dates = pd.bdate_range("2021-01-04", periods=3)
    sig = pd.DataFrame({"date": dates, "ticker": "A"})
    pd.testing.assert_frame_equal(lag_signal_dates(sig, dates, 0), sig)
    with pytest.raises(ValueError):
        lag_signal_dates(pd.DataFrame({"date": [pd.Timestamp("2000-01-03")], "ticker": ["A"]}), dates)


# ------------------------------------------------------------------- signal
def _actionable(n_buy, n_sell, date="2021-03-01"):
    rows = []
    for i in range(n_buy):
        rows.append((pd.Timestamp(date), f"B{i}", "BUY", 1.0 + i, -0.004, -2.0))
    for i in range(n_sell):
        rows.append((pd.Timestamp(date), f"S{i}", "SELL", 1.0 + i, 0.004, 2.0))
    return pd.DataFrame(rows, columns=["date", "ticker", "action", "signal_strength", "drift", "drift_zscore"])


def test_signal_flat_when_book_too_thin():
    assert build_daily_signal(_actionable(config.MIN_BOOK_SIZE - 1, 10)).empty


def test_signal_keeps_top_quintile_with_equal_weights():
    sig = build_daily_signal(_actionable(20, 20))
    longs = sig[sig["book_side"] == "LONG"]
    assert longs["signal_strength"].min() >= np.quantile(np.arange(20) + 1.0, config.SIGNAL_QUINTILE)
    np.testing.assert_allclose(longs["portfolio_weight"].sum(), 1.0)
    np.testing.assert_allclose(sig[sig["book_side"] == "SHORT"]["portfolio_weight"].sum(), 1.0)


# ----------------------------------------------------------------- backtest
def test_forward_return_matches_hand_calc_and_needs_full_horizon():
    px = _prices()
    d0 = px.index[10]
    assert _forward_return(px, "T0", d0, 5) == pytest.approx(px["T0"].iloc[15] / px["T0"].iloc[10] - 1)
    assert _forward_return(px, "T0", px.index[-3], 5) is None


def test_drawdown_is_fraction_of_peak():
    equity = pd.Series([1.0, 1.5, 2.0, 1.0, 1.2])
    dd = _compute_drawdown(equity)
    assert dd.iloc[2] == 0 and dd.iloc[3] == pytest.approx(-0.5)


def test_backtest_pnl_does_not_depend_on_prices_before_entry(monkeypatch):
    monkeypatch.setattr(config, "HOLDING_PERIOD_DAYS", 5)
    px = _prices(n=60, k=12)
    entry = px.index[20]
    sig = pd.DataFrame(
        [(entry, f"T{i}", "LONG", 0.5) for i in range(6)]
        + [(entry, f"T{i}", "SHORT", 0.5) for i in range(6, 12)],
        columns=["date", "ticker", "book_side", "portfolio_weight"],
    )
    base = run_long_short_backtest(sig, px)
    px2 = px.copy()
    px2.iloc[:20] *= 4.0
    alt = run_long_short_backtest(sig, px2)
    pd.testing.assert_series_equal(base["net_ls_return"], alt["net_ls_return"])
    assert base["net_ls_return"].iloc[:20].eq(0).all()  # nothing earned before entry
    assert base["cumulative_pnl"].iloc[0] == 1.0
