"""Checks on the event list, the windows and the inference."""
import numpy as np
import pandas as pd
import pytest

from src import events, study


# ---------- events ----------

def tables():
    ch = pd.DataFrame({
        "date": pd.to_datetime(["2021-03-22", "2021-03-22", "2021-06-01", "2021-06-01", "2015-05-01", "2030-01-01"]),
        "added_ticker": ["NEW", "UP6", "X1", "X2", None, "LATE"],
        "removed_ticker": ["DN6", "UP5", "GONE", "OTH", "OLD", None],
        "reason": ["Market capitalization change.", "x", "GONE was acquired by BIG.", "x", "x", "x"],
        "announcement_date": pd.to_datetime(["2021-03-12", "2021-03-12", None, "2021-05-27", None, None]),
    })
    s6 = pd.DataFrame({"date": pd.to_datetime(["2021-03-22", "2021-03-19"]), "added_ticker": ["DN6", "ZZ"],
                       "removed_ticker": ["QQ", "UP6"]})
    s5 = pd.DataFrame({"Symbol": ["UP5"], "Date added": pd.to_datetime(["2021-03-22"])})
    return ch, s6, s5


def test_classification():
    ev = events.build_events(*tables()).set_index("ticker")
    assert ev.loc["NEW", "group"] == "addition_new"
    assert ev.loc["UP6", "group"] == "addition_from_sp600"          # left the S&P 600 three days earlier
    assert ev.loc["DN6", "group"] == "deletion_to_sp600"
    assert ev.loc["UP5", "group"] == "deletion_to_sp500"
    assert ev.loc["GONE", "group"] == "deletion_mna"
    assert ev.loc["OTH", "group"] == "deletion_other"
    assert "LATE" not in ev.index                                    # outside the event window
    assert ev["event_id"].is_unique and ev.loc["NEW", "event_id"] == "20210322-NEW-a"


def test_reference_tables_build():
    ev = events.build_events(*events.load_reference())
    assert len(ev) > 1000 and ev["event_id"].is_unique
    assert set(ev["group"]) == {"addition_new", "addition_from_sp600", "deletion_to_sp600", "deletion_to_sp500",
                                "deletion_mna", "deletion_other"}
    assert ev["date"].between(events.EVENT_START, events.EVENT_END).all()


# ---------- windows ----------

def window(n=60, start="2021-02-01"):
    days = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"day": days, "ret": 0.0, "night": 0.0, "bench_ret": 0.0, "bench_night": 0.0})


def test_reaction_is_split_into_overnight_and_intraday():
    w = window()
    eff, ann = w["day"].iloc[40], w["day"].iloc[34]                  # announced after the close of day 34
    w.loc[35, ["ret", "night"]] = [(1.05 * 1.01) - 1, 0.05]          # gaps 5% then gains 1% in the session
    out = study.event_windows(w, eff, ann)
    assert out["A0"] == w["day"].iloc[35] and out["gap"] == 5
    assert out["OVERNIGHT"] == pytest.approx(0.05) and out["INTRADAY"] == pytest.approx(0.01)
    assert out["RUNUP"] == 0 and out["BEFORE"] == 0 and out["AFTER"] == 0 and out["DAY_BEFORE"] == 0
    assert out["NEAR"] == pytest.approx(1.05 * 1.01 - 1)             # the reaction falls in days -5 to -1


def test_windows_are_abnormal_and_placed_correctly():
    w = window()
    eff, ann = w["day"].iloc[40], w["day"].iloc[34]
    w["bench_ret"] = 0.001
    w.loc[34, "ret"] = 0.02          # session before A0
    w.loc[37, "ret"] = 0.03          # between the first close and the fund trade
    w.loc[40, "ret"] = 0.04          # effective day
    w.loc[45, "ret"] = 0.01          # after
    out = study.event_windows(w, eff, ann)
    assert out["DAY_BEFORE"] == pytest.approx(0.02 - 0.001)
    assert out["RUNUP"] == pytest.approx(0.03 - 4 * 0.001)           # days 36 to 39
    assert out["DAY0"] == pytest.approx(0.04 - 0.001)
    assert out["AFTER"] == pytest.approx(0.04 + 0.01 - 11 * 0.001)   # days 40 to 50
    assert out["POST"] == pytest.approx(0.01 - 10 * 0.001)


def test_effective_date_on_a_holiday_moves_to_the_next_session():
    w = window()
    saturday = w["day"].iloc[39] + pd.Timedelta(days=1)
    assert study.event_windows(w, saturday, pd.NaT)["E"] == w["day"].iloc[40]


def test_no_announcement_windows_without_a_usable_date():
    w = window()
    eff = w["day"].iloc[40]
    assert "A0" not in study.event_windows(w, eff, pd.NaT)
    assert "A0" not in study.event_windows(w, eff, eff)                               # same day
    assert "A0" not in study.event_windows(w, eff, eff - pd.Timedelta(days=45))       # too long ago


def test_missing_return_gives_nan_not_zero():
    w = window()
    w.loc[42, "ret"] = np.nan
    out = study.event_windows(w, w["day"].iloc[40], pd.NaT)
    assert np.isnan(out["POST"]) and out["PRE"] == 0


# ---------- inference ----------

def test_same_date_events_count_once():
    vals = pd.Series([0.10] * 50 + [-0.01, -0.02, -0.01, -0.02, -0.01, -0.02])
    dates = pd.Series([pd.Timestamp("2021-01-04")] * 50 + list(pd.bdate_range("2021-02-01", periods=6)))
    out = study.clustered(vals, dates)
    assert out["n_dates"] == 7 and out["n_events"] == 56
    assert out["mean"] == pytest.approx((0.10 - 0.09) / 7)           # one date with fifty events is one observation
    assert abs(out["t"]) < 1


def test_too_few_dates_gives_nothing():
    assert study.clustered(pd.Series([0.1, 0.2]), pd.Series(pd.bdate_range("2021-01-04", periods=2))) is None


def test_t_distribution_p_values():
    assert study.t_dist_p(2.045, 29) == pytest.approx(0.05, abs=0.001)
    assert study.t_dist_p(-2.045, 29) == pytest.approx(0.05, abs=0.001)
    assert study.t_dist_p(0.0, 10) == pytest.approx(1.0)


def test_holm():
    assert study.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


# ---------- the frozen data ----------

def test_snapshot_matches_the_event_list():
    r = study.load_returns()
    ev = events.build_events(*events.load_reference())
    assert set(r["event_id"]) <= set(ev.loc[ev["group"] != "deletion_mna", "event_id"])
    assert r["ret"].notna().all() and r["bench_ret"].notna().all()
    assert r.groupby("event_id")["day"].apply(lambda d: d.is_monotonic_increasing).all()
    assert r["ret"].abs().max() < 3
