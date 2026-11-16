"""Event study of S&P 400 additions and deletions.

Design, fixed before any abnormal return was computed for this version:

  events      every stock entering or leaving the S&P 400 from 2012, excluding deletions caused
              by mergers. From 2020 each event is classified by origin or destination.
  abnormal    stock return minus the return of MDY (the S&P MidCap 400 ETF) on the same day
  dates       E  = first trading day on or after the effective date. Index funds trade at the
                   close of E-1.
              A0 = first trading day after the announcement date. S&P announces after the close,
                   so the close before A0 is the last price without the news.
  windows around E    PRE [-20,-6], NEAR [-5,-1], DAY0 [0], POST [+1,+10]
  windows around A0   BEFORE [A0-10, A0-1]
                      DAY_BEFORE the last session before A0, part of BEFORE (a check on the dates)
                      OVERNIGHT  close before A0 to the open of A0   (not tradable on the news)
                      INTRADAY   open of A0 to its close             (tradable at the open)
                      RUNUP      close of A0 to the close of E-1     (tradable at the close)
                      AFTER      [E, E+10]                           (tradable with the funds)
  inference   events that share a date are averaged first, then a t-test runs across dates.
              The nine tradable tests (three windows, three groups with index fund flow) get a
              Holm correction.
"""
from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "snapshot" / "event_returns.csv.gz"
E_WINDOWS = {"PRE": (-20, -6), "NEAR": (-5, -1), "DAY0": (0, 0), "POST": (1, 10)}
A_WINDOWS = ["BEFORE", "DAY_BEFORE", "OVERNIGHT", "INTRADAY", "RUNUP", "AFTER"]
TRADABLE = ["INTRADAY", "RUNUP", "AFTER"]
FLOW_GROUPS = ["addition_new", "addition_from_sp600", "deletion_to_sp600"]
MIN_DATES = 5
MAX_GAP_DAYS = 30


def load_returns() -> pd.DataFrame:
    return pd.read_csv(SNAPSHOT, parse_dates=["day"])


def event_windows(w: pd.DataFrame, effective: pd.Timestamp, announced) -> dict:
    """Cumulative abnormal returns of one event. w holds its daily rows in date order."""
    days = w["day"].to_numpy()
    ar = (w["ret"] - w["bench_ret"]).to_numpy()
    night = (w["night"] - w["bench_night"]).to_numpy()
    intraday = ((1 + w["ret"]) / (1 + w["night"]) - (1 + w["bench_ret"]) / (1 + w["bench_night"])).to_numpy()
    e = int(np.searchsorted(days, np.datetime64(effective)))
    out = {"E": pd.Timestamp(days[e])}

    def car(lo: int, hi: int) -> float:
        if hi < lo:
            return 0.0
        if lo < 0 or hi >= len(ar):
            return float("nan")
        seg = ar[lo: hi + 1]
        return float(seg.sum()) if not np.isnan(seg).any() else float("nan")

    for name, (a, b) in E_WINDOWS.items():
        out[name] = car(e + a, e + b)
    if pd.notna(announced) and 1 <= (effective - announced).days <= MAX_GAP_DAYS:
        a0 = int(np.searchsorted(days, np.datetime64(announced), side="right"))
        if 10 <= a0 < e:
            out.update({"A0": pd.Timestamp(days[a0]), "gap": e - a0, "BEFORE": car(a0 - 10, a0 - 1),
                        "DAY_BEFORE": car(a0 - 1, a0 - 1),
                        "OVERNIGHT": float(night[a0]), "INTRADAY": float(intraday[a0]),
                        "RUNUP": car(a0 + 1, e - 1), "AFTER": car(e, e + 10)})
    return out


def compute(events: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """One row per event with its windows. Events without prices keep NaN windows."""
    by_id = {k: g for k, g in returns.groupby("event_id", sort=False)}
    rows = []
    for e in events.itertuples():
        row = {"event_id": e.event_id, "date": e.date, "announcement_date": e.announcement_date,
               "ticker": e.ticker, "type": e.type, "group": e.group, "has_prices": e.event_id in by_id}
        if row["has_prices"]:
            row.update(event_windows(by_id[e.event_id], e.date, e.announcement_date))
        rows.append(row)
    return pd.DataFrame(rows)


def t_dist_p(t: float, df: int) -> float:
    """Two sided p-value of a t statistic, by numerical integration of the density."""
    if not np.isfinite(t) or df < 1:
        return float("nan")
    from math import gamma, pi
    x = np.linspace(0.0, abs(t), 20001)
    dens = gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2)) * (1 + x ** 2 / df) ** (-(df + 1) / 2)
    area = float(((dens[1:] + dens[:-1]) / 2 * np.diff(x)).sum())
    return max(0.0, 1.0 - 2.0 * area)


def clustered(values: pd.Series, dates: pd.Series) -> dict | None:
    """Mean across dates of the per-date average, with its t-test."""
    ok = values.notna()
    by_date = values[ok].groupby(dates[ok]).mean()
    n = len(by_date)
    if n < MIN_DATES:
        return None
    m, sd = float(by_date.mean()), float(by_date.std(ddof=1))
    t = m / (sd / sqrt(n))
    return {"mean": m, "t": float(t), "p": t_dist_p(t, n - 1), "n_dates": int(n), "n_events": int(ok.sum()),
            "median_event": float(values[ok].median()), "share_positive": float((values[ok] > 0).mean())}


def summarize(df: pd.DataFrame) -> dict:
    out = {"n_events": int(len(df)), "n_with_prices": int(df["has_prices"].sum()),
           "n_with_announcement": int(df["INTRADAY"].notna().sum()) if "INTRADAY" in df else 0}
    for w in E_WINDOWS:
        out[w] = clustered(df[w], df["date"]) if w in df else None
    for w in A_WINDOWS:
        out[w] = clustered(df[w], df["A0"]) if w in df and "A0" in df else None
    if "INTRADAY" in df and "A0" in df:
        out["FROM_OPEN"] = clustered(df["INTRADAY"] + df["RUNUP"], df["A0"])
        out["REACTION"] = clustered(df["OVERNIGHT"] + df["INTRADAY"], df["A0"])
    return out


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running, m = {}, 0.0, len(items)
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
