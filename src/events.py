"""S&P 400 membership changes, classified by where the stock came from or went to.

Source: the changes tables of the Wikipedia articles List of S&P 400 companies and List of
S&P 600 companies, and the current S&P 500 member list (CC BY-SA 4.0), saved in
data/reference/. scripts/fetch_data.py downloads them again.

Groups:
  addition_new         joined the S&P 400 from outside the S&P 1500 family or from the S&P 500
  addition_from_sp600  moved up from the S&P 600
  deletion_to_sp600    moved down to the S&P 600
  deletion_to_sp500    moved up to the S&P 500
  deletion_mna         left because of a merger, acquisition or spin-off (not studied)
  deletion_other       any other deletion

Origin and destination can only be told apart from 2020, when the S&P 600 change table begins.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
EVENT_START, EVENT_END = "2012-01-01", "2026-09-30"
CLASSIFIED_FROM = "2020-01-01"
MATCH_DAYS = 10
MNA = re.compile(r"acqui|merg|spun|spin|bought|takeover|purchase")


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ch = pd.read_csv(REF / "sp400_changes.csv", comment="#", parse_dates=["date", "announcement_date"])
    s6 = pd.read_csv(REF / "sp600_changes.csv", parse_dates=["date"]).dropna(subset=["date"])
    s5 = pd.read_csv(REF / "sp500_current_members.csv", parse_dates=["Date added"])
    return ch, s6, s5


def build_events(ch: pd.DataFrame, s6: pd.DataFrame, s5: pd.DataFrame) -> pd.DataFrame:
    """One row per stock entering or leaving the S&P 400."""
    up = dict(zip(s5["Symbol"], s5["Date added"]))
    to600 = [(str(r.added_ticker).strip(), r.date) for r in s6.itertuples()]
    from600 = [(str(r.removed_ticker).strip(), r.date) for r in s6.itertuples()]

    def near(pairs, ticker, date):
        return any(t == ticker and abs((d - date).days) <= MATCH_DAYS for t, d in pairs)

    rows = []
    for r in ch.dropna(subset=["date"]).itertuples():
        if not (pd.Timestamp(EVENT_START) <= r.date <= pd.Timestamp(EVENT_END)):
            continue
        mna = bool(MNA.search(str(r.reason).lower()))
        base = {"date": r.date, "announcement_date": r.announcement_date, "reason": r.reason}
        if isinstance(r.added_ticker, str) and r.added_ticker.strip():
            t = r.added_ticker.strip()
            rows.append({**base, "ticker": t, "type": "addition",
                         "group": "addition_from_sp600" if near(from600, t, r.date) else "addition_new"})
        if isinstance(r.removed_ticker, str) and r.removed_ticker.strip():
            t = r.removed_ticker.strip()
            if mna:
                g = "deletion_mna"
            elif near(to600, t, r.date):
                g = "deletion_to_sp600"
            elif t in up and abs((up[t] - r.date).days) <= MATCH_DAYS:
                g = "deletion_to_sp500"
            else:
                g = "deletion_other"
            rows.append({**base, "ticker": t, "type": "deletion", "group": g})
    ev = pd.DataFrame(rows).sort_values(["date", "ticker", "type"]).reset_index(drop=True)
    ev.insert(0, "event_id", [f"{d:%Y%m%d}-{t}-{k[0]}" for d, t, k in zip(ev["date"], ev["ticker"], ev["type"])])
    return ev
