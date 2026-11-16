"""Download prices for every event from Yahoo Finance and freeze the returns around each one.

    pip install yfinance
    python scripts/fetch_data.py

Writes data/snapshot/event_returns.csv.gz: for every event with prices, one row per trading
day from 30 days before the earlier of the announcement and effective dates to 15 days after
the effective date, with the close to close return of the stock and of the benchmark (MDY, the
SPDR S&P MidCap 400 ETF) and the overnight part of each (previous close to open). Only these
windows are stored, not full price histories.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import events  # noqa: E402

BENCH = "MDY"
BEFORE, AFTER = 30, 15
OUT = ROOT / "data" / "snapshot" / "event_returns.csv.gz"


def download(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    closes, opens = [], []
    names = sorted({t.replace(".", "-") for t in tickers} | {BENCH})
    for i in range(0, len(names), 60):
        px = yf.download(names[i:i + 60], start="2011-09-01", end="2026-11-01", auto_adjust=True,
                         progress=False, threads=True)
        closes.append(px["Close"])
        opens.append(px["Open"])
    c, o = pd.concat(closes, axis=1), pd.concat(opens, axis=1)
    keep = ~c.columns.duplicated()
    return c.loc[:, keep].sort_index(), o.loc[:, keep].sort_index()


def main() -> None:
    ev = events.build_events(*events.load_reference())
    ev = ev[ev["group"] != "deletion_mna"]
    close, open_ = download(ev["ticker"].unique().tolist())
    ret = close.pct_change(fill_method=None)
    night = open_ / close.shift(1) - 1.0
    days = close.index
    rows = []
    for e in ev.itertuples():
        col = e.ticker.replace(".", "-")
        if col not in close.columns:
            continue
        pos = days.searchsorted(e.date)
        first = pos
        if pd.notna(e.announcement_date):
            first = min(pos, days.searchsorted(e.announcement_date, side="right"))
        lo, hi = first - BEFORE, pos + AFTER
        if lo < 1 or hi >= len(days):
            continue
        w = pd.DataFrame({"event_id": e.event_id, "day": days[lo:hi + 1],
                          "ret": ret[col].iloc[lo:hi + 1].to_numpy(), "night": night[col].iloc[lo:hi + 1].to_numpy(),
                          "bench_ret": ret[BENCH].iloc[lo:hi + 1].to_numpy(),
                          "bench_night": night[BENCH].iloc[lo:hi + 1].to_numpy()})
        if w["ret"].notna().all():
            rows.append(w)
    out = pd.concat(rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.round(6).to_csv(OUT, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"{out['event_id'].nunique()} of {len(ev)} events with prices, {len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
