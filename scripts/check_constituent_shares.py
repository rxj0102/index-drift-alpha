"""Compare float_shares in data/reference/index_constituents.csv with SEC-reported shares outstanding.

Source: SEC XBRL companyconcept dei:EntityCommonStockSharesOutstanding, latest value dated 2023-07-01 to 2023-12-31
(the file's as_of_date is 2023-09-30). Shares outstanding is an upper bound on float, so a listed float far above it
cannot be right. Output: results/constituent_share_check.csv

Usage: EDGAR_USER_AGENT="Name email" python scripts/check_constituent_shares.py
"""
import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": os.getenv("EDGAR_USER_AGENT", "index-drift-alpha research")}


def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))


def main():
    tick = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in get("https://www.sec.gov/files/company_tickers.json").values()}
    cons = pd.read_csv(ROOT / "data" / "reference" / "index_constituents.csv")
    rows = []
    for _, r in cons.iterrows():
        cik, sec = tick.get(r.ticker), None
        if cik:
            try:
                f = get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/dei/EntityCommonStockSharesOutstanding.json")
                pts = [u for u in f["units"]["shares"] if "2023-07-01" <= u["end"] <= "2023-12-31"]
                sec = max(pts, key=lambda u: u["end"])["val"] if pts else None
            except Exception:
                sec = None
            time.sleep(0.12)
        rows.append({"ticker": r.ticker, "listed_float_shares": int(r.float_shares), "sec_shares_outstanding": sec,
                     "ratio_listed_to_sec": round(r.float_shares / sec, 3) if sec else None})
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "constituent_share_check.csv", index=False)
    m = out.dropna(subset=["ratio_listed_to_sec"])
    print(f"matched {len(m)} of {len(out)}; off by more than 20 percent: {int((~m.ratio_listed_to_sec.between(0.8, 1.2)).sum())}")


if __name__ == "__main__":
    main()
