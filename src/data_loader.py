"""
data_loader.py — Market data acquisition, caching, and reference file loaders.

All functions cache aggressively to ``data/processed/`` (Parquet) or
``data/raw/`` (JSON) so that downstream notebook runs do not hammer external
APIs.  Cache validity is checked by date range coverage, not just file
existence.

External dependencies
---------------------
* yfinance      — adjusted close prices, ETF shares outstanding
* SEC EDGAR     — N-CEN filings via EDGAR full-text search API
* requests      — HTTP for EDGAR; no API key required

Known limitations
-----------------
See individual function docstrings for detailed caveats.  The most
significant are:

1. yfinance ``shares_outstanding`` for ETFs reports total authorised shares,
   not actual outstanding float.  It is a directional proxy only.
2. SEC N-CEN filings are annual, not daily — they calibrate but do not
   replace a real-time creation/redemption feed.
3. Float shares in ``index_constituents.csv`` are held constant at the
   last reconstitution date.  Intraday buybacks and secondary offerings
   are not captured.

Usage
-----
    from src.data_loader import fetch_constituent_prices, load_index_constituents
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parquet_path(stem: str) -> Path:
    """Return a canonical Parquet path in data/processed/."""
    return config.PATHS["processed"] / f"{stem}.parquet"


def _json_path(stem: str) -> Path:
    """Return a canonical JSON path in data/raw/."""
    return config.PATHS["raw"] / f"{stem}.json"


def _covers_range(df: pd.DataFrame, start: str, end: str) -> bool:
    """Check whether a cached DataFrame's DatetimeIndex covers [start, end]."""
    if df.empty:
        return False
    idx = pd.to_datetime(df.index)
    return idx.min() <= pd.Timestamp(start) and idx.max() >= pd.Timestamp(end)


# ---------------------------------------------------------------------------
# 1. fetch_constituent_prices
# ---------------------------------------------------------------------------

def fetch_constituent_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers via yfinance.

    Prices are stored in wide format (date × ticker).  A Parquet cache in
    ``data/processed/`` is written on first call and reloaded on subsequent
    calls that cover the same date range.

    Parameters
    ----------
    tickers : list[str]
        List of equity ticker symbols.
    start : str
        Start date string in ISO format (``"YYYY-MM-DD"``).
    end : str
        End date string in ISO format (``"YYYY-MM-DD"``).

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame: index = trading dates (``pd.DatetimeIndex``),
        columns = tickers, values = adjusted close price (USD).

    Notes
    -----
    * Tickers with more than 5 % missing data are logged as warnings.
      Missing data degrades drift calculations because the denominator of the
      realized-weight formula (Σ price × float_shares) becomes understated,
      causing all remaining constituents' realized weights to appear inflated.
    * The cache file is named ``prices_{start}_{end}.parquet``.  If a partial
      cache exists but does not cover the full requested range, data is
      re-downloaded.
    * yfinance rate limits at ~2000 requests/hour.  For large universes
      (>200 tickers) use batch download, which this function does via
      ``yf.download(tickers, ...)`` in a single call.
    """
    cache_stem = f"prices_{start}_{end}"
    cache_file = _parquet_path(cache_stem)

    if cache_file.exists():
        logger.info("Loading price cache from %s", cache_file)
        df = pd.read_parquet(cache_file)
        if _covers_range(df, start, end):
            missing_rate = df.isna().mean()
            noisy = missing_rate[missing_rate > 0.05]
            if not noisy.empty:
                logger.warning(
                    "Tickers with >5%% missing data in cache: %s",
                    noisy.index.tolist(),
                )
            return df

    logger.info(
        "Downloading prices for %d tickers [%s → %s]", len(tickers), start, end
    )
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns when multiple tickers requested
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=0)
    else:
        df = raw[["Close"]] if "Close" in raw.columns else raw

    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index)

    # Validate missing data
    missing_rate = df.isna().mean()
    noisy = missing_rate[missing_rate > 0.05]
    if not noisy.empty:
        logger.warning(
            "Tickers with >5%% missing data — drift signal quality degraded: %s",
            noisy.index.tolist(),
        )

    config.PATHS["processed"].mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)
    logger.info("Price data cached to %s (%d rows × %d cols)", cache_file,
                df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# 2. fetch_etf_aum_history
# ---------------------------------------------------------------------------

def fetch_etf_aum_history(etf_tickers: list[str]) -> pd.DataFrame:
    """Proxy daily ETF AUM as price × shares_outstanding via yfinance.

    Parameters
    ----------
    etf_tickers : list[str]
        List of ETF ticker symbols (e.g. ``["SPY", "MDY", "IWM"]``).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ``[date, etf_ticker, aum_usd, shares_outstanding]``.

    Notes
    -----
    **Critical limitation** — yfinance ``fast_info.shares`` for ETFs returns
    total *authorised* shares, not shares *outstanding* (actually in investor
    hands).  The correct figure is the daily creation/redemption unit count
    reported to the fund administrator, which is only available via:

    * Fund sponsor daily NAV and shares-outstanding feeds (Bloomberg/FactSet).
    * SEC Form N-CEN (annual filing) — see :func:`fetch_sec_ncen_flows`.
    * DTCC CNS/DTC records (institutional access only).

    For this research prototype the yfinance proxy provides a directional
    AUM series that is useful for ranking ETFs by size and for rescaling
    flow estimates, but the absolute dollar levels will diverge from the
    true AUM.  A reasonable calibration approach is to anchor the series
    to the most recently available N-CEN share count and propagate forward
    using the yfinance daily ratio.
    """
    records = []
    for ticker in etf_tickers:
        logger.info("Fetching AUM proxy for %s", ticker)
        try:
            etf = yf.Ticker(ticker)
            hist = etf.history(period="max", auto_adjust=True)
            if hist.empty:
                logger.warning("No price history for %s", ticker)
                continue

            # shares_outstanding is a scalar from fast_info
            try:
                shares_out = etf.fast_info.shares
            except Exception:
                shares_out = None

            if shares_out is None or shares_out == 0:
                logger.warning(
                    "shares_outstanding unavailable for %s — AUM proxy will be NaN",
                    ticker,
                )
                shares_out = float("nan")

            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            for date_idx, row in hist.iterrows():
                records.append(
                    {
                        "date": date_idx,
                        "etf_ticker": ticker,
                        "aum_usd": row["Close"] * shares_out,
                        "shares_outstanding": shares_out,
                    }
                )
        except Exception as exc:
            logger.error("Failed to fetch AUM for %s: %s", ticker, exc)

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["etf_ticker", "date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. fetch_sec_ncen_flows
# ---------------------------------------------------------------------------

def fetch_sec_ncen_flows(etf_ticker: str) -> pd.DataFrame:
    """Query SEC EDGAR full-text search for N-CEN filings for an ETF.

    N-CEN (Annual Report for Registered Investment Companies) is filed once
    per year and contains total shares outstanding, creation units issued,
    and creation units redeemed.  This function parses the EDGAR EFTS API
    response and extracts key metrics.

    Parameters
    ----------
    etf_ticker : str
        ETF ticker symbol (e.g. ``"MDY"``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ``[period_end, shares_outstanding, units_issued, units_redeemed,
        net_flow_units]``.
        Returns an empty DataFrame if no filings are found.

    Notes
    -----
    **Frequency limitation** — N-CEN is filed annually, typically 60–75 days
    after the fund's fiscal year-end.  It should be used as a *calibration
    anchor* for the yfinance daily AUM proxy (see :func:`fetch_etf_aum_history`),
    not as a real-time flow signal.

    True real-time creation/redemption data requires access to:

    * Authorised Participant (AP) transaction records.
    * Fund administrator daily NAV files.
    * Bloomberg ETF flow estimates (``ETFG <ETF> <GO>``).

    The EDGAR EFTS API URL pattern used here is::

        https://efts.sec.gov/LATEST/search-index?q={etf_ticker}&forms=N-CEN

    Response is JSON; each hit contains a filing URL from which the XML
    document can be retrieved.  Full XML parsing of N-CEN forms is complex
    (schema varies by fiscal year); this implementation extracts a subset of
    commonly available fields.

    Results are cached to ``data/raw/ncen_{etf_ticker}.json``.
    """
    cache_file = _json_path(f"ncen_{etf_ticker}")

    if cache_file.exists():
        logger.info("Loading N-CEN cache for %s from %s", etf_ticker, cache_file)
        with open(cache_file) as fh:
            cached = json.load(fh)
        return _parse_ncen_cache(cached)

    url = (
        f"https://efts.sec.gov/LATEST/search-index?q={etf_ticker}&forms=N-CEN"
        f"&dateRange=custom&startdt=2015-01-01&enddt=2024-01-01"
    )
    logger.info("Querying EDGAR EFTS for N-CEN filings: %s", url)

    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "research-bot/1.0"})
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        logger.error("EDGAR query failed for %s: %s", etf_ticker, exc)
        return pd.DataFrame(columns=[
            "period_end", "shares_outstanding", "units_issued",
            "units_redeemed", "net_flow_units",
        ])

    # Cache raw response
    config.PATHS["raw"].mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("N-CEN response cached to %s", cache_file)

    return _parse_ncen_cache(payload)


def _parse_ncen_cache(payload: dict) -> pd.DataFrame:
    """Parse EDGAR EFTS JSON payload into a tidy DataFrame.

    The N-CEN XML schema is filing-date dependent and highly nested.
    This parser extracts what is reliably available in the EFTS search
    result metadata; full field extraction requires fetching and parsing
    each individual filing's primary document XML.
    """
    hits = payload.get("hits", {}).get("hits", [])
    records = []
    for hit in hits:
        src = hit.get("_source", {})
        # file_date is the best available proxy for period_end here
        file_date = src.get("file_date", None)
        if file_date:
            records.append(
                {
                    "period_end": pd.Timestamp(file_date),
                    # Fields below require parsing the actual XML; set NaN
                    # unless a more comprehensive parser is implemented.
                    "shares_outstanding": float("nan"),
                    "units_issued": float("nan"),
                    "units_redeemed": float("nan"),
                    "net_flow_units": float("nan"),
                }
            )
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("period_end").reset_index(drop=True)
    logger.info("Parsed %d N-CEN records from EDGAR payload.", len(df))
    return df


# ---------------------------------------------------------------------------
# 4. load_index_constituents
# ---------------------------------------------------------------------------

def load_index_constituents(index_name: str) -> pd.DataFrame:
    """Load index constituent file and validate weight integrity.

    Parameters
    ----------
    index_name : str
        Index identifier, e.g. ``"SP400"``, ``"SP500"``.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame for the requested index with columns:
        ``[ticker, company_name, index_name, target_weight, float_shares,
        market_cap, sector, gics_sub_industry, as_of_date]``.

    Notes
    -----
    * Raises a ``FileNotFoundError`` if the reference CSV is missing.
    * Emits a ``logging.WARNING`` (not an exception) if weights do not sum
      to 1.0 ± 0.001, because minor floating-point rounding across many
      tickers is expected.  A deviation exceeding 0.5 % is a data quality
      issue that should be investigated.
    * ``float_shares`` is held constant at the reconstitution snapshot date
      in ``as_of_date``.  In a production system, quarterly 13F/13G filings
      and SEC Form 144 data would refresh this figure at higher frequency.
      Stale float shares cause the denominator of the realized-weight formula
      to drift from true index float, introducing a systematic bias in
      measured drift.
    """
    ref_path = config.PATHS["constituents"]
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Constituent reference file not found: {ref_path}"
        )

    df = pd.read_csv(ref_path)
    df = df[df["index_name"] == index_name].copy()

    if df.empty:
        raise ValueError(f"No constituents found for index '{index_name}'.")

    df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    weight_sum = df["target_weight"].sum()
    if abs(weight_sum - 1.0) > 0.001:
        logger.warning(
            "Constituent weights for '%s' sum to %.6f (expected 1.0 ± 0.001). "
            "Investigate data quality.",
            index_name, weight_sum,
        )
    else:
        logger.debug("Constituent weights for '%s' sum to %.6f — OK.", index_name, weight_sum)

    logger.info(
        "Loaded %d constituents for '%s' (as of %s).",
        len(df), index_name, df["as_of_date"].max().date(),
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. load_rebalance_calendar
# ---------------------------------------------------------------------------

def load_rebalance_calendar(index_name: str) -> pd.DataFrame:
    """Load scheduled rebalance events and compute blackout windows.

    Parameters
    ----------
    index_name : str
        Index identifier, e.g. ``"SP400"``.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with columns:
        ``[index_name, rebalance_date, rebalance_type, announcement_date,
        blackout_start, blackout_end]``.

    Notes
    -----
    The blackout window (``REBALANCE_BLACKOUT_DAYS`` either side of the
    rebalance date) exists because:

    1. In the days immediately before a scheduled rebalance, index fund
       managers must trade to reset constituent weights.  This demand is
       perfectly predictable — every front-runner on the street is already
       positioned to capture this flow.  Adding to the crowd in this window
       increases execution costs and reduces alpha.
    2. In the days immediately after, residual correction flows and
       post-announcement price pressure resolve in an unpredictable direction.

    The blackout window is symmetric for simplicity; in practice, the PRE
    window is more dangerous than the POST window for this strategy.
    """
    cal_path = config.PATHS["rebalance_calendar"]
    if not cal_path.exists():
        raise FileNotFoundError(
            f"Rebalance calendar not found: {cal_path}"
        )

    df = pd.read_csv(cal_path)
    df = df[df["index_name"] == index_name].copy()

    if df.empty:
        raise ValueError(f"No rebalance events found for index '{index_name}'.")

    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
    df["announcement_date"] = pd.to_datetime(df["announcement_date"])

    n = config.REBALANCE_BLACKOUT_DAYS
    df["blackout_start"] = df["rebalance_date"] - pd.Timedelta(days=n)
    df["blackout_end"] = df["rebalance_date"] + pd.Timedelta(days=n)

    df = df.sort_values("rebalance_date").reset_index(drop=True)
    logger.info(
        "Loaded %d rebalance events for '%s' (±%d day blackout).",
        len(df), index_name, n,
    )
    return df


# ---------------------------------------------------------------------------
# 6. is_blackout_date
# ---------------------------------------------------------------------------

def is_blackout_date(
    date: pd.Timestamp,
    rebalance_calendar: pd.DataFrame,
) -> bool:
    """Return True if *date* falls within any scheduled rebalance blackout.

    Parameters
    ----------
    date : pd.Timestamp
        The date to test.
    rebalance_calendar : pd.DataFrame
        Output of :func:`load_rebalance_calendar` — must contain
        ``blackout_start`` and ``blackout_end`` columns.

    Returns
    -------
    bool
        ``True`` if the date is in any blackout window; ``False`` otherwise.

    Notes
    -----
    This function is called inside :func:`drift_engine.flag_actionable_drift`
    for **every** date in the backtest period.  For performance it operates
    on pre-computed ``blackout_start``/``blackout_end`` columns rather than
    computing offsets at call time.

    Trading near scheduled rebalances is the "existential timing risk" of
    this strategy.  The rebalancing pressure that drives the alpha is most
    reliably predictable 6–20 days *before* the rebalance date.  Within
    5 days of the event, execution footprint from other front-runners
    dominates and the expected edge net of costs turns negative.
    """
    mask = (rebalance_calendar["blackout_start"] <= date) & \
           (date <= rebalance_calendar["blackout_end"])
    return bool(mask.any())
