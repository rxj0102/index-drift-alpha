"""
config.py — Central configuration for index-drift-alpha.

All tunable parameters, paths, and strategy constants live here.
Import this module rather than hard-coding values anywhere in src/.

Usage
-----
    from config import DRIFT_THRESHOLD_HARD, PATHS

Notes
-----
All filesystem paths are derived from the repository root so the project
is portable across machines.  Pass paths as ``pathlib.Path`` objects so
that downstream code avoids OS-specific separator issues.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root (everything else is relative to this)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
PATHS: dict[str, Path] = {
    "raw":                 ROOT / "data" / "raw",
    "processed":           ROOT / "data" / "processed",
    "reference":           ROOT / "data" / "reference",
    "outputs":             ROOT / "outputs",
    # Reference seed files
    "constituents":        ROOT / "data" / "reference" / "index_constituents.csv",
    "rebalance_calendar":  ROOT / "data" / "reference" / "rebalance_calendar.csv",
    "etf_index_map":       ROOT / "data" / "reference" / "etf_index_map.csv",
    # Processed artefacts (written at runtime)
    "actionable_drift":    ROOT / "data" / "processed" / "actionable_drift.csv",
}

# ---------------------------------------------------------------------------
# Target indices and their ETF proxies
# ---------------------------------------------------------------------------
INDEX_ETF_MAP: dict[str, dict] = {
    "SP500": {
        "etf": "SPY",
        "rebalance_months": [3, 6, 9, 12],
        "description": "S&P 500 — large-cap, highly HFT-crowded rebalancing",
    },
    "SP400": {
        "etf": "MDY",
        "rebalance_months": [3, 6, 9, 12],
        "description": "S&P 400 MidCap — primary universe; lower HFT penetration",
    },
    "Russell1000": {
        "etf": "IWB",
        "rebalance_months": [6],
        "description": "Russell 1000 — annual June reconstitution",
    },
    "Russell2000": {
        "etf": "IWM",
        "rebalance_months": [6],
        "description": "Russell 2000 — annual June reconstitution; highest impact",
    },
}

# ---------------------------------------------------------------------------
# Drift signal parameters
# ---------------------------------------------------------------------------
# Raw drift thresholds (in weight-point terms, not bps of return)
DRIFT_THRESHOLD_SOFT: float = 0.0015   # 15 bps weight drift → elevated watch
DRIFT_THRESHOLD_HARD: float = 0.0030   # 30 bps weight drift → strong signal

# Rolling window for drift z-score normalisation
DRIFT_LOOKBACK_DAYS: int = 20

# Z-score threshold to trigger an actionable signal (WATCH → BUY/SELL)
DRIFT_ZSCORE_THRESHOLD: float = 1.5

# Winsorisation cap applied to z-scores before any downstream use
DRIFT_ZSCORE_WINSOR_CAP: float = 4.0

# ---------------------------------------------------------------------------
# Signal and portfolio parameters
# ---------------------------------------------------------------------------
SIGNAL_QUINTILE: float = 0.80          # top/bottom 20% for long/short book
HOLDING_PERIOD_DAYS: int = 10          # expected mean-reversion window
REBALANCE_BLACKOUT_DAYS: int = 5       # ± days around scheduled rebalance

# Minimum names required in each book before a day's signal is emitted
MIN_BOOK_SIZE: int = 5

# Signal decay horizons to evaluate (trading days)
SIGNAL_DECAY_HORIZONS: list[int] = [1, 3, 5, 10, 15, 20]

# IC benchmark for a meaningful microstructure signal
IC_MEANINGFUL_THRESHOLD: float = 0.04  # Spearman IC > 4% considered material

# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------
BACKTEST_START: str = "2015-01-01"
BACKTEST_END: str   = "2023-12-31"
UNIVERSE: str       = "SP400"          # default: mid-cap, less HFT crowding

# One-way transaction cost assumption (mid-cap liquidity)
# Stress-test at 15 bps and 25 bps for sensitivity analysis
TRANSACTION_COST_BPS: float = 8.0

# High daily turnover threshold — above this the signal is too noisy
TURNOVER_RED_FLAG: float = 0.40        # 40 % daily turnover = red flag

# ---------------------------------------------------------------------------
# ETF flow parameters
# ---------------------------------------------------------------------------
CREATION_UNIT_SIZE: int = 50_000       # standard ETF creation/redemption unit

# ---------------------------------------------------------------------------
# Rebalance window analysis
# ---------------------------------------------------------------------------
# Sub-periods around each scheduled rebalance (trading days, relative to event)
WINDOW_PRE_START:  int = -20
WINDOW_PRE_END:    int = -6
WINDOW_NEAR_START: int = -5
WINDOW_NEAR_END:   int = -1
WINDOW_POST_START: int = 1
WINDOW_POST_END:   int = 10

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("DRIFT_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
