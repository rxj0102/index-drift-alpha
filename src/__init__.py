"""
index-drift-alpha
=================

Index Rebalancing Slippage Harvesting via Float-Adjusted Weight Drift.

Package layout
--------------
data_loader   — price fetching, SEC EDGAR N-CEN, constituent & calendar loaders
drift_engine  — core math: realized weights, drift, z-scores, flow estimates
signal        — signal construction, decay, IC, ETF-flow confirmation
backtest      — long/short backtests, event studies, performance metrics
visualize     — all publication-quality charts (saved to outputs/)
utils         — lag_dataframe, winsorize_df (look-ahead-bias guards)

Entry point
-----------
Run the drift computation standalone smoke test::

    python -m src.drift_engine --index SP400
"""

__version__ = "0.1.0"
__author__ = "index-drift-alpha"
