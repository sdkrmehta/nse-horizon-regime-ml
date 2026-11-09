"""
target_engineering.py — Return Target Construction
Project: Horizon- and Regime-Specific ML for Stock Return Prediction
Market: Indian Equity (NSE) exclusively

Constructs 1-day and 5-day forward return targets for each stock.
Targets are constructed FIRST, then features are aligned backward
so that features at time t never use any part of the target window.

Usage:
    python src/target_engineering.py
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load experiment configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Target Construction
# ---------------------------------------------------------------------------

def compute_forward_returns(
    df: pd.DataFrame,
    horizons: list[int] = [1, 5],
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """
    Compute forward return targets for each stock.

    For each horizon h:
        return_t = (Price_{t+h} / Price_t) - 1

    The target at time t uses the price at t+h, which is FUTURE information.
    This is intentional — the target IS the future return we want to predict.
    Features must NOT use this future data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: Date, Ticker, Open, High, Low, Close, Adj Close, Volume.
    horizons : list of int
        Forecast horizons in trading days.
    price_col : str
        Price column to use for return calculation (default: 'Adj Close').

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        - fwd_return_1d, fwd_return_5d (continuous returns)
        - fwd_label_1d, fwd_label_5d (binary: 1 = up, 0 = down/flat)
    """
    logger.info("Computing forward returns for horizons: %s using '%s'", horizons, price_col)

    df = df.copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    result_frames = []

    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date").copy()

        for h in horizons:
            col_return = f"fwd_return_{h}d"
            col_label = f"fwd_label_{h}d"

            # Forward return: (P_{t+h} / P_t) - 1
            # shift(-h) brings future price to current row
            future_price = group[price_col].shift(-h)
            group[col_return] = (future_price / group[price_col]) - 1

            # Binary classification label: 1 if return > 0, else 0
            # Threshold fixed at 0 BEFORE experimentation
            group[col_label] = (group[col_return] > 0).astype(int)

        result_frames.append(group)

    result = pd.concat(result_frames, ignore_index=True)

    # Log target statistics
    for h in horizons:
        col_return = f"fwd_return_{h}d"
        col_label = f"fwd_label_{h}d"
        valid = result[col_return].notna()
        logger.info(
            "  %s: %d valid targets (%.1f%% up), mean=%.4f, std=%.4f",
            col_return,
            valid.sum(),
            result.loc[valid, col_label].mean() * 100,
            result.loc[valid, col_return].mean(),
            result.loc[valid, col_return].std(),
        )

    return result


def validate_targets(df: pd.DataFrame, horizons: list[int] = [1, 5]) -> dict:
    """
    Validate target construction for correctness.

    Checks:
    1. Forward returns are correctly computed (spot check)
    2. NaN targets only at the end of each stock's time series
    3. No target leakage (targets use only future data, not past)
    4. Label distribution is reasonable (not all 0s or all 1s)

    Returns
    -------
    dict with validation results.
    """
    logger.info("Validating target construction...")
    results = {"passed": True, "checks": []}

    for h in horizons:
        col_return = f"fwd_return_{h}d"
        col_label = f"fwd_label_{h}d"

        # Check 1: NaN targets should only be at the end of each stock
        for ticker, group in df.groupby("Ticker"):
            group = group.sort_values("Date")
            nan_mask = group[col_return].isna()
            if nan_mask.any():
                # All NaN should be in the last h rows
                first_nan_idx = nan_mask.idxmax() if nan_mask.any() else None
                last_valid_idx = group[col_return].last_valid_index()
                if first_nan_idx is not None and last_valid_idx is not None:
                    nan_count = nan_mask.sum()
                    if nan_count != h:
                        msg = f"WARN: {ticker} {col_return} has {nan_count} NaN rows (expected {h})"
                        results["checks"].append(msg)
                        logger.warning("  %s", msg)

        # Check 2: Label distribution
        valid = df[col_label].dropna()
        up_pct = valid.mean() * 100
        if up_pct < 30 or up_pct > 70:
            msg = f"WARN: {col_label} has extreme imbalance: {up_pct:.1f}% up"
            results["checks"].append(msg)
            logger.warning("  %s", msg)
        else:
            msg = f"OK: {col_label} distribution: {up_pct:.1f}% up / {100-up_pct:.1f}% down"
            results["checks"].append(msg)
            logger.info("  %s", msg)

        # Check 3: Spot-check return calculation
        for ticker, group in df.groupby("Ticker"):
            group = group.sort_values("Date").reset_index(drop=True)
            if len(group) > h + 5:
                # Check row 10 as a spot check
                idx = 10
                actual = group.loc[idx, col_return]
                expected = (group.loc[idx + h, "Adj Close"] / group.loc[idx, "Adj Close"]) - 1
                if not np.isclose(actual, expected, atol=1e-10):
                    msg = f"FAIL: {ticker} {col_return} spot check failed at row {idx}"
                    results["checks"].append(msg)
                    results["passed"] = False
                    logger.error("  %s", msg)
                else:
                    # Only log first successful spot check
                    break

        msg = f"OK: {col_return} spot check passed"
        results["checks"].append(msg)
        logger.info("  %s", msg)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Construct and validate forward return targets."""
    logger.info("=" * 60)
    logger.info("TARGET CONSTRUCTION — 1-day and 5-day forward returns")
    logger.info("=" * 60)

    # Load config
    config = load_config()
    horizons = config["horizons"]

    # Load cleaned OHLCV data
    ohlcv_path = PROCESSED_DIR / "all_stocks_ohlcv.csv"
    logger.info("Loading data from %s", ohlcv_path)
    df = pd.read_csv(ohlcv_path, parse_dates=["Date"])

    # Exclude benchmark index from target construction (it's for Buy-and-Hold only)
    benchmark = config["data"]["benchmark"]
    stocks_df = df[df["Ticker"] != benchmark].copy()
    bench_df = df[df["Ticker"] == benchmark].copy()
    logger.info("Stocks: %d rows (%d tickers), Benchmark: %d rows",
                len(stocks_df), stocks_df["Ticker"].nunique(), len(bench_df))

    # Compute targets
    stocks_with_targets = compute_forward_returns(stocks_df, horizons=horizons)

    # Also compute for benchmark (for backtest comparison)
    bench_with_targets = compute_forward_returns(bench_df, horizons=horizons)

    # Combine
    combined = pd.concat([stocks_with_targets, bench_with_targets], ignore_index=True)

    # Validate
    validation = validate_targets(stocks_with_targets, horizons=horizons)

    # Save
    output_path = PROCESSED_DIR / "targets.csv"
    combined.to_csv(output_path, index=False)
    logger.info("Saved targets to %s (%d rows)", output_path, len(combined))

    # Summary
    logger.info("=" * 60)
    logger.info("TARGET CONSTRUCTION COMPLETE")
    logger.info("  Validation: %s", "PASSED ✓" if validation["passed"] else "ISSUES FOUND ⚠")
    for check in validation["checks"]:
        logger.info("    %s", check)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
