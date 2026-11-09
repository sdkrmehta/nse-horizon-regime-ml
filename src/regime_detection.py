"""
regime_detection.py — Market Regime Detection
Project: Horizon- and Regime-Specific ML for Stock Return Prediction
Market: Indian Equity (NSE) exclusively

Classifies each trading day into one of three volatility regimes:
  Low (0), Medium (1), High (2)

Method:
  1. Compute 20-day rolling standard deviation of daily returns (realized vol)
  2. For each walk-forward split, compute quantile thresholds (33rd, 67th)
     using ONLY the training window data
  3. Assign regime labels to test data using those frozen thresholds
  4. Report observation counts per regime per period

LEAKAGE RULE (AGENTS.md §6 — Non-Negotiable):
  - Regime thresholds computed from training data ONLY
  - No future information used to label past decisions
  - Thresholds frozen per evaluation window

Usage:
    python src/regime_detection.py
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
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Volatility Computation
# ---------------------------------------------------------------------------

VOLATILITY_WINDOW = 20  # 20-day rolling window — pre-specified, not tuned

def compute_rolling_volatility(
    df: pd.DataFrame,
    window: int = VOLATILITY_WINDOW,
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """
    Compute rolling realized volatility per stock.

    Uses trailing 20-day standard deviation of log returns,
    annualized by multiplying by sqrt(252).

    This is backward-looking by construction — safe.
    """
    df = df.copy()
    result_frames = []

    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date").copy()

        # Daily log returns
        log_returns = np.log(group[price_col] / group[price_col].shift(1))

        # Rolling std (trailing window, ends at t)
        rolling_std = log_returns.rolling(window=window, min_periods=window).std()

        # Annualized realized volatility
        group["realized_vol"] = rolling_std * np.sqrt(252)

        result_frames.append(group)

    result = pd.concat(result_frames, ignore_index=True)
    logger.info("Computed %d-day rolling volatility for %d tickers",
                window, df["Ticker"].nunique())

    return result


# ---------------------------------------------------------------------------
# Regime Assignment
# ---------------------------------------------------------------------------

def compute_regime_thresholds(
    vol_series: pd.Series,
    quantiles: tuple[float, float] = (0.33, 0.67),
) -> tuple[float, float]:
    """
    Compute regime thresholds from a volatility series.

    Returns (low_threshold, high_threshold) such that:
      - vol <= low_threshold  → Low volatility (regime 0)
      - low < vol <= high     → Medium volatility (regime 1)
      - vol > high_threshold  → High volatility (regime 2)

    Parameters
    ----------
    vol_series : pd.Series
        Realized volatility values (training data only).
    quantiles : tuple of float
        Lower and upper quantile cutoffs (default: 33rd and 67th percentile).
    """
    clean = vol_series.dropna()
    low_thresh = clean.quantile(quantiles[0])
    high_thresh = clean.quantile(quantiles[1])
    return float(low_thresh), float(high_thresh)


def assign_regime_labels(
    vol_series: pd.Series,
    low_thresh: float,
    high_thresh: float,
) -> pd.Series:
    """
    Assign regime labels based on thresholds.

    Returns:
      0 = Low volatility
      1 = Medium volatility
      2 = High volatility
    """
    labels = pd.Series(np.nan, index=vol_series.index, dtype="Int64")
    valid = vol_series.notna()
    labels[valid & (vol_series <= low_thresh)] = 0
    labels[valid & (vol_series > low_thresh) & (vol_series <= high_thresh)] = 1
    labels[valid & (vol_series > high_thresh)] = 2
    return labels


REGIME_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def assign_regimes_walk_forward(
    df: pd.DataFrame,
    splits: list[dict],
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Assign regime labels for each walk-forward split.

    For each split:
      1. Extract realized_vol from the TRAINING period
      2. Compute quantile thresholds (33rd, 67th) from training data only
      3. Apply those thresholds to label BOTH training and test data
      4. Store thresholds and regime counts

    This ensures NO future information leaks into regime definitions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Date, Ticker, realized_vol.
    splits : list of dict
        Walk-forward split definitions from experiment.yaml.

    Returns
    -------
    df : pd.DataFrame
        With regime columns: regime_split_1, regime_split_2, ..., regime_split_6_final
    split_info : list of dict
        Threshold and count information per split.
    """
    df = df.copy()
    split_info = []

    for split in splits:
        label = split["label"]
        train_start = pd.Timestamp(split["train_start"])
        train_end = pd.Timestamp(split["train_end"])
        test_start = pd.Timestamp(split["test_start"])
        test_end = pd.Timestamp(split["test_end"])

        logger.info("Processing %s: train [%s → %s], test [%s → %s]",
                     label, train_start.date(), train_end.date(),
                     test_start.date(), test_end.date())

        # 1. Get training period volatility (across ALL stocks in the universe)
        train_mask = (df["Date"] >= train_start) & (df["Date"] <= train_end)
        train_vol = df.loc[train_mask, "realized_vol"]

        # 2. Compute thresholds from training data ONLY
        low_thresh, high_thresh = compute_regime_thresholds(train_vol)
        logger.info("  Thresholds (from training): low=%.4f, high=%.4f",
                     low_thresh, high_thresh)

        # 3. Apply thresholds to ALL data (train + test)
        regime_col = f"regime_{label}"
        df[regime_col] = assign_regime_labels(df["realized_vol"], low_thresh, high_thresh)

        # 4. Count regimes in test period
        test_mask = (df["Date"] >= test_start) & (df["Date"] <= test_end)
        test_regimes = df.loc[test_mask, regime_col].dropna()
        counts = test_regimes.value_counts().sort_index()

        info = {
            "split": label,
            "train_period": f"{train_start.date()} → {train_end.date()}",
            "test_period": f"{test_start.date()} → {test_end.date()}",
            "low_threshold": low_thresh,
            "high_threshold": high_thresh,
            "test_counts": {REGIME_NAMES.get(k, k): int(v) for k, v in counts.items()},
            "test_total": int(test_regimes.shape[0]),
            "test_proportions": {
                REGIME_NAMES.get(k, k): round(v / test_regimes.shape[0] * 100, 1)
                for k, v in counts.items()
            },
        }
        split_info.append(info)

        for regime_id, name in REGIME_NAMES.items():
            count = counts.get(regime_id, 0)
            pct = count / test_regimes.shape[0] * 100 if test_regimes.shape[0] > 0 else 0
            logger.info("  Test %s: %s = %d (%.1f%%)", label, name, count, pct)

    return df, split_info


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_regime_report(split_info: list[dict], output_path: Path) -> None:
    """Generate a markdown report of regime assignments."""
    lines = [
        "# Regime Detection Report",
        "",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Method**: 20-day rolling realized volatility with quantile thresholds (33rd/67th percentile)",
        f"**Regime Definition**: Low (≤ Q33), Medium (Q33–Q67), High (> Q67)",
        f"**Leakage Control**: Thresholds computed from training data ONLY per split",
        "",
        "---",
        "",
        "## Per-Split Regime Summary",
        "",
    ]

    for info in split_info:
        lines.append(f"### {info['split']}")
        lines.append("")
        lines.append(f"- **Training period**: {info['train_period']}")
        lines.append(f"- **Test period**: {info['test_period']}")
        lines.append(f"- **Thresholds**: Low ≤ {info['low_threshold']:.4f}, "
                      f"High > {info['high_threshold']:.4f}")
        lines.append("")
        lines.append("| Regime | Test Count | Test % |")
        lines.append("|--------|-----------|--------|")
        for name in ["Low", "Medium", "High"]:
            count = info["test_counts"].get(name, 0)
            pct = info["test_proportions"].get(name, 0)
            lines.append(f"| {name} | {count} | {pct}% |")
        lines.append(f"| **Total** | **{info['test_total']}** | **100%** |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Threshold stability across splits
    lines.append("## Threshold Stability Across Splits")
    lines.append("")
    lines.append("| Split | Low Threshold | High Threshold |")
    lines.append("|-------|--------------|----------------|")
    for info in split_info:
        lines.append(f"| {info['split']} | {info['low_threshold']:.4f} | "
                      f"{info['high_threshold']:.4f} |")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Regime report saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run regime detection pipeline."""
    logger.info("=" * 60)
    logger.info("REGIME DETECTION — Volatility-Based (20-day rolling)")
    logger.info("=" * 60)

    # Load config
    config = load_config()
    splits = config["walk_forward"]["splits"]

    # Load features+targets data
    data_path = PROCESSED_DIR / "features_targets.csv"
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path, parse_dates=["Date"])

    # Exclude benchmark (regime detection on individual stocks only)
    benchmark = config["data"]["benchmark"]
    stocks_df = df[df["Ticker"] != benchmark].copy()
    bench_df = df[df["Ticker"] == benchmark].copy()
    logger.info("Working with %d stock rows, %d benchmark rows",
                len(stocks_df), len(bench_df))

    # Step 1: Compute rolling volatility (if not already present)
    if "realized_vol" not in stocks_df.columns:
        stocks_df = compute_rolling_volatility(stocks_df)
        bench_df = compute_rolling_volatility(bench_df)
    else:
        logger.info("realized_vol column already exists, skipping computation")

    # Step 2: Assign regimes per walk-forward split
    logger.info("-" * 40)
    logger.info("Assigning regimes per walk-forward split...")
    stocks_df, split_info = assign_regimes_walk_forward(stocks_df, splits)

    # Also assign to benchmark using same thresholds (for reference)
    # Reuse the thresholds from stocks
    for i, split in enumerate(splits):
        label = split["label"]
        regime_col = f"regime_{label}"
        low_t = split_info[i]["low_threshold"]
        high_t = split_info[i]["high_threshold"]
        bench_df[regime_col] = assign_regime_labels(
            bench_df["realized_vol"], low_t, high_t
        )

    # Combine back
    combined = pd.concat([stocks_df, bench_df], ignore_index=True)
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Step 3: Save
    output_path = PROCESSED_DIR / "features_targets.csv"
    combined.to_csv(output_path, index=False)
    logger.info("Saved updated dataset to %s (%d rows × %d cols)",
                output_path, len(combined), combined.shape[1])

    # Step 4: Generate report
    report_path = PROJECT_ROOT / "reports" / "market_regime_report.md"
    generate_regime_report(split_info, report_path)

    # Step 5: Save regime info as JSON
    import json
    regime_json_path = PROCESSED_DIR / "regime_info.json"
    with open(regime_json_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, default=str)
    logger.info("Saved regime info to %s", regime_json_path)

    # Summary
    logger.info("=" * 60)
    logger.info("REGIME DETECTION COMPLETE")
    logger.info("  Method: 20-day rolling realized volatility")
    logger.info("  Thresholds: 33rd / 67th percentile (training data only)")
    logger.info("  Splits processed: %d", len(split_info))
    for info in split_info:
        logger.info("    %s: Low=%d, Med=%d, High=%d (test period)",
                     info["split"],
                     info["test_counts"].get("Low", 0),
                     info["test_counts"].get("Medium", 0),
                     info["test_counts"].get("High", 0))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
