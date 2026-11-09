"""
data_loader.py — Data Acquisition & Audit Module
Project: Horizon- and Regime-Specific ML for Stock Return Prediction
Market: Indian Equity (NSE) exclusively

This module downloads daily OHLCV data from Yahoo Finance for NSE-listed
Indian stocks, audits data quality, and saves raw + cleaned datasets.

Usage:
    python src/data_loader.py
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load experiment configuration from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return config


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_ohlcv(tickers: list[str], start: str, end: str,
                   output_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV data from Yahoo Finance for each ticker.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance ticker symbols (e.g., 'RELIANCE.NS').
    start, end : str
        Date range in 'YYYY-MM-DD' format.
    output_dir : Path
        Directory to save raw CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from ticker to its OHLCV DataFrame.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    failed = []

    for ticker in tickers:
        logger.info("Downloading %s  [%s → %s] ...", ticker, start, end)
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )

            if df.empty:
                logger.warning("  ⚠ No data returned for %s", ticker)
                failed.append(ticker)
                continue

            # Flatten MultiIndex columns if present (yfinance sometimes adds ticker level)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Ensure index is DatetimeIndex named 'Date'
            df.index.name = "Date"
            df = df.sort_index()

            # Save raw CSV
            safe_name = ticker.replace("^", "").replace(".", "_")
            csv_path = output_dir / f"{safe_name}.csv"
            df.to_csv(csv_path)
            logger.info("  ✓ %s — %d rows  [%s → %s]",
                        ticker, len(df),
                        df.index.min().strftime("%Y-%m-%d"),
                        df.index.max().strftime("%Y-%m-%d"))

            data[ticker] = df

        except Exception as e:
            logger.error("  ✗ Failed to download %s: %s", ticker, e)
            failed.append(ticker)

    if failed:
        logger.warning("Failed tickers: %s", failed)

    return data


def save_download_metadata(tickers: list[str], start: str, end: str,
                           output_dir: Path, data: dict) -> None:
    """Save download provenance metadata as JSON."""
    try:
        import yfinance as yf
        yf_version = yf.__version__
    except Exception:
        yf_version = "unknown"

    metadata = {
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance (yfinance)",
        "yfinance_version": yf_version,
        "date_range": {"start": start, "end": end},
        "tickers_requested": tickers,
        "tickers_downloaded": list(data.keys()),
        "tickers_failed": [t for t in tickers if t not in data],
        "rows_per_ticker": {t: len(df) for t, df in data.items()},
        "market": "Indian Equity — NSE (National Stock Exchange of India)",
        "currency": "INR",
        "timezone": "Asia/Kolkata (IST, UTC+5:30)",
    }

    meta_path = output_dir / "download_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved download metadata to %s", meta_path)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_single_stock(ticker: str, df: pd.DataFrame) -> dict:
    """
    Run quality checks on a single stock's OHLCV DataFrame.

    Returns a dict with audit results.
    """
    result = {
        "ticker": ticker,
        "rows": len(df),
        "date_start": str(df.index.min().date()) if len(df) > 0 else None,
        "date_end": str(df.index.max().date()) if len(df) > 0 else None,
        "issues": [],
    }

    if df.empty:
        result["issues"].append("EMPTY: No data returned")
        return result

    # 1. Check date ordering
    if not df.index.is_monotonic_increasing:
        result["issues"].append("DATE_ORDER: Dates are not strictly ascending")

    # 2. Check for duplicate dates
    dup_count = df.index.duplicated().sum()
    result["duplicate_dates"] = int(dup_count)
    if dup_count > 0:
        result["issues"].append(f"DUPLICATES: {dup_count} duplicate date(s) found")

    # 3. Check for NaN/null values per column
    nan_counts = df.isna().sum().to_dict()
    result["nan_per_column"] = {k: int(v) for k, v in nan_counts.items()}
    total_nans = sum(nan_counts.values())
    if total_nans > 0:
        result["issues"].append(f"MISSING: {total_nans} total NaN values across columns")

    # 4. Check for zero-volume days
    if "Volume" in df.columns:
        zero_vol = (df["Volume"] == 0).sum()
        result["zero_volume_days"] = int(zero_vol)
        if zero_vol > 0:
            result["issues"].append(f"ZERO_VOLUME: {zero_vol} days with zero volume")

    # 5. Check for negative prices
    price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df.columns]
    for col in price_cols:
        neg = (df[col] < 0).sum()
        if neg > 0:
            result["issues"].append(f"NEGATIVE_PRICE: {neg} negative values in {col}")

    # 6. Adj Close vs Close divergence → corporate actions
    if "Close" in df.columns and "Adj Close" in df.columns:
        ratio = df["Adj Close"] / df["Close"]
        # When ratio changes significantly, it indicates a split or dividend
        ratio_changes = ratio.diff().abs()
        big_changes = (ratio_changes > 0.01).sum()
        result["adj_close_divergence_days"] = int(big_changes)
        if big_changes > 0:
            result["issues"].append(
                f"CORPORATE_ACTIONS: {big_changes} days with significant Adj Close/Close divergence"
            )

    # 7. Trading days estimate (approximate — ~245 trading days per year in India)
    if len(df) > 0:
        years = (df.index.max() - df.index.min()).days / 365.25
        expected_days = int(years * 245)
        result["expected_trading_days"] = expected_days
        result["actual_trading_days"] = len(df)
        gap = expected_days - len(df)
        if abs(gap) > expected_days * 0.05:  # >5% deviation
            result["issues"].append(
                f"TRADING_DAYS: Expected ~{expected_days}, got {len(df)} (gap={gap})"
            )

    # 8. Compute basic price stats
    if "Close" in df.columns:
        result["close_min"] = float(df["Close"].min())
        result["close_max"] = float(df["Close"].max())
        result["close_mean"] = float(df["Close"].mean())

    result["status"] = "PASS" if len(result["issues"]) == 0 else "REVIEW"
    return result


def audit_all_stocks(data: dict[str, pd.DataFrame]) -> list[dict]:
    """Run audit on all downloaded stocks."""
    logger.info("Running data quality audit on %d tickers...", len(data))
    results = []
    for ticker, df in data.items():
        audit = audit_single_stock(ticker, df)
        status_icon = "✓" if audit["status"] == "PASS" else "⚠"
        logger.info("  %s %s — %d rows, %d issue(s)",
                     status_icon, ticker, audit["rows"], len(audit["issues"]))
        results.append(audit)
    return results


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

def clean_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Clean and combine all stock DataFrames into a single long-format DataFrame.

    Steps:
    - Remove duplicate dates (keep first)
    - Sort by date
    - Forward-fill small gaps (≤3 days), leave larger gaps as NaN
    - Add ticker column
    - Combine into one DataFrame

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns: Date, Ticker, Open, High, Low, Close, Adj Close, Volume
    """
    frames = []

    for ticker, df in data.items():
        df = df.copy()

        # Remove duplicates
        df = df[~df.index.duplicated(keep="first")]

        # Sort by date
        df = df.sort_index()

        # Forward-fill small gaps (up to 3 consecutive NaN days)
        df = df.ffill(limit=3)

        # Add ticker column
        df["Ticker"] = ticker

        # Reset index to make Date a column
        df = df.reset_index()

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    logger.info("Combined cleaned dataset: %d rows × %d columns, %d tickers",
                len(combined), combined.shape[1], combined["Ticker"].nunique())

    return combined


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_audit_report(audit_results: list[dict], output_path: Path) -> None:
    """Generate a markdown audit report from audit results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Data Audit Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Market**: Indian Equity — NSE (National Stock Exchange of India)",
        f"**Currency**: INR | **Timezone**: IST (UTC+5:30)",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total tickers audited | {len(audit_results)} |",
        f"| Passed (no issues) | {sum(1 for a in audit_results if a['status'] == 'PASS')} |",
        f"| Needs review | {sum(1 for a in audit_results if a['status'] == 'REVIEW')} |",
        "",
        "---",
        "",
        "## Per-Stock Details",
        "",
    ]

    for audit in audit_results:
        status_icon = "✅" if audit["status"] == "PASS" else "⚠️"
        lines.append(f"### {status_icon} {audit['ticker']}")
        lines.append("")
        lines.append(f"| Item | Value |")
        lines.append(f"|------|-------|")
        lines.append(f"| Rows | {audit['rows']} |")
        lines.append(f"| Date range | {audit.get('date_start', '—')} → {audit.get('date_end', '—')} |")
        lines.append(f"| Duplicate dates | {audit.get('duplicate_dates', 0)} |")
        lines.append(f"| Expected trading days | {audit.get('expected_trading_days', '—')} |")
        lines.append(f"| Actual trading days | {audit.get('actual_trading_days', '—')} |")

        if "zero_volume_days" in audit:
            lines.append(f"| Zero-volume days | {audit['zero_volume_days']} |")
        if "adj_close_divergence_days" in audit:
            lines.append(f"| Corporate action signals | {audit['adj_close_divergence_days']} |")
        if "close_min" in audit:
            lines.append(f"| Close price range | ₹{audit['close_min']:.2f} – ₹{audit['close_max']:.2f} |")
            lines.append(f"| Close price mean | ₹{audit['close_mean']:.2f} |")

        # NaN breakdown
        if "nan_per_column" in audit:
            nan_items = {k: v for k, v in audit["nan_per_column"].items() if v > 0}
            if nan_items:
                lines.append(f"| NaN values | {nan_items} |")

        lines.append("")

        if audit["issues"]:
            lines.append("**Issues:**")
            for issue in audit["issues"]:
                lines.append(f"- {issue}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Audit report saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Orchestrate the full data acquisition and audit pipeline."""
    logger.info("=" * 60)
    logger.info("DATA ACQUISITION & AUDIT — Indian Market (NSE)")
    logger.info("=" * 60)

    # 1. Load config
    config = load_config()
    tickers = config["data"]["universe"]
    benchmark = config["data"]["benchmark"]
    start = config["data"]["date_range"]["start"]
    end = config["data"]["date_range"]["end"]

    all_tickers = tickers + [benchmark]
    logger.info("Universe: %d stocks + 1 benchmark = %d tickers",
                len(tickers), len(all_tickers))
    logger.info("Date range: %s → %s", start, end)

    # 2. Download raw data
    logger.info("-" * 40)
    logger.info("STEP 1: Downloading OHLCV data...")
    data = download_ohlcv(all_tickers, start, end, RAW_DIR)

    # 3. Save metadata
    save_download_metadata(all_tickers, start, end, RAW_DIR, data)

    # 4. Audit
    logger.info("-" * 40)
    logger.info("STEP 2: Running data quality audit...")
    audit_results = audit_all_stocks(data)

    # 5. Generate audit report
    report_path = REPORTS_DIR / "data_audit_report.md"
    generate_audit_report(audit_results, report_path)

    # 6. Clean and combine
    logger.info("-" * 40)
    logger.info("STEP 3: Cleaning and combining data...")
    combined = clean_data(data)

    # 7. Save processed data
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DIR / "all_stocks_ohlcv.csv"
    combined.to_csv(processed_path, index=False)
    logger.info("Saved processed data to %s (%d rows)", processed_path, len(combined))

    # 8. Save audit results as JSON for programmatic access
    audit_json_path = PROCESSED_DIR / "audit_results.json"
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2, default=str)
    logger.info("Saved audit JSON to %s", audit_json_path)

    # 9. Summary
    logger.info("=" * 60)
    logger.info("DATA ACQUISITION COMPLETE")
    logger.info("  Raw CSVs:     %s (%d files)", RAW_DIR, len(list(RAW_DIR.glob("*.csv"))))
    logger.info("  Processed:    %s", processed_path)
    logger.info("  Audit report: %s", report_path)
    passed = sum(1 for a in audit_results if a["status"] == "PASS")
    review = sum(1 for a in audit_results if a["status"] == "REVIEW")
    logger.info("  Audit: %d PASS, %d REVIEW", passed, review)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
