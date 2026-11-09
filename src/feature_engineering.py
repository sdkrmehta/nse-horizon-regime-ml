"""
feature_engineering.py — Leakage-Safe Feature Engineering
Project: Horizon- and Regime-Specific ML for Stock Return Prediction
Market: Indian Equity (NSE) exclusively

Constructs OHLCV-derived technical features using ONLY data available
at prediction time. Every feature at time t uses information from t
and earlier — never from t+1 onward.

Feature families:
  1. Returns     — Lagged returns (1, 2, 5, 10, 20 days)
  2. Trend       — SMA, EMA, price/SMA ratio
  3. Momentum    — RSI, ROC, Williams %R
  4. Volatility  — Rolling std, ATR, realized volatility
  5. Volume      — Volume change, rolling volume stats, relative volume
  6. Calendar    — Day of week, month

LEAKAGE RULE: All rolling windows end at time t. No centered windows.
No future observations. This is a HARD constraint (AGENTS.md §6).

Usage:
    python src/feature_engineering.py
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
# Feature Functions — All operate per-stock, all backward-looking
# ---------------------------------------------------------------------------

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lagged returns: how much the price changed over the past N days.
    These are BACKWARD-looking — perfectly safe.
    """
    close = df["Adj Close"]
    for lag in [1, 2, 5, 10, 20]:
        df[f"return_{lag}d"] = close.pct_change(lag)
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Moving averages and price-relative-to-MA ratios.
    All computed as trailing (backward) windows ending at t.
    """
    close = df["Adj Close"]

    for window in [5, 10, 20, 50]:
        # Simple Moving Average (trailing)
        sma = close.rolling(window=window, min_periods=window).mean()
        df[f"sma_{window}"] = sma

        # Price relative to SMA (how far price is from its average)
        df[f"price_sma_ratio_{window}"] = close / sma

        # Exponential Moving Average (trailing, uses all past data with decay)
        ema = close.ewm(span=window, adjust=False).mean()
        df[f"ema_{window}"] = ema

    # SMA crossover signals (short-term vs long-term)
    df["sma_5_20_ratio"] = df["sma_5"] / df["sma_20"]
    df["sma_10_50_ratio"] = df["sma_10"] / df["sma_50"]

    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum indicators — all backward-looking.
    """
    close = df["Adj Close"]
    high = df["High"]
    low = df["Low"]

    # --- RSI (Relative Strength Index) — 14-day ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --- Rate of Change (ROC) ---
    for period in [5, 10, 20]:
        df[f"roc_{period}"] = close.pct_change(period) * 100

    # --- Williams %R (14-day) ---
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Uses trailing window — safe
    hh_14 = high.rolling(window=14, min_periods=14).max()
    ll_14 = low.rolling(window=14, min_periods=14).min()
    df["williams_r_14"] = ((hh_14 - close) / (hh_14 - ll_14).replace(0, np.nan)) * -100

    # --- MACD (12, 26, 9) — all exponential, all trailing ---
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = macd_line - signal_line

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility measures — all trailing windows.
    """
    close = df["Adj Close"]
    high = df["High"]
    low = df["Low"]
    returns = close.pct_change()

    # --- Rolling standard deviation of returns ---
    for window in [5, 10, 20]:
        df[f"volatility_{window}d"] = returns.rolling(
            window=window, min_periods=window
        ).std()

    # --- Realized volatility (annualized) ---
    df["realized_vol_20d"] = returns.rolling(
        window=20, min_periods=20
    ).std() * np.sqrt(252)

    # --- Average True Range (ATR) — 14-day ---
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = true_range.ewm(span=14, adjust=False).mean()

    # --- Bollinger Band width (20-day, 2 std) ---
    sma_20 = close.rolling(window=20, min_periods=20).mean()
    std_20 = close.rolling(window=20, min_periods=20).std()
    df["bb_width_20"] = (2 * std_20) / sma_20.replace(0, np.nan)

    # --- High-Low range as fraction of close ---
    df["hl_range_pct"] = (high - low) / close

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume-derived features — all trailing.
    """
    volume = df["Volume"]

    # Volume change (day-over-day)
    df["volume_change_1d"] = volume.pct_change()

    # Rolling volume statistics
    for window in [5, 10, 20]:
        vol_ma = volume.rolling(window=window, min_periods=window).mean()
        df[f"volume_sma_{window}"] = vol_ma

        # Relative volume (current volume / rolling average)
        df[f"relative_volume_{window}"] = volume / vol_ma.replace(0, np.nan)

    # Volume trend (is volume increasing or decreasing?)
    df["volume_trend_5d"] = volume.rolling(5, min_periods=5).mean().pct_change()

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar features — known at decision time (safe).
    """
    dt = pd.to_datetime(df["Date"])

    # Day of week (0=Monday, 4=Friday)
    df["day_of_week"] = dt.dt.dayofweek

    # Month (1-12)
    df["month"] = dt.dt.month

    return df


# ---------------------------------------------------------------------------
# Master Feature Builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features for each stock independently.

    Features are computed per-stock to prevent cross-stock leakage.
    All features use only backward-looking (trailing) calculations.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with targets (from target_engineering.py output).

    Returns
    -------
    pd.DataFrame
        Input data with all feature columns added.
    """
    logger.info("Building features for %d tickers...", df["Ticker"].nunique())

    result_frames = []
    feature_cols = None

    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date").copy()

        group = add_return_features(group)
        group = add_trend_features(group)
        group = add_momentum_features(group)
        group = add_volatility_features(group)
        group = add_volume_features(group)
        group = add_calendar_features(group)

        # Clip any inf/-inf produced by zero-division (e.g. volume ratios when
        # rolling avg volume = 0, or RSI when avg_loss = 0). Replace with NaN so
        # the warm-up / imputation logic handles them consistently.
        num_cols = group.select_dtypes(include=[np.number]).columns
        group[num_cols] = group[num_cols].replace([np.inf, -np.inf], np.nan)

        if feature_cols is None:
            # Identify which columns are features (newly added)
            original_cols = {"Date", "Ticker", "Open", "High", "Low", "Close",
                             "Adj Close", "Volume", "fwd_return_1d", "fwd_return_5d",
                             "fwd_label_1d", "fwd_label_5d"}
            feature_cols = [c for c in group.columns if c not in original_cols]
            logger.info("  Generated %d feature columns", len(feature_cols))

        result_frames.append(group)

    result = pd.concat(result_frames, ignore_index=True)
    result = result.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return result, feature_cols


# ---------------------------------------------------------------------------
# Leakage Audit
# ---------------------------------------------------------------------------

def audit_features_for_leakage(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int] = [1, 5],
) -> dict:
    """
    Audit features for potential data leakage.

    Checks:
    1. No feature at time t is correlated > 0.95 with the target
       (would indicate the feature contains future info)
    2. Features at time t don't change when future data is removed
       (verified by construction — all rolling windows are trailing)
    3. Feature NaN pattern: NaN should be at the START (warm-up period),
       not scattered or at the end

    Returns
    -------
    dict with audit results.
    """
    logger.info("Running leakage audit on %d features...", len(feature_cols))
    results = {"passed": True, "suspicious": [], "checks": []}

    # Check 1: Suspiciously high correlation with targets
    for h in horizons:
        target_col = f"fwd_return_{h}d"
        if target_col not in df.columns:
            continue

        valid_mask = df[target_col].notna()
        for feat in feature_cols:
            if feat not in df.columns or df[feat].notna().sum() < 100:
                continue
            mask = valid_mask & df[feat].notna()
            if mask.sum() < 100:
                continue
            corr = df.loc[mask, feat].corr(df.loc[mask, target_col])
            if abs(corr) > 0.90:
                msg = f"HIGH_CORR: {feat} vs {target_col} = {corr:.4f}"
                results["suspicious"].append(msg)
                results["passed"] = False
                logger.warning("  ⚠ %s", msg)

    # Check 2: Feature NaN pattern (should be at start, not end)
    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date")
        for feat in feature_cols[:5]:  # Spot-check first 5 features
            vals = group[feat]
            if vals.isna().any():
                # Find last NaN position
                last_nan_pos = vals.isna()[::-1].idxmax()
                first_valid_pos = vals.first_valid_index()
                # NaN should all be before first valid value
                if first_valid_pos is not None:
                    nans_after_valid = vals.loc[first_valid_pos:].isna().sum()
                    if nans_after_valid > 0:
                        msg = f"NAN_PATTERN: {ticker}/{feat} has {nans_after_valid} NaN after first valid"
                        results["checks"].append(msg)
                        # This is informational, not necessarily leakage
        break  # Only spot-check first ticker

    if results["passed"]:
        msg = "PASSED: No features with suspiciously high target correlation (>0.90)"
        results["checks"].append(msg)
        logger.info("  ✓ %s", msg)

    return results


def get_feature_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Generate summary statistics for all features."""
    stats = []
    for col in feature_cols:
        s = df[col]
        stats.append({
            "feature": col,
            "count": int(s.notna().sum()),
            "nan_count": int(s.isna().sum()),
            "nan_pct": round(s.isna().mean() * 100, 2),
            "mean": round(s.mean(), 6) if s.notna().any() else None,
            "std": round(s.std(), 6) if s.notna().any() else None,
            "min": round(s.min(), 6) if s.notna().any() else None,
            "max": round(s.max(), 6) if s.notna().any() else None,
        })
    return pd.DataFrame(stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Build and audit all features."""
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING — Leakage-Safe Technical Features")
    logger.info("=" * 60)

    # Load config
    config = load_config()
    horizons = config["horizons"]

    # Load targets data
    targets_path = PROCESSED_DIR / "targets.csv"
    logger.info("Loading targets from %s", targets_path)
    df = pd.read_csv(targets_path, parse_dates=["Date"])
    logger.info("Loaded %d rows, %d tickers", len(df), df["Ticker"].nunique())

    # Build features
    df_feat, feature_cols = build_features(df)

    # Leakage audit
    logger.info("-" * 40)
    audit = audit_features_for_leakage(df_feat, feature_cols, horizons=horizons)

    # Feature summary
    summary = get_feature_summary(df_feat, feature_cols)

    # Warm-up period: drop rows where ANY feature is NaN due to rolling windows
    # The longest rolling window is 50 (SMA-50), so first ~50 rows per stock will have NaN
    warmup_nans_before = df_feat[feature_cols].isna().any(axis=1).sum()
    logger.info("Rows with at least one NaN feature (warm-up): %d", warmup_nans_before)

    # Save full dataset (with NaN — let the validation framework handle dropping)
    output_path = PROCESSED_DIR / "features_targets.csv"
    df_feat.to_csv(output_path, index=False)
    logger.info("Saved features+targets to %s (%d rows × %d cols)",
                output_path, len(df_feat), df_feat.shape[1])

    # Save feature list
    feat_list_path = PROCESSED_DIR / "feature_list.txt"
    with open(feat_list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(feature_cols))
    logger.info("Saved feature list (%d features) to %s", len(feature_cols), feat_list_path)

    # Save feature summary
    summary_path = PROCESSED_DIR / "feature_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved feature summary to %s", summary_path)

    # Final report
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("  Total features: %d", len(feature_cols))
    logger.info("  Feature families:")

    families = {
        "Returns": [c for c in feature_cols if c.startswith("return_")],
        "Trend": [c for c in feature_cols if any(c.startswith(p) for p in ["sma_", "ema_", "price_sma_"])],
        "Momentum": [c for c in feature_cols if any(c.startswith(p) for p in ["rsi_", "roc_", "williams_", "macd"])],
        "Volatility": [c for c in feature_cols if any(c.startswith(p) for p in ["volatility_", "realized_", "atr_", "bb_", "hl_"])],
        "Volume": [c for c in feature_cols if any(c.startswith(p) for p in ["volume_", "relative_"])],
        "Calendar": [c for c in feature_cols if c in ["day_of_week", "month"]],
    }
    for family, cols in families.items():
        logger.info("    %s: %d features", family, len(cols))

    logger.info("  Leakage audit: %s", "PASSED ✓" if audit["passed"] else "ISSUES ⚠")
    logger.info("  Warm-up NaN rows: %d (will be handled in validation)", warmup_nans_before)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
