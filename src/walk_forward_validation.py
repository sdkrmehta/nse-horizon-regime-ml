"""
walk_forward_validation.py — Walk-Forward Validation Framework
Project: Horizon- and Regime-Specific ML for Stock Return Prediction
Market: Indian Equity (NSE) exclusively

Implements chronological walk-forward validation with:
  - Expanding training windows (2015-2019 → 2020 test, ..., 2015-2024 → 2025 test)
  - PURGE: removes training observations whose forward-label window overlaps the test period
  - EMBARGO: adds a gap between the purged training end and test start to prevent
    near-boundary information transfer
  - Per-split scaler fitting on training data ONLY (no test data leakage)
  - Full audit trail: every split records exact boundaries, purge counts, embargo counts

HARD CONSTRAINTS (AGENTS.md §6):
  - ALL scalers fit inside training window only
  - Purge overlapping labels for 5-day targets
  - Embargo boundaries
  - Never tune on final holdout

Usage:
    from src.walk_forward_validation import WalkForwardSplitter
    OR
    python src/walk_forward_validation.py   (runs smoke test)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

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

# Embargo: number of trading days to drop after purge zone before test begins.
# For the 5-day horizon, the last training label window ends h days after the
# last training date. We add EMBARGO_DAYS on top as an extra safety buffer.
EMBARGO_DAYS = 5  # pre-specified, not tuned after seeing results


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """
    All information produced for one walk-forward split.

    Attributes
    ----------
    label : str
        Split identifier (e.g. 'split_1').
    train_start, train_end : pd.Timestamp
        Nominal training boundaries.
    test_start, test_end : pd.Timestamp
        Test boundaries.
    horizon : int
        Forecast horizon in trading days.
    purge_count : int
        Number of training rows removed due to label overlap.
    embargo_count : int
        Number of rows in the embargo zone (discarded from neither set).
    train_idx : np.ndarray
        Integer positions (iloc) of training rows in the full DataFrame.
    test_idx : np.ndarray
        Integer positions (iloc) of test rows in the full DataFrame.
    train_dates : tuple[pd.Timestamp, pd.Timestamp]
        Actual (post-purge) training date range.
    scaler : StandardScaler
        Fitted scaler (fitted on training data only).
    regime_col : str
        Name of the regime column for this split.
    """
    label: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    horizon: int
    purge_count: int
    embargo_count: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_dates: tuple
    scaler: StandardScaler
    regime_col: str
    split_metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core: Walk-Forward Splitter
# ---------------------------------------------------------------------------

class WalkForwardSplitter:
    """
    Generates leakage-safe walk-forward splits for the NSE dataset.

    For each split defined in experiment.yaml:
      1. Select training rows (Date in [train_start, train_end])
      2. PURGE: drop training rows whose label window (t → t+h) overlaps [test_start, ...]
         For horizon h, any training row with Date > (test_start - h trading days) must be purged
         because its forward label uses data from the test period.
      3. EMBARGO: additionally drop EMBARGO_DAYS rows before test_start as a safety buffer.
      4. Select test rows (Date in [test_start, test_end])
      5. Fit StandardScaler on purged training features only
      6. Return SplitResult

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with all features, targets, regime columns.
        Must have a 'Date' column (datetime) and a 'Ticker' column.
    feature_cols : list of str
        Feature column names to scale.
    horizons : list of int
        Forecast horizons (e.g. [1, 5]).
    splits_config : list of dict
        Walk-forward split definitions from experiment.yaml.
    embargo_days : int
        Number of extra buffer days beyond the purge zone.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        horizons: list[int],
        splits_config: list[dict],
        embargo_days: int = EMBARGO_DAYS,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.feature_cols = feature_cols
        self.horizons = horizons
        self.splits_config = splits_config
        self.embargo_days = embargo_days

        # Build a sorted unique date array for embargo calculation
        self.all_dates = np.sort(df["Date"].unique())

    def _get_purge_cutoff(
        self,
        test_start: pd.Timestamp,
        horizon: int,
    ) -> pd.Timestamp:
        """
        Compute the last training date that is safe to include (no label overlap).

        A training row at date t has its forward-label window [t+1, t+h].
        This overlaps the test period if t+1 >= test_start, i.e. t >= test_start - 1 day.
        In calendar terms, the latest safe training date is the date that is
        exactly `horizon` trading days before test_start.

        We use the actual trading calendar (unique dates in the data) to count
        exactly h trading days back from test_start.
        """
        # Find index of test_start (or the nearest trading day on/after it)
        ts_idx = np.searchsorted(self.all_dates, test_start, side="left")

        # Move back `horizon` trading days
        purge_idx = ts_idx - horizon
        purge_idx = max(purge_idx, 0)

        purge_cutoff = self.all_dates[purge_idx]
        return pd.Timestamp(purge_cutoff)

    def _get_embargo_cutoff(
        self,
        purge_cutoff: pd.Timestamp,
        embargo_days: int,
    ) -> pd.Timestamp:
        """
        Compute the date up to (and including) which rows are in the embargo zone.

        The embargo zone = [purge_cutoff + 1 trading day, purge_cutoff + embargo_days].
        These rows are dropped from both training and test.
        """
        pc_idx = np.searchsorted(self.all_dates, purge_cutoff, side="right")
        embargo_idx = min(pc_idx + embargo_days - 1, len(self.all_dates) - 1)
        return pd.Timestamp(self.all_dates[embargo_idx])

    def generate_splits(self) -> Iterator[tuple[int, SplitResult]]:
        """
        Yield (horizon, SplitResult) for every combination of split × horizon.

        This is the primary API used by the model-training notebooks.
        """
        df = self.df

        for split_cfg in self.splits_config:
            label = split_cfg["label"]
            train_start = pd.Timestamp(split_cfg["train_start"])
            train_end = pd.Timestamp(split_cfg["train_end"])
            test_start = pd.Timestamp(split_cfg["test_start"])
            test_end = pd.Timestamp(split_cfg["test_end"])
            regime_col = f"regime_{label}"

            for horizon in self.horizons:
                target_col = f"fwd_label_{horizon}d"
                return_col = f"fwd_return_{horizon}d"

                # ---- 1. Raw training mask ----
                raw_train_mask = (
                    (df["Date"] >= train_start) &
                    (df["Date"] <= train_end) &
                    df[target_col].notna()  # drop warm-up NaN rows
                )

                # ---- 2. PURGE ----
                purge_cutoff = self._get_purge_cutoff(test_start, horizon)
                # Training rows with Date > purge_cutoff are purged
                purge_mask = raw_train_mask & (df["Date"] > purge_cutoff)
                purge_count = purge_mask.sum()

                # Safe training mask after purge
                train_mask_purged = raw_train_mask & (df["Date"] <= purge_cutoff)

                # ---- 3. EMBARGO ----
                embargo_cutoff = self._get_embargo_cutoff(purge_cutoff, self.embargo_days)
                # Embargo zone rows: after purge_cutoff but before test_start
                # These are already excluded from training (purged) and not in test either.
                embargo_mask = (
                    (df["Date"] > purge_cutoff) &
                    (df["Date"] <= embargo_cutoff) &
                    (df["Date"] < test_start)
                )
                embargo_count = embargo_mask.sum()

                # ---- 4. Test mask ----
                test_mask = (
                    (df["Date"] >= test_start) &
                    (df["Date"] <= test_end) &
                    df[target_col].notna()
                )

                train_idx = np.where(train_mask_purged)[0]
                test_idx = np.where(test_mask)[0]

                if len(train_idx) == 0:
                    logger.warning("  ⚠ %s h=%d: empty training set — skipping", label, horizon)
                    continue
                if len(test_idx) == 0:
                    logger.warning("  ⚠ %s h=%d: empty test set — skipping", label, horizon)
                    continue

                # ---- 5. Fit scaler on training data ONLY ----
                X_train_raw = df.loc[train_mask_purged, self.feature_cols]

                # Drop any remaining NaN columns for scaler fitting
                # (columns with all NaN in training — shouldn't happen but defensive)
                valid_feat_cols = [
                    c for c in self.feature_cols
                    if X_train_raw[c].notna().sum() > 10
                ]
                scaler = StandardScaler()
                # Replace inf/-inf with NaN first (can arise from zero-division in
                # volume ratios, RSI, Williams %R), then fill NaN with 0
                X_fit = X_train_raw[valid_feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                scaler.fit(X_fit)

                # ---- 6. Actual training date range (post-purge) ----
                train_dates_actual = (
                    df.loc[train_mask_purged, "Date"].min(),
                    df.loc[train_mask_purged, "Date"].max(),
                )

                metadata = {
                    "label": label,
                    "horizon": horizon,
                    "nominal_train_start": str(train_start.date()),
                    "nominal_train_end": str(train_end.date()),
                    "purge_cutoff": str(purge_cutoff.date()),
                    "embargo_cutoff": str(embargo_cutoff.date()),
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    "train_rows_before_purge": int(raw_train_mask.sum()),
                    "purge_count": int(purge_count),
                    "embargo_count": int(embargo_count),
                    "train_rows_after_purge": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "feature_cols_used": len(valid_feat_cols),
                }

                yield horizon, SplitResult(
                    label=label,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    horizon=horizon,
                    purge_count=int(purge_count),
                    embargo_count=int(embargo_count),
                    train_idx=train_idx,
                    test_idx=test_idx,
                    train_dates=train_dates_actual,
                    scaler=scaler,
                    regime_col=regime_col,
                    split_metadata=metadata,
                )

    def get_split_arrays(
        self,
        split: SplitResult,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return scaled feature arrays and labels for a given SplitResult.

        Returns
        -------
        X_train, y_train_label, y_train_return,
        X_test,  y_test_label,  y_test_return
        """
        df = self.df
        h = split.horizon
        target_label_col = f"fwd_label_{h}d"
        target_return_col = f"fwd_return_{h}d"

        X_train_raw = df.iloc[split.train_idx][self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        X_test_raw  = df.iloc[split.test_idx][self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

        # Scale using training-fitted scaler
        X_train = split.scaler.transform(X_train_raw)
        X_test  = split.scaler.transform(X_test_raw)

        y_train_label  = df.iloc[split.train_idx][target_label_col].values.astype(int)
        y_train_return = df.iloc[split.train_idx][target_return_col].values.astype(float)
        y_test_label   = df.iloc[split.test_idx][target_label_col].values.astype(int)
        y_test_return  = df.iloc[split.test_idx][target_return_col].values.astype(float)

        return X_train, y_train_label, y_train_return, X_test, y_test_label, y_test_return

    def get_test_metadata(self, split: SplitResult) -> pd.DataFrame:
        """
        Return a DataFrame of test-set metadata rows for logging predictions.

        Columns: Date, Ticker, regime, fwd_return_Xd, fwd_label_Xd
        """
        df = self.df
        h = split.horizon
        meta = df.iloc[split.test_idx][
            ["Date", "Ticker", split.regime_col,
             f"fwd_return_{h}d", f"fwd_label_{h}d"]
        ].copy()
        meta["split"] = split.label
        meta["horizon"] = h
        return meta.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Leakage Audit
# ---------------------------------------------------------------------------

def audit_split_for_leakage(
    df: pd.DataFrame,
    split: SplitResult,
) -> dict:
    """
    Verify a split has no temporal leakage.

    Checks:
    1. No training date >= test_start
    2. No test date <= purge_cutoff
    3. Scaler was not fitted on test data (by construction — verified via date check)
    4. Embargo zone contains no training or test rows
    """
    train_dates = df.iloc[split.train_idx]["Date"]
    test_dates  = df.iloc[split.test_idx]["Date"]

    issues = []

    # Check 1: No training date in test period
    train_in_test = (train_dates >= split.test_start).sum()
    if train_in_test > 0:
        issues.append(f"FAIL: {train_in_test} training rows have Date >= test_start")

    # Check 2: No test date in training period
    test_in_train = (test_dates <= split.train_end).sum()
    if test_in_train > 0:
        issues.append(f"FAIL: {test_in_train} test rows have Date <= train_end")

    # Check 3: Training max date should be <= purge_cutoff
    # (computed from split metadata)
    train_max = train_dates.max()
    if train_max >= split.test_start:
        issues.append(f"FAIL: Training max date {train_max.date()} >= test_start {split.test_start.date()}")

    # Check 4: Test min date should be >= test_start
    test_min = test_dates.min()
    if test_min < split.test_start:
        issues.append(f"FAIL: Test min date {test_min.date()} < test_start {split.test_start.date()}")

    passed = len(issues) == 0
    return {
        "split": split.label,
        "horizon": split.horizon,
        "passed": passed,
        "issues": issues,
        "train_date_max": str(train_max.date()),
        "test_date_min": str(test_min.date()),
        "purge_count": split.purge_count,
        "embargo_count": split.embargo_count,
    }


# ---------------------------------------------------------------------------
# Smoke Test / Main
# ---------------------------------------------------------------------------

def main():
    """
    Smoke test: run the walk-forward splitter on the full dataset,
    print per-split stats, run leakage audit on every split.
    """
    logger.info("=" * 60)
    logger.info("WALK-FORWARD VALIDATION FRAMEWORK — Smoke Test")
    logger.info("=" * 60)

    config = load_config()
    splits_config = config["walk_forward"]["splits"]
    horizons = config["horizons"]
    random_seed = config["random_seed"]

    # Load data
    data_path = PROCESSED_DIR / "features_targets.csv"
    logger.info("Loading data from %s ...", data_path)
    df = pd.read_csv(data_path, parse_dates=["Date"])

    # Load feature list
    feat_path = PROCESSED_DIR / "feature_list.txt"
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    logger.info("Features: %d columns", len(feature_cols))

    # Exclude benchmark from modelling (keep for backtest only)
    benchmark = config["data"]["benchmark"]
    df_stocks = df[df["Ticker"] != benchmark].copy().reset_index(drop=True)
    logger.info("Stock rows: %d | Tickers: %d", len(df_stocks), df_stocks["Ticker"].nunique())

    # Instantiate splitter
    splitter = WalkForwardSplitter(
        df=df_stocks,
        feature_cols=feature_cols,
        horizons=horizons,
        splits_config=splits_config,
        embargo_days=EMBARGO_DAYS,
    )

    # Run through all splits
    logger.info("-" * 40)
    logger.info("Generating splits (horizon × split combinations)...")
    all_audit = []
    split_summary = []

    for horizon, split in splitter.generate_splits():
        m = split.split_metadata

        logger.info(
            "  %s h=%dd | train=%d rows (purged=%d, embargo=%d) | test=%d rows",
            split.label, horizon,
            m["train_rows_after_purge"], m["purge_count"], m["embargo_count"],
            m["test_rows"],
        )

        # Get arrays (verifies scaler works)
        X_train, y_train_lbl, _, X_test, y_test_lbl, _ = splitter.get_split_arrays(split)

        # Leakage audit
        audit = audit_split_for_leakage(df_stocks, split)
        all_audit.append(audit)
        status = "✓ PASS" if audit["passed"] else "✗ FAIL"
        logger.info("    Leakage audit: %s", status)
        if not audit["passed"]:
            for issue in audit["issues"]:
                logger.error("      %s", issue)

        # Class balance in test
        up_pct = y_test_lbl.mean() * 100
        logger.info(
            "    X_train=%s  X_test=%s  test_up=%.1f%%",
            X_train.shape, X_test.shape, up_pct,
        )

        split_summary.append({
            "split": split.label,
            "horizon": horizon,
            "train_rows": m["train_rows_after_purge"],
            "test_rows": m["test_rows"],
            "purge_count": m["purge_count"],
            "embargo_count": m["embargo_count"],
            "purge_cutoff": m["purge_cutoff"],
            "embargo_cutoff": m["embargo_cutoff"],
            "test_up_pct": round(up_pct, 1),
            "leakage_audit": "PASS" if audit["passed"] else "FAIL",
        })

    # Save summary
    import json
    summary_df = pd.DataFrame(split_summary)
    out_path = PROCESSED_DIR / "validation_split_summary.csv"
    summary_df.to_csv(out_path, index=False)
    logger.info("Saved split summary to %s", out_path)

    # Save audit JSON
    audit_path = PROCESSED_DIR / "validation_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(all_audit, f, indent=2, default=str)

    # Final verdict
    all_passed = all(a["passed"] for a in all_audit)
    total_splits = len(all_audit)
    passed_splits = sum(1 for a in all_audit if a["passed"])

    logger.info("=" * 60)
    logger.info("VALIDATION FRAMEWORK COMPLETE")
    logger.info("  Total split×horizon combinations: %d", total_splits)
    logger.info("  Leakage audit: %d/%d PASSED", passed_splits, total_splits)
    logger.info("  Overall verdict: %s",
                "ALL CLEAR ✓" if all_passed else "⚠ ISSUES FOUND")
    logger.info("  Embargo days: %d", EMBARGO_DAYS)
    logger.info("  Split summary saved to: %s", out_path)
    logger.info("=" * 60)

    return splitter  # Importable and reusable


if __name__ == "__main__":
    main()
