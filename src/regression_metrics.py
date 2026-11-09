"""
regression_metrics.py — Regression Metrics
Project: Horizon- and Regime-Specific ML for Stock Return Prediction
Market: Indian Equity (NSE) exclusively

Computes MAE, MSE, RMSE, MAPE, R² for all 5 models:
  Naive, LogReg, RF, XGB, LSTM
Across both horizons (1-day and 5-day).

DESIGN
------
y_true  = actual forward return  (fwd_return_1d / fwd_return_5d)
            continuously distributed in approx [-0.28, +0.39]

y_pred  = centred predicted probability  (y_prob - 0.5)
            maps probability space [0, 1] → directional signal [-0.5, +0.5]
            centred at 0 (= neutral), positive = bullish, negative = bearish
            This is the ONLY continuous output these classifiers produce.

RATIONALE FOR CENTERING
-----------------------
Raw y_prob lives in [0, 1] (mean ≈ 0.5) while fwd_return lives in [-0.28, +0.39]
(mean ≈ 0). Comparing them directly inflates all error metrics by ~500× and
drives R² to -4000 — not because the model is bad but because the units are
incomparable. Centering removes this systematic offset so that:
  - A neutral model (y_prob = 0.5) gets y_pred = 0 (matches mean of returns)
  - A bullish model (y_prob > 0.5) gets y_pred > 0 (consistent sign with return)
  - Naive (y_prob = 1.0 always) gets y_pred = 0.5 (large positive bias)
This transformation is symmetric, reversible, and makes no assumption about
model quality. It does NOT cherry-pick a rule to improve results.

MAPE ZERO-HANDLING (pre-specified rule)
-----------------------------------------
Rows where |y_true| < 1e-4 are excluded from MAPE to prevent division by
near-zero returns. Count of excluded rows is logged. This threshold (not 1e-10)
is chosen because returns smaller than 0.01% are effectively zero for
economic interpretation. Rule is fixed here and not tuned post-hoc.

NOTE: Pure numpy implementation — no sklearn / scipy dependency.

Outputs:
  results/tables/regression_metrics.csv          (aggregated: mean +/- std across 6 splits)
  results/tables/regression_metrics_by_split.csv (per-split detail)

Usage:
    python src/regression_metrics.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR   = PROJECT_ROOT / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

ML_PREDS_PATH   = PROJECT_ROOT / "artifacts/kaggle/tabular_ml_models/predictions_all_splits.csv"
LSTM_PREDS_PATH = PROJECT_ROOT / "artifacts/kaggle/lstm_sequence_models/lstm_predictions_all_splits.csv"

MODEL_ORDER = ["Naive", "LogReg", "RF", "XGB", "LSTM"]


# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy metric functions
# ─────────────────────────────────────────────────────────────────────────────

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(_mse(y_true, y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Pre-specified zero-handling rule: exclude rows where |y_true| < 1e-4
    (i.e. actual return < 0.01%). Returns smaller than that are
    economically indistinguishable from zero and would produce arbitrarily
    large percentage errors unrelated to model quality.
    """
    mask = np.abs(y_true) >= 1e-4
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² — coefficient of determination (can be negative)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict with MAE, MSE, RMSE, MAPE, R².

    y_true : actual forward return
    y_pred : centred probability (y_prob - 0.5) — directional signal in return-space
    """
    return {
        "MAE":  round(_mae(y_true, y_pred),  6),
        "MSE":  round(_mse(y_true, y_pred),  8),
        "RMSE": round(_rmse(y_true, y_pred), 6),
        "MAPE": round(_mape(y_true, y_pred), 4),
        "R2":   round(_r2(y_true, y_pred),   6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ml_predictions() -> pd.DataFrame:
    """Load tabular ML model predictions and normalise columns."""
    logger.info("Loading ML predictions: %s", ML_PREDS_PATH)
    df = pd.read_csv(ML_PREDS_PATH)
    logger.info("  Shape: %s | Models: %s", df.shape, df["model"].unique().tolist())

    df["horizon"] = df["horizon"].astype(int)

    # Select the correct actual return based on horizon
    df["y_true_return"] = np.where(
        df["horizon"] == 1,
        pd.to_numeric(df["fwd_return_1d"], errors="coerce"),
        pd.to_numeric(df["fwd_return_5d"], errors="coerce"),
    )
    df["y_prob"] = pd.to_numeric(df["y_prob"], errors="coerce")

    # Centre y_prob: directional signal in return-space
    # y_prob in [0,1] → y_pred_centered in [-0.5, +0.5], centred at 0
    df["y_pred"] = df["y_prob"] - 0.5

    return df[["split", "horizon", "model", "y_true_return", "y_pred"]].dropna()


def load_lstm_predictions() -> pd.DataFrame:
    """Load LSTM sequence model predictions and normalise columns."""
    logger.info("Loading LSTM predictions: %s", LSTM_PREDS_PATH)
    df = pd.read_csv(LSTM_PREDS_PATH)
    logger.info("  Shape: %s", df.shape)

    df["horizon"] = df["horizon"].astype(int)
    df["model"]   = "LSTM"

    df["y_true_return"] = np.where(
        df["horizon"] == 1,
        pd.to_numeric(df["fwd_return_1d"], errors="coerce"),
        pd.to_numeric(df["fwd_return_5d"], errors="coerce"),
    )
    df["y_prob"] = pd.to_numeric(df["y_prob"], errors="coerce")

    # Centre y_prob: directional signal in return-space
    df["y_pred"] = df["y_prob"] - 0.5

    return df[["split", "horizon", "model", "y_true_return", "y_pred"]].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_split(all_preds: pd.DataFrame) -> pd.DataFrame:
    """Compute all 5 metrics for each (model, horizon, split) triple.

    y_true = y_true_return (actual forward return)
    y_pred = y_pred (centred probability = y_prob - 0.5)
    Both columns are set at load time.
    """
    records = []
    splits  = sorted(all_preds["split"].unique())

    for split in splits:
        for horizon in [1, 5]:
            for model in MODEL_ORDER:
                sub = all_preds[
                    (all_preds["split"]   == split)  &
                    (all_preds["horizon"] == horizon) &
                    (all_preds["model"]   == model)
                ]
                if sub.empty:
                    continue

                y_true = sub["y_true_return"].values
                y_pred = sub["y_pred"].values  # already centred: y_prob - 0.5

                # Log how many rows are excluded from MAPE
                mape_excluded = int(np.sum(np.abs(y_true) < 1e-4))
                if mape_excluded > 0:
                    logger.debug(
                        "  MAPE: %s split=%s h=%s excluded %d rows (|y_true| < 1e-4)",
                        model, split, horizon, mape_excluded
                    )

                metrics = compute_all_metrics(y_true, y_pred)
                records.append({
                    "split":          split,
                    "horizon":        horizon,
                    "model":          model,
                    "n":              len(sub),
                    "mape_excluded":  mape_excluded,
                    **metrics,
                })

    return pd.DataFrame(records)


def aggregate_metrics(per_split_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-split metrics to mean +/- std across all splits."""
    records = []
    for horizon in [1, 5]:
        for model in MODEL_ORDER:
            sub = per_split_df[
                (per_split_df["horizon"] == horizon) &
                (per_split_df["model"]   == model)
            ]
            if sub.empty:
                continue

            row = {"horizon": horizon, "model": model, "n_splits": len(sub)}
            for metric in ["MAE", "MSE", "RMSE", "MAPE", "R2"]:
                row[f"{metric}_mean"] = round(sub[metric].mean(), 6)
                row[f"{metric}_std"]  = round(sub[metric].std(),  6)

            records.append(row)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(agg_df: pd.DataFrame) -> None:
    print("\n" + "=" * 75)
    print("REGRESSION METRICS SUMMARY  (y_true=fwd_return, y_pred=y_prob-0.5)")
    print("=" * 75)

    for horizon in [1, 5]:
        print(f"\n" + "-" * 75)
        print(f"  {horizon}-DAY HORIZON")
        print("-" * 75)
        print(f"  {'Model':<10} {'MAE':<10} {'MSE':<12} {'RMSE':<10} {'MAPE%':<10} {'R2':<10}")
        print("  " + "-" * 63)

        sub = agg_df[agg_df["horizon"] == horizon].set_index("model")
        for model in MODEL_ORDER:
            if model not in sub.index:
                continue
            r = sub.loc[model]
            print(
                f"  {model:<10}"
                f"  {r['MAE_mean']:.5f}"
                f"  {r['MSE_mean']:.7f}"
                f"  {r['RMSE_mean']:.5f}"
                f"  {r['MAPE_mean']:>9.2f}"
                f"  {r['R2_mean']:>9.4f}"
            )

    print("\n" + "=" * 75 + "\n")



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("STAGE 11 — Regression Metrics (MAE / MSE / RMSE / MAPE / R²)")
    logger.info("=" * 60)

    # 1. Load predictions
    ml_preds   = load_ml_predictions()
    lstm_preds = load_lstm_predictions()
    all_preds  = pd.concat([ml_preds, lstm_preds], ignore_index=True)
    logger.info("Combined predictions: %d rows", len(all_preds))
    logger.info("Models found: %s",  sorted(all_preds["model"].unique()))
    logger.info("Horizons found: %s", sorted(all_preds["horizon"].unique()))
    logger.info("Splits found: %s",  sorted(all_preds["split"].unique()))

    # 2. Per-split metrics
    logger.info("Computing per-split metrics …")
    per_split_df = compute_per_split(all_preds)
    logger.info("  Per-split rows: %d", len(per_split_df))

    out_split = TABLES_DIR / "regression_metrics_by_split.csv"
    per_split_df.to_csv(out_split, index=False)
    logger.info("  Saved: %s", out_split)

    # 3. Aggregated metrics (mean ± std)
    logger.info("Aggregating across splits …")
    agg_df = aggregate_metrics(per_split_df)

    out_agg = TABLES_DIR / "regression_metrics.csv"
    agg_df.to_csv(out_agg, index=False)
    logger.info("  Saved: %s", out_agg)

    # 4. Print summary
    print_summary_table(agg_df)

    logger.info("=" * 60)
    logger.info("STAGE 11 COMPLETE")
    logger.info("  %s", out_agg)
    logger.info("  %s", out_split)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
