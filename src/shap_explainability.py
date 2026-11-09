"""
shap_explainability.py — SHAP Feature Importance & Stability
Project: Horizon- and Regime-Specific ML for Stock Return Prediction
Market: Indian Equity (NSE) exclusively

Assumption (logged): SHAP computed locally (not Kaggle) because dataset fits
in memory and training is fast enough (~5-10 min total). This is a deviation
from the Kaggle-only protocol for heavy compute, made explicitly here.

Analyses:
  - Re-trains XGBoost (same frozen hyperparams from models.yaml) per split × horizon
  - SHAP TreeExplainer on TEST data only (no leakage)
  - Reports global + per-regime feature importance
  - Cross-split stability: Spearman r, Kendall's W (answers RQ6)

Usage:
    python src/shap_explainability.py
"""
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
CONFIG_PATH   = PROJECT_ROOT / "configs" / "experiment.yaml"
MODELS_PATH   = PROJECT_ROOT / "configs" / "models.yaml"
DATA_PATH     = PROJECT_ROOT / "data" / "processed" / "features_targets.csv"
FEAT_LIST     = PROJECT_ROOT / "data" / "processed" / "feature_list.txt"
RESULTS_DIR   = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
TRADING_DAYS = 252


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_purge_cutoff(all_dates, test_start, horizon):
    ts_np  = np.datetime64(pd.Timestamp(test_start), "ns")
    idx    = np.searchsorted(all_dates, ts_np, side="left")
    return pd.Timestamp(all_dates[max(idx - horizon, 0)])


def clean(X):
    return np.where(np.isfinite(X), X, 0.0)


def spearman_r(a, b):
    from scipy import stats
    return stats.spearmanr(a, b).correlation


def kendalls_w(rank_matrix):
    from scipy import stats
    m, n = rank_matrix.shape
    S    = np.sum((rank_matrix.sum(axis=0) - m * (n + 1) / 2) ** 2)
    W    = 12 * S / (m ** 2 * (n ** 3 - n))
    chi2 = m * (n - 1) * W
    p    = 1 - stats.chi2.cdf(chi2, df=n - 1)
    return round(W, 4), round(chi2, 4), round(p, 4)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("STAGE 10 — SHAP Feature Importance & Stability")
    logger.info("NOTE: Running locally (not Kaggle) — see module docstring.")
    logger.info("=" * 60)

    # ── Lazy imports (heavy) ─────────────────────────────────────────────
    try:
        import shap
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        logger.error("Missing package: %s.  Run: pip install shap xgboost scikit-learn", e)
        raise

    # ── Load config ──────────────────────────────────────────────────────
    with open(CONFIG_PATH) as f:
        exp_cfg = yaml.safe_load(f)
    with open(MODELS_PATH) as f:
        mdl_cfg = yaml.safe_load(f)

    splits_cfg   = exp_cfg["walk_forward"]["splits"]
    horizons     = exp_cfg["horizons"]         # [1, 5]
    benchmark    = exp_cfg["data"]["benchmark"]  # ^NSEI
    xgb_raw      = mdl_cfg["xgboost"]
    # Filter params sklearn/XGB can actually use
    xgb_params = {k: v for k, v in xgb_raw.items()
                  if k not in ("use_label_encoder", "eval_metric")}
    xgb_params["random_state"] = RANDOM_SEED
    xgb_params["n_jobs"]       = -1

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading features_targets.csv...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    with open(FEAT_LIST) as f:
        feature_cols = [l.strip() for l in f if l.strip()]

    df_stocks = df[df["Ticker"] != benchmark].copy().reset_index(drop=True)
    all_dates = np.sort(df_stocks["Date"].values.astype("datetime64[ns]"))

    logger.info("Stocks df: %s  Features: %d", df_stocks.shape, len(feature_cols))

    REGIME_NAMES = {0: "Low", 1: "Medium", 2: "High"}

    imp_by_split  = []   # global importance per split × horizon
    regime_imp    = []   # importance per split × horizon × regime

    total   = len(splits_cfg) * len(horizons)
    combo   = 0

    for split_cfg in splits_cfg:
        label       = split_cfg["label"]
        train_start = pd.Timestamp(split_cfg["train_start"])
        test_start  = pd.Timestamp(split_cfg["test_start"])
        test_end    = pd.Timestamp(split_cfg["test_end"])
        regime_col  = f"regime_{label}"

        for horizon in horizons:
            combo += 1
            t0         = time.time()
            target_col = f"fwd_label_{horizon}d"
            logger.info("[%d/%d] %s  h=%dd", combo, total, label, horizon)

            # Purge cutoff (no embargo needed — test data is never touched)
            purge_cutoff = get_purge_cutoff(all_dates, test_start, horizon)

            train_mask = (
                (df_stocks["Date"] >= train_start) &
                (df_stocks["Date"] <= purge_cutoff) &
                df_stocks[target_col].notna()
            )
            test_mask = (
                (df_stocks["Date"] >= test_start) &
                (df_stocks["Date"] <= test_end) &
                df_stocks[target_col].notna()
            )

            df_train = df_stocks[train_mask].copy()
            df_test  = df_stocks[test_mask].copy()

            if len(df_train) < 100 or len(df_test) < 10:
                logger.warning("  SKIP — insufficient rows")
                continue

            # Scale on training only
            X_train = clean(df_train[feature_cols].values)
            X_test  = clean(df_test[feature_cols].values)
            y_train = df_train[target_col].values

            scaler      = StandardScaler()
            X_train_sc  = scaler.fit_transform(X_train)
            X_test_sc   = scaler.transform(X_test)

            # Re-train XGBoost with frozen hyperparams (same as Stage 6)
            model = XGBClassifier(**xgb_params)
            model.fit(X_train_sc, y_train, verbose=False)

            # SHAP — TreeExplainer on test data ONLY
            explainer   = shap.TreeExplainer(model)
            shap_vals   = explainer.shap_values(X_test_sc)   # (n_test, n_feat)
            mean_abs    = np.abs(shap_vals).mean(axis=0)      # (n_feat,)

            # Global importance row
            imp_row = {"split": label, "horizon": horizon}
            imp_row.update(dict(zip(feature_cols, mean_abs.tolist())))
            imp_by_split.append(imp_row)

            # Per-regime importance
            if regime_col in df_test.columns:
                for rid, rname in REGIME_NAMES.items():
                    rmask = df_test[regime_col].values == rid
                    if rmask.sum() < 5:
                        continue
                    reg_shap = np.abs(shap_vals[rmask]).mean(axis=0)
                    reg_row  = {
                        "split": label, "horizon": horizon,
                        "regime": rname, "n": int(rmask.sum()),
                    }
                    reg_row.update(dict(zip(feature_cols, reg_shap.tolist())))
                    regime_imp.append(reg_row)

            # Top-5 log
            top5_idx = mean_abs.argsort()[::-1][:5]
            top5     = [(feature_cols[i], round(float(mean_abs[i]), 5)) for i in top5_idx]
            logger.info("  Top-5: %s  (%ds)", top5, int(time.time() - t0))

    # ── Stability analysis ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STABILITY ANALYSIS")
    logger.info("=" * 60)

    imp_df    = pd.DataFrame(imp_by_split)
    regime_df = pd.DataFrame(regime_imp)

    stability_rows   = []
    feat_rank_rows   = []

    for horizon in horizons:
        sub    = imp_df[imp_df["horizon"] == horizon]
        splits = sub["split"].tolist()
        mat    = sub[feature_cols].values   # (n_splits, n_features)

        # Mean importance across splits
        mean_imp = sub[feature_cols].mean().sort_values(ascending=False)
        logger.info("Top-15 features (h=%dd, mean |SHAP|):", horizon)
        for rank, (feat, val) in enumerate(mean_imp.head(15).items(), 1):
            logger.info("  %2d. %-30s %.6f", rank, feat, val)

        # Cross-split Spearman r
        n_sp     = len(splits)
        corr_vals = []
        for i in range(n_sp):
            for j in range(i + 1, n_sp):
                r = spearman_r(mat[i], mat[j])
                corr_vals.append(r)
                stability_rows.append({
                    "horizon":   horizon,
                    "split_a":   splits[i],
                    "split_b":   splits[j],
                    "spearman_r": round(r, 4),
                })

        mean_corr = float(np.mean(corr_vals)) if corr_vals else np.nan
        logger.info("  Mean pairwise Spearman r (h=%dd): %.4f", horizon, mean_corr)

        # Kendall's W on ranks
        rank_matrix = np.array([(-row).argsort().argsort() for row in mat])
        W, chi2, p = kendalls_w(rank_matrix)
        logger.info("  Kendall's W=%.4f  chi2=%.4f  p=%.4f", W, chi2, p)

        stability_rows.append({
            "horizon":    horizon,
            "split_a":    "ALL",
            "split_b":    "ALL",
            "spearman_r": round(mean_corr, 4),
            "kendalls_W": W,
            "chi2":       chi2,
            "p_value":    p,
        })

        # Feature ranking table
        for rank, (feat, val) in enumerate(mean_imp.items(), 1):
            row_d = {
                "feature":        feat,
                "horizon":        horizon,
                "mean_abs_shap":  round(float(val), 6),
                "global_rank":    rank,
            }
            for sp_label, sp_row in zip(splits, mat):
                feat_idx = feature_cols.index(feat)
                row_d[f"imp_{sp_label}"] = round(float(sp_row[feat_idx]), 6)
            feat_rank_rows.append(row_d)

    # Regime comparison
    logger.info("Per-regime feature rank correlations (1d horizon):")
    h1_reg = regime_df[regime_df["horizon"] == 1]
    for r1_name in ["Low", "Medium", "High"]:
        for r2_name in ["Low", "Medium", "High"]:
            if r1_name >= r2_name:
                continue
            v1 = h1_reg[h1_reg["regime"] == r1_name][feature_cols].mean()
            v2 = h1_reg[h1_reg["regime"] == r2_name][feature_cols].mean()
            if len(v1) > 0 and len(v2) > 0:
                corr = spearman_r(v1.values, v2.values)
                logger.info("  %s vs %s: Spearman r = %.4f", r1_name, r2_name, corr)

    # ── Save outputs ─────────────────────────────────────────────────────
    stab_df      = pd.DataFrame(stability_rows)
    feat_rank_df = pd.DataFrame(feat_rank_rows)

    paths = {
        "shap_importance_by_split.csv":  imp_df,
        "shap_importance_by_regime.csv": regime_df,
        "shap_stability.csv":            stab_df,
        "feature_rankings.csv":          feat_rank_df,
    }
    for fname, dframe in paths.items():
        p = RESULTS_DIR / fname
        dframe.to_csv(p, index=False)
        logger.info("Saved: %s (%d rows)", p, len(dframe))

    # Summary JSON
    summary_1d = stab_df[(stab_df["horizon"] == 1) & (stab_df["split_a"] == "ALL")]
    summary_5d = stab_df[(stab_df["horizon"] == 5) & (stab_df["split_a"] == "ALL")]
    summary = {
        "top5_features_1d": feat_rank_df[feat_rank_df["horizon"] == 1]
                             .head(5)[["feature", "mean_abs_shap"]].to_dict(orient="records"),
        "top5_features_5d": feat_rank_df[feat_rank_df["horizon"] == 5]
                             .head(5)[["feature", "mean_abs_shap"]].to_dict(orient="records"),
        "stability_spearman_1d": float(summary_1d["spearman_r"].iloc[0]) if not summary_1d.empty else None,
        "stability_spearman_5d": float(summary_5d["spearman_r"].iloc[0]) if not summary_5d.empty else None,
        "kendalls_W_1d": float(summary_1d["kendalls_W"].iloc[0]) if not summary_1d.empty else None,
        "kendalls_W_5d": float(summary_5d["kendalls_W"].iloc[0]) if not summary_5d.empty else None,
    }
    with open(RESULTS_DIR / "explainability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary JSON saved.")

    logger.info("=" * 60)
    logger.info("STAGE 10 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
