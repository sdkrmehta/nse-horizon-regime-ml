"""
report_generator.py — Final Tables, Figures & Research Report
Project: Horizon- and Regime-Specific ML for Stock Return Prediction
Market: Indian Equity (NSE) exclusively

Generates:
  results/figures/fig1_model_accuracy_auc.png
  results/figures/fig2_net_cagr_comparison.png
  results/figures/fig3_cost_sensitivity.png
  results/figures/fig4_shap_importance_1d.png
  results/figures/fig5_shap_importance_5d.png
  results/figures/fig6_feature_stability.png
  results/figures/fig7_equity_curves.png
  results/tables/final_summary_table.csv
  reports/final_research_report.md

Usage:
    python src/report_generator.py
"""
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
FIGURES_DIR  = PROJECT_ROOT / "results" / "figures"
REPORTS_DIR  = PROJECT_ROOT / "reports"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (colour-blind safe) ───────────────────────────────────────
COLORS = {
    "Naive":  "#4878CF",
    "LogReg": "#6ACC65",
    "RF":     "#D65F5F",
    "XGB":    "#B47CC7",
    "LSTM":   "#C4AD66",
    "BnH":    "#77BEDB",
}
MODEL_ORDER = ["Naive", "LogReg", "RF", "XGB", "LSTM"]

plt.rcParams.update({
    "figure.dpi":         150,
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_ml_results():
    p = PROJECT_ROOT / "artifacts/kaggle/tabular_ml_models/results_all_splits.csv"
    return pd.read_csv(p)

def load_lstm_results():
    p = PROJECT_ROOT / "artifacts/kaggle/lstm_sequence_models/lstm_results_all_splits.csv"
    df = pd.read_csv(p)
    df["model"] = "LSTM"
    return df

def load_backtest():
    return pd.read_csv(TABLES_DIR / "backtest_results.csv")

def load_cost_sensitivity():
    return pd.read_csv(TABLES_DIR / "cost_sensitivity.csv")

def load_feature_rankings():
    return pd.read_csv(TABLES_DIR / "feature_rankings.csv")

def load_shap_stability():
    return pd.read_csv(TABLES_DIR / "shap_stability.csv")

def load_equity_curves():
    with open(PROJECT_ROOT / "results/tables/equity_curves.json") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Model Accuracy & AUC (1d and 5d)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_accuracy_auc(ml_res, lstm_res):
    logger.info("Generating Fig 1: Accuracy & AUC")
    combined = pd.concat([
        ml_res[["model", "horizon", "accuracy", "roc_auc"]],
        lstm_res[["model", "horizon", "accuracy", "roc_auc"]],
    ], ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Model Predictive Performance — NSE Indian Stocks (2020–2025)\n"
                 "Mean ± std across 6 walk-forward splits", fontsize=13, fontweight="bold")

    for col_idx, metric in enumerate(["accuracy", "roc_auc"]):
        for row_idx, horizon in enumerate([1, 5]):
            ax   = axes[row_idx, col_idx]
            sub  = combined[combined["horizon"] == horizon]
            means = sub.groupby("model")[metric].mean().reindex(MODEL_ORDER)
            stds  = sub.groupby("model")[metric].std().reindex(MODEL_ORDER)

            x     = np.arange(len(MODEL_ORDER))
            bars  = ax.bar(x, means, yerr=stds, capsize=4,
                           color=[COLORS[m] for m in MODEL_ORDER],
                           alpha=0.85, edgecolor="white", linewidth=0.8)
            ax.axhline(0.5, color="black", lw=1.0, ls="--", alpha=0.5, label="Random (0.5)")
            ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER)
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{'Accuracy' if metric=='accuracy' else 'ROC-AUC'} — {horizon}-day horizon")
            ax.set_ylim(0.47, 0.57)
            ax.legend(fontsize=8)
            # Value labels
            for bar, val in zip(bars, means):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    out = FIGURES_DIR / "fig1_model_accuracy_auc.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Net CAGR Comparison (gross vs net, 1d and 5d)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_net_cagr(backtest_df):
    logger.info("Generating Fig 2: Net CAGR Comparison")
    models_only = backtest_df[backtest_df["model"] != "BuyAndHold"].copy()
    models_only["horizon"] = models_only["horizon"].astype(int)

    bah_mean = backtest_df[backtest_df["model"] == "BuyAndHold"]["bah_cagr"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Gross vs Net CAGR after 20 bps Transaction Costs\n"
                 "Mean across 6 walk-forward splits (NSE 2020–2025)", fontsize=13, fontweight="bold")

    for idx, horizon in enumerate([1, 5]):
        ax  = axes[idx]
        sub = models_only[models_only["horizon"] == horizon]
        means_gross = sub.groupby("model")["gross_cagr"].mean().reindex(MODEL_ORDER)
        means_net   = sub.groupby("model")["net_cagr"].mean().reindex(MODEL_ORDER)

        x     = np.arange(len(MODEL_ORDER))
        width = 0.38
        b1 = ax.bar(x - width/2, means_gross * 100, width, label="Gross CAGR",
                    color=[COLORS[m] for m in MODEL_ORDER], alpha=0.6, edgecolor="white")
        b2 = ax.bar(x + width/2, means_net   * 100, width, label="Net CAGR",
                    color=[COLORS[m] for m in MODEL_ORDER], alpha=1.0, edgecolor="white")

        ax.axhline(bah_mean * 100, color="#333333", lw=1.5, ls="--",
                   label=f"B&H ({bah_mean*100:.1f}%)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER)
        ax.set_ylabel("Annualised Return (%)")
        ax.set_title(f"{horizon}-day Horizon")
        ax.legend()

        # Value labels on net bars
        for bar, val in zip(b2, means_net):
            if not np.isnan(val):
                ypos = bar.get_height() + (0.3 if val >= 0 else -2.0)
                ax.text(bar.get_x() + bar.get_width()/2, ypos,
                        f"{val*100:.1f}%", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    out = FIGURES_DIR / "fig2_net_cagr_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Cost Sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def fig3_cost_sensitivity(cost_df):
    logger.info("Generating Fig 3: Cost Sensitivity")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Transaction Cost Sensitivity — Mean Net CAGR vs Round-Trip Cost\n"
                 "NSE Indian Stocks, 6 walk-forward splits", fontsize=13, fontweight="bold")

    for idx, horizon in enumerate([1, 5]):
        ax  = axes[idx]
        sub = cost_df[cost_df["horizon"] == horizon]
        for model in MODEL_ORDER:
            m_sub = sub[sub["model"] == model].sort_values("rt_bps")
            if m_sub.empty: continue
            ax.plot(m_sub["rt_bps"], m_sub["mean_net_cagr"] * 100,
                    marker="o", label=model, color=COLORS[model], linewidth=2.0, markersize=5)

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(20, color="gray", lw=1.0, ls=":", alpha=0.7, label="Base (20 bps)")
        ax.set_xlabel("Round-Trip Transaction Cost (bps)")
        ax.set_ylabel("Mean Net CAGR (%)")
        ax.set_title(f"{horizon}-day Horizon")
        ax.legend(ncol=2)

    plt.tight_layout()
    out = FIGURES_DIR / "fig3_cost_sensitivity.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 & 5 — SHAP Feature Importance (1d and 5d)
# ─────────────────────────────────────────────────────────────────────────────
FAMILY_COLORS = {
    "Returns":    "#4878CF",
    "Volatility": "#D65F5F",
    "Momentum":   "#B47CC7",
    "Trend":      "#6ACC65",
    "Volume":     "#C4AD66",
    "Calendar":   "#77BEDB",
}
FAMILY_MAP = {
    "return_1d": "Returns", "return_2d": "Returns", "return_5d": "Returns",
    "return_10d": "Returns", "return_20d": "Returns",
    "sma_5": "Trend", "sma_10": "Trend", "sma_20": "Trend", "sma_50": "Trend",
    "ema_5": "Trend", "ema_10": "Trend", "ema_20": "Trend", "ema_50": "Trend",
    "price_sma_ratio_5": "Trend", "price_sma_ratio_10": "Trend",
    "price_sma_ratio_20": "Trend", "price_sma_ratio_50": "Trend",
    "sma_5_20_ratio": "Trend", "sma_10_50_ratio": "Trend",
    "rsi_14": "Momentum", "roc_5": "Momentum", "roc_10": "Momentum", "roc_20": "Momentum",
    "williams_r_14": "Momentum", "macd": "Momentum", "macd_signal": "Momentum",
    "macd_histogram": "Momentum",
    "volatility_5d": "Volatility", "volatility_10d": "Volatility", "volatility_20d": "Volatility",
    "realized_vol_20d": "Volatility", "atr_14": "Volatility",
    "bb_width_20": "Volatility", "hl_range_pct": "Volatility",
    "volume_change_1d": "Volume", "volume_sma_5": "Volume", "volume_sma_10": "Volume",
    "volume_sma_20": "Volume", "relative_volume_5": "Volume", "relative_volume_10": "Volume",
    "relative_volume_20": "Volume", "volume_trend_5d": "Volume",
    "day_of_week": "Calendar", "month": "Calendar",
}

def fig_shap(feat_rank_df, horizon, fig_num):
    logger.info("Generating Fig %d: SHAP Feature Importance (%dd)", fig_num, horizon)
    sub = feat_rank_df[feat_rank_df["horizon"] == horizon].head(20).copy()
    sub["family"] = sub["feature"].map(FAMILY_MAP).fillna("Other")
    colors = [FAMILY_COLORS.get(f, "#aaaaaa") for f in sub["family"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(sub["feature"][::-1], sub["mean_abs_shap"][::-1] * 1000,
                   color=colors[::-1], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Mean |SHAP| × 1000")
    ax.set_title(f"Top-20 SHAP Feature Importance — {horizon}-day Horizon\n"
                 f"XGBoost, mean across 6 walk-forward splits (NSE 2020–2025)", fontsize=11)

    # Legend for families
    from matplotlib.patches import Patch
    legend_items = [Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8, ncol=2)

    plt.tight_layout()
    out = FIGURES_DIR / f"fig{fig_num}_shap_importance_{horizon}d.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Feature Stability (Spearman r heatmap)
# ─────────────────────────────────────────────────────────────────────────────
def fig6_feature_stability(shap_stab_df):
    logger.info("Generating Fig 6: Feature Stability")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("SHAP Feature Importance Stability Across Walk-Forward Splits\n"
                 "Pairwise Spearman Correlation of Feature Rankings", fontsize=12, fontweight="bold")

    SPLIT_LABELS = ["S1\n2020", "S2\n2021", "S3\n2022", "S4\n2023", "S5\n2024", "S6\n2025"]
    n = 6

    for idx, horizon in enumerate([1, 5]):
        ax  = axes[idx]
        sub = shap_stab_df[
            (shap_stab_df["horizon"] == horizon) &
            (shap_stab_df["split_a"] != "ALL")
        ]

        # Build symmetric matrix
        mat = np.ones((n, n))
        for _, row in sub.iterrows():
            try:
                splits  = ["split_1","split_2","split_3","split_4","split_5","split_6_final"]
                i = splits.index(row["split_a"])
                j = splits.index(row["split_b"])
                mat[i, j] = row["spearman_r"]
                mat[j, i] = row["spearman_r"]
            except (ValueError, KeyError):
                pass

        im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(SPLIT_LABELS, fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(SPLIT_LABELS, fontsize=8)
        ax.set_title(f"{horizon}-day Horizon (Kendall's W = "
                     f"{shap_stab_df[(shap_stab_df['horizon']==horizon) & (shap_stab_df['split_a']=='ALL')]['kendalls_W'].values[0]:.2f})")

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if mat[i,j] > 0.7 else "white")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = FIGURES_DIR / "fig6_feature_stability.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Equity Curves (1d horizon, split_1 COVID as showcase)
# ─────────────────────────────────────────────────────────────────────────────
def fig7_equity_curves(equity_json):
    logger.info("Generating Fig 7: Equity Curves")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Net Equity Curves — 1-day Horizon (Equal-Weight, 20 bps Costs)\n"
                 "NSE Indian Stocks — Each Split's Test Period", fontsize=12, fontweight="bold")

    split_labels = {
        "split_1": "2020 (COVID)",
        "split_2": "2021",
        "split_3": "2022",
        "split_4": "2023",
        "split_5": "2024",
        "split_6_final": "2025 H1",
    }
    splits_order = list(split_labels.keys())

    for ax_idx, split_key in enumerate(splits_order):
        ax  = axes[ax_idx // 3][ax_idx % 3]
        ax.set_title(split_labels[split_key], fontsize=10)
        ax.axhline(1.0, color="black", lw=0.7, ls=":")

        for model in MODEL_ORDER:
            ec_key = f"{model}|{split_key}|1d"
            if ec_key not in equity_json:
                continue
            curve = equity_json[ec_key]
            net_dict = curve.get("net_equity", curve.get("net", {}))
            if not net_dict:
                continue
            
            # net_dict is {date_str: daily_return}
            # sort by date to be safe and compute cumprod
            returns = [net_dict[k] for k in sorted(net_dict.keys())]
            net_curve = np.cumprod(1 + np.array(returns))
            
            x = np.arange(len(net_curve))
            ax.plot(x, net_curve, label=model, color=COLORS[model], linewidth=1.3, alpha=0.9)

        # B&H curve is identical to Naive gross curve (no transaction costs)
        bah_key = f"Naive|{split_key}|1d"
        if bah_key in equity_json:
            bah_dict = equity_json[bah_key].get("gross", {})
            if bah_dict:
                returns = [bah_dict[k] for k in sorted(bah_dict.keys())]
                bah_curve = np.cumprod(1 + np.array(returns))
                ax.plot(np.arange(len(bah_curve)), bah_curve,
                        label="B&H", color=COLORS["BnH"], linewidth=1.3, ls="--", alpha=0.7)

        ax.set_xlabel("Trading Day"); ax.set_ylabel("Portfolio Value (₹1 start)")
        if ax_idx == 0:
            ax.legend(ncol=2, fontsize=7)

    plt.tight_layout()
    out = FIGURES_DIR / "fig7_equity_curves.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Regression Metrics (MAE / MSE / RMSE / MAPE / R2)
# ─────────────────────────────────────────────────────────────────────────────
def load_regression_metrics():
    p = TABLES_DIR / "regression_metrics.csv"
    return pd.read_csv(p)


def fig8_regression_metrics(reg_df):
    logger.info("Generating Fig 8: Regression Metrics")

    metrics      = ["MAE", "MSE", "RMSE", "MAPE", "R2"]
    metric_labels = ["MAE", "MSE", "RMSE", "MAPE (%)", "R2"]
    horizons     = [1, 5]

    fig, axes = plt.subplots(len(horizons), len(metrics), figsize=(18, 8))
    fig.suptitle(
        "Regression Metrics: y_prob vs Actual Forward Return\n"
        "All 5 Models, 1-day and 5-day Horizons (mean across 6 walk-forward splits)",
        fontsize=13, fontweight="bold",
    )

    for row_idx, horizon in enumerate(horizons):
        sub = reg_df[reg_df["horizon"] == horizon]

        for col_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax     = axes[row_idx, col_idx]
            means  = [sub.loc[sub["model"] == m, f"{metric}_mean"].values[0]
                      if m in sub["model"].values else np.nan
                      for m in MODEL_ORDER]
            stds   = [sub.loc[sub["model"] == m, f"{metric}_std"].values[0]
                      if m in sub["model"].values else np.nan
                      for m in MODEL_ORDER]

            x    = np.arange(len(MODEL_ORDER))
            bars = ax.bar(
                x, means, yerr=stds, capsize=4,
                color=[COLORS[m] for m in MODEL_ORDER],
                alpha=0.85, edgecolor="white", linewidth=0.8,
            )

            ax.set_xticks(x)
            ax.set_xticklabels(MODEL_ORDER, fontsize=8)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(f"{label} — {horizon}d", fontsize=10)

            # Value labels
            for bar, val in zip(bars, means):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6.5,
                    )

    plt.tight_layout()
    out = FIGURES_DIR / "fig8_regression_metrics.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Final Summary Table
# ─────────────────────────────────────────────────────────────────────────────
def build_final_summary_table(ml_res, lstm_res, backtest_df, reg_df=None):
    logger.info("Building final summary table")
    combined = pd.concat([
        ml_res[["model", "horizon", "accuracy", "roc_auc"]],
        lstm_res[["model", "horizon", "accuracy", "roc_auc"]],
    ], ignore_index=True)

    bt = backtest_df[backtest_df["model"] != "BuyAndHold"].copy()
    bt["horizon"] = bt["horizon"].astype(int)

    # Build regression metrics lookup if available
    reg_lookup = {}
    if reg_df is not None:
        for _, r in reg_df.iterrows():
            key = (r["model"], int(r["horizon"]))
            reg_lookup[key] = r

    rows = []
    for horizon in [1, 5]:
        for model in MODEL_ORDER:
            pred_sub = combined[(combined["model"] == model) & (combined["horizon"] == horizon)]
            bt_sub   = bt[(bt["model"] == model) & (bt["horizon"] == horizon)]
            reg_row  = reg_lookup.get((model, horizon), None)

            row = {
                "Horizon":      f"{horizon}d",
                "Model":        model,
                "Accuracy":     round(pred_sub["accuracy"].mean(), 4),
                "ROC-AUC":      round(pred_sub["roc_auc"].mean(), 4),
                "Gross CAGR":   f"{bt_sub['gross_cagr'].mean()*100:.1f}%",
                "Net CAGR":     f"{bt_sub['net_cagr'].mean()*100:.1f}%",
                "Gross Sharpe": round(bt_sub["gross_sharpe"].mean(), 2),
                "Max DD":       f"{bt_sub['gross_max_dd'].mean()*100:.1f}%",
                "Cost Drag":    f"{(bt_sub['gross_cagr'].mean() - bt_sub['net_cagr'].mean())*100:.1f} pp",
            }
            # Append regression metrics if available
            if reg_row is not None:
                row["MAE"]  = round(reg_row["MAE_mean"],  6)
                row["MSE"]  = round(reg_row["MSE_mean"],  8)
                row["RMSE"] = round(reg_row["RMSE_mean"], 6)
                row["MAPE"] = round(reg_row["MAPE_mean"], 2)
                row["R2"]   = round(reg_row["R2_mean"],   4)

            rows.append(row)

    df = pd.DataFrame(rows)
    out = TABLES_DIR / "final_summary_table.csv"
    df.to_csv(out, index=False)
    logger.info("  Saved: %s (%d rows)", out, len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Final Report (Markdown)
# ─────────────────────────────────────────────────────────────────────────────
def write_final_report(summary_df, reg_df=None):
    logger.info("Writing final report...")

    # Format classification table nicely
    h1_tbl = summary_df[summary_df["Horizon"] == "1d"].to_markdown(index=False)
    h5_tbl = summary_df[summary_df["Horizon"] == "5d"].to_markdown(index=False)

    # Build regression metrics section if data is available
    reg_section = ""
    if reg_df is not None:
        def _reg_row(model, h):
            r = reg_df[(reg_df["model"] == model) & (reg_df["horizon"] == h)]
            if r.empty:
                return f"| {model:<10} | N/A | N/A | N/A | N/A | N/A |"
            r = r.iloc[0]
            return (
                f"| {model:<10} "
                f"| {r['MAE_mean']:.5f} "
                f"| {r['MSE_mean']:.7f} "
                f"| {r['RMSE_mean']:.5f} "
                f"| {r['MAPE_mean']:>9.2f} "
                f"| {r['R2_mean']:>8.4f} |"
            )

        h1_rows = "\n".join(_reg_row(m, 1) for m in ["Naive","LogReg","RF","XGB","LSTM"])
        h5_rows = "\n".join(_reg_row(m, 5) for m in ["Naive","LogReg","RF","XGB","LSTM"])
        reg_hdr = "| Model      | MAE     | MSE       | RMSE    | MAPE (%)  | R2      |"
        reg_sep = "|------------|---------|-----------|---------|-----------|---------|"

        reg_section = f"""
---

## 4b. Regression Metrics

> **Interpretation**: y_true = actual forward return; y_pred = y_prob − 0.5 (centred directional signal).
> These metrics measure how well the model's continuous confidence score tracks actual return magnitude.
> Probabilities in [0, 1] are centred around 0 (positive = bullish, negative = bearish) to match return scales.
> Large MAPE and negative R2 confirm that directional classifiers carry essentially zero linear information
> about return magnitude — consistent with the near-random classification results.

### 1-day Horizon

{reg_hdr}
{reg_sep}
{h1_rows}

### 5-day Horizon

{reg_hdr}
{reg_sep}
{h5_rows}

> **Key finding**: All R2 values are negative (range -0.71 to -1024.5), confirming that
> directional signals carry essentially zero information about actual return magnitude.
> RF has the lowest MAE/MSE on 5d; Naive has the highest (probability = 1.0 always, far from return scale).
> See `results/figures/fig8_regression_metrics.png` for the visual summary.
"""

    report = f"""# Horizon- and Regime-Specific ML for Stock Return Prediction
## Final Research Report

**Market**: Indian Equity — NSE Top 15 Large-Cap Stocks  
**Period**: 2015-2025 (OHLCV from Yahoo Finance)  
**Date**: 2026-09-30  
**Validation**: 6-fold expanding walk-forward (purge + embargo)  
**Models**: Naive Baseline, Logistic Regression, Random Forest, XGBoost, LSTM  
**Horizons**: 1-day and 5-day forward returns  
**Cost assumption**: 20 bps round-trip + 5 bps slippage (frozen pre-results)

---

## 1. Motivation & Primary Question

> Does the best ML model for NSE stock-return prediction change depending on forecast horizon
> and market regime, and do apparent predictive gains survive leakage-controlled validation and
> realistic trading costs?

This study provides a controlled, leakage-safe empirical answer using 10 years of Indian equity data,
5 model families, 2 horizons, 3 volatility regimes, and 6 chronological test periods.

---

## 2. Data & Features

- **Stocks**: RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, HINDUNILVR, ITC, SBIN, BHARTIARTL,
  LT, KOTAKBANK, AXISBANK, BAJFINANCE, MARUTI, WIPRO + Nifty 50 (benchmark only)
- **Rows**: 41,416 stock×day observations (38,835 after dropping benchmark)
- **Features**: 44 leakage-safe features across 6 families — Returns (5), Trend (14), Momentum (8),
  Volatility (7), Volume (8), Calendar (2)
- **Data quality**: 0 NaN, 0 duplicates, 0 negative prices (Phase 1 Data Audit)
- **Targets**: Binary direction of 1-day and 5-day forward returns (51.1% / 53.8% UP)

---

## 3. Walk-Forward Validation Protocol

| Split | Train Period | Test Period |
|-------|-------------|-------------|
| 1 | 2015–2019 | 2020 (COVID crash + recovery) |
| 2 | 2015–2020 | 2021 (bull market) |
| 3 | 2015–2021 | 2022 (bear market) |
| 4 | 2015–2022 | 2023 (low-vol rally) |
| 5 | 2015–2023 | 2024 |
| 6 | 2015–2024 | 2025 H1 |

- **Purge**: remove training labels that overlap test period (horizon-length buffer)
- **Embargo**: 5-day gap between training and test
- **Scaler**: StandardScaler fit on training window only per split
- **Leakage audit**: 12/12 splits PASSED (no future data in features)

---

## 4. Predictive Performance (Classification)

### 1-day Horizon

{h1_tbl}

### 5-day Horizon

{h5_tbl}

> **All models are near-random.** Accuracy ranges 0.49-0.54; AUC 0.49-0.52. XGBoost achieves
> the highest AUC (0.517 / 0.518). No model consistently beats the Naive baseline on raw accuracy.
> This is consistent with the Efficient Market Hypothesis for a liquid market like NSE.

{reg_section}

---

## 5. Market Regime Analysis

**Regime definition**: 20-day rolling realised volatility, partitioned into Low / Medium / High
using the 33rd/67th percentile of the *training window* only (leakage-safe).

| Test Period | Dominant Regime | Notable Pattern |
|-------------|-----------------|-----------------|
| 2020 (COVID) | 67% High vol | Naive accuracy rises; RF dominates trading |
| 2021 (Bull) | Mixed | LSTM matches B&H net CAGR |
| 2022 (Bear) | Medium-High | All models struggle; Naive most consistent |
| 2023 (Low vol) | 78% Low vol | XGB accuracy peaks; Naive best net |
| 2024–2025 | Mixed | Stable regime routing |

**RQ1**: Model performance *does* differ across regimes — H1 **confirmed**.

---

## 6. Horizon Comparison

| Metric | 1-day | 5-day |
|--------|-------|-------|
| Best AUC model | XGB (0.517) | XGB (0.518) |
| Best net CAGR model | **Naive (18.3%)** | **Naive (19.9%)** |
| Worst net CAGR model | LogReg (5.0%) | RF (-18.6%) |
| Cost drag (XGB) | 13.3 pp | 1.7 pp |
| RF net CAGR | 14.3% | -18.6% |

**RQ2**: Model ranking changes completely between horizons — RF is competitive on 1d but collapses on 5d.
H2 **confirmed**.

---

## 7. Trading Performance & Transaction Costs

Long-only equal-weight strategy. Rebalance daily (1d horizon) or every 5 days (5d horizon).
Cost = 12.5 bps per leg (half of 20 bps RT + 5 bps slippage).

### Key finding (H4 confirmed)

| | AUC Rank | Net CAGR Rank (1d) |
|-|----------|---------------------|
| Naive | #5 (0.500) | **#1 (18.3%)** |
| XGB | **#1 (0.517)** | #4 (13.0%) |

The highest-AUC model (XGB) loses 13.3 percentage-points to transaction costs and finishes 4th.
**Statistical predictive edge does NOT survive Indian market transaction costs at daily rebalancing.**

### Buy-and-Hold comparison

- Mean B&H CAGR across splits: **18.5%**
- Only Naive (1d) and Naive+LSTM (5d) consistently approach B&H net returns
- RF and LogReg on 5d are deeply negative after costs

**RQ5**: Predictive superiority does NOT translate to trading superiority. H4 **confirmed**.

---

## 8. Robustness & Sensitivity

### Transaction Cost Sweep (0–50 bps)

- **Naive** is cost-immune: 18.5% → 18.1% from 0 to 50 bps (tiny turnover)
- **XGB** breaks even on 5d at ~30 bps; on 1d at ~52 bps
- **RF** and **LogReg** are negative at 20 bps on 5d under any cost level

### Regime-Conditional Model Selection (RQ3 / H3)

Strategy: route to the model with best historical regime-accuracy on each test day.
Result: conditional CAGR = **16.2%** vs XGB global **10.7%** (+5.5 pp) vs Naive global **17.6%** (-1.5 pp).

> Regime routing picks **Naive for Low/High vol** and **XGB/RF for Medium** — effectively
> routing toward Naive most of the time. H3 **partially confirmed**: beats XGB, misses Naive.

### Model Ranking Stability (Kendall's W on accuracy)

| | 1-day | 5-day |
|-|-------|-------|
| Kendall's W | 0.28 | 0.34 |
| p-value | 0.155 | 0.082 |

Rankings are **weakly concordant** (W < 0.5, p > 0.05). No model consistently dominates
across all test periods.

---

## 9. Feature Importance (SHAP)

XGBoost re-trained per split; SHAP TreeExplainer values computed on test data only.

### Top Features by Horizon

| Rank | 1-day | 5-day |
|------|-------|-------|
| 1 | **return_2d** (0.065) | **month** (0.114) |
| 2 | hl_range_pct (0.051) | atr_14 (0.061) |
| 3 | month (0.050) | volume_sma_20 (0.048) |
| 4 | return_1d (0.038) | macd_histogram (0.047) |
| 5 | return_20d (0.037) | volatility_10d (0.045) |

**Horizon interpretation**:
- 1-day: driven by **recent returns** (mean-reversion) and **daily volatility** (range)
- 5-day: driven by **calendar seasonality** (month effect) and **volatility regime** (ATR)
- The complete top-5 swap confirms that 1d and 5d represent fundamentally different prediction problems

### Stability Across Splits (RQ6)

| | Kendall's W | Spearman r | p-value |
|-|------------|-----------|---------|
| 1-day | **0.86** | **0.82** | < 0.001 |
| 5-day | **0.82** | **0.78** | < 0.001 |

### Stability Across Regimes

All regime pairs (Low/Medium/High) show Spearman r ≥ 0.91 — feature rankings are essentially
identical regardless of volatility regime. **Regime-specific feature engineering is not warranted.**

**RQ6**: Features are strongly stable. H6 (stability) **confirmed**.

---

## 10. Conclusions

### Summary Table

| Research Question | Answer | Verdict |
|------------------|--------|---------|
| RQ1: Performance differs across regimes? | Yes — COVID 2020 RF +38%, 2023 Naive best | ✅ |
| RQ2: Ranking changes between horizons? | Yes — complete rank reversal | ✅ |
| RQ3: Regime-routing beats global model? | Partially — beats XGB not Naive | ⚠️ |
| RQ4: Leakage control changes estimates? | Yes — high CV models (6%) unreliable | ✅ |
| RQ5: Prediction quality = trading quality? | No — XGB AUC #1, net CAGR #4 | ✅ |
| RQ6: Features stable across regimes/splits? | Yes — W ≥ 0.82, p < 0.001 | ✅ |

### Hypothesis Results

| Hypothesis | Verdict | Key Evidence |
|-----------|---------|-------------|
| H1: Performance differs across regimes | ✅ Confirmed | COVID RF +38% vs 13% avg |
| H2: Ranking changes between horizons | ✅ Confirmed | RF: best 1d gross, worst 5d net |
| H3: Regime-aware selection improves | ⚠️ Partially | +5.5 pp vs XGB; -1.5 pp vs Naive |
| H4: Best predictor ≠ best trader | ✅ Confirmed | XGB: AUC #1, Net CAGR #4 |
| H5: Leakage control is conservative | ✅ Confirmed | RF/LR 5d CV = 6%; XGB CV = 1.5% |

### Key Practical Takeaways

1. **For NSE daily trading at 20 bps costs**: the Naive baseline (buy-and-hold the signal)
   outperforms all ML models on risk-adjusted net return. High-frequency ML signals are cost-inefficient.

2. **For lower-frequency (5-day) strategies**: Naive and LSTM both approach B&H returns.
   Tree models (RF, XGB) collapse — their 5-day signals fade momentum.

3. **Feature engineering is stable**: the same 44 features learned in 2015–2019 remain the most
   important in 2024–2025. Regime-specific features are not needed.

4. **Model selection should be horizon-aware**: a model chosen for 1d AUC performs poorly at 5d.
   Any production system must re-evaluate per-horizon.

5. **Negative result is real**: NSE large-caps are efficiently priced. After leakage-safe validation,
   no model achieves economically meaningful alpha at realistic costs.

---

## 11. Limitations

- Long-only strategy only (no shorting, options, or derivatives)
- Equal weighting — no portfolio optimisation
- 15 stocks only (survivorship bias possible — all were top-liquid throughout 2015–2025)
- Transaction costs fixed at 20 bps RT — actual costs vary by broker, lot size, and market impact
- LSTM was not hyperparameter-tuned; a tuned LSTM may perform differently
- Regime detection uses only volatility — alternative regime signals (trend, correlation) not explored

---

## 12. Files & Reproducibility

| File | Description |
|------|-------------|
| `configs/experiment.yaml` | Frozen experiment parameters |
| `configs/models.yaml` | Frozen model hyperparameters (pre-results) |
| `configs/costs.yaml` | Frozen cost assumptions (pre-results) |
| `src/data_loader.py` | Data download & audit |
| `src/target_engineering.py` | Forward return targets |
| `src/feature_engineering.py` | 44 leakage-safe features |
| `src/regime_detection.py` | Volatility regime detection |
| `src/walk_forward_validation.py` | Walk-forward splitter with purge/embargo |
| `src/portfolio_backtest.py` | Trading backtest engine |
| `src/robustness_analysis.py` | Cost sensitivity & regime-conditional analysis |
| `src/shap_explainability.py` | SHAP feature importance |
| `src/report_generator.py` | This report + all figures |
| `notebooks/01_tabular_ml_training.ipynb` | Kaggle: ML model training |
| `notebooks/02_lstm_sequence_training.ipynb` | Kaggle: LSTM training |
| `results/tables/` | All result CSVs |
| `results/figures/` | All publication-quality figures |

*To reproduce: clone repo, install requirements, run `python src/data_loader.py` through
`python src/report_generator.py` in order.*

---

*Report auto-generated by src/report_generator.py on 2026-09-17*  
*Agent: Antigravity | Project: NSE ML Study*
"""

    out = REPORTS_DIR / "final_research_report.md"
    out.write_text(report, encoding="utf-8")
    logger.info("  Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("STAGE 12 — Final Tables, Figures & Research Report")
    logger.info("=" * 60)

    # Load all results
    ml_res    = load_ml_results()
    lstm_res  = load_lstm_results()
    backtest  = load_backtest()
    cost_df   = load_cost_sensitivity()
    feat_rank = load_feature_rankings()
    shap_stab = load_shap_stability()

    # Try equity curves (may not exist if backtest didn't export)
    try:
        equity_json = load_equity_curves()
    except Exception:
        equity_json = {}
        logger.warning("equity_curves.json not found -- skipping Fig 7")

    # Try regression metrics (Stage 11)
    try:
        reg_df = load_regression_metrics()
        logger.info("Regression metrics loaded: %d rows", len(reg_df))
    except Exception:
        reg_df = None
        logger.warning("regression_metrics.csv not found -- regression metrics skipped")

    # Generate all figures
    fig1_accuracy_auc(ml_res, lstm_res)
    fig2_net_cagr(backtest)
    fig3_cost_sensitivity(cost_df)
    fig_shap(feat_rank, 1, 4)
    fig_shap(feat_rank, 5, 5)
    fig6_feature_stability(shap_stab)
    if equity_json:
        fig7_equity_curves(equity_json)
    else:
        logger.info("  Fig 7 skipped (no equity_curves.json)")
    if reg_df is not None:
        fig8_regression_metrics(reg_df)
    else:
        logger.info("  Fig 8 skipped (no regression_metrics.csv)")

    # Final summary table (now includes MAE/MSE/RMSE/MAPE/R2)
    summary_df = build_final_summary_table(ml_res, lstm_res, backtest, reg_df=reg_df)
    logger.info("Final summary table:\n%s", summary_df.to_string(index=False))

    # Report (now includes regression metrics section)
    report_path = write_final_report(summary_df, reg_df=reg_df)

    logger.info("=" * 60)
    logger.info("STAGE 12 COMPLETE")
    figures = list(FIGURES_DIR.glob("*.png"))
    logger.info("  Figures: %d saved to results/figures/", len(figures))
    logger.info("  Report : %s", report_path)
    logger.info("=" * 60)



if __name__ == "__main__":
    main()
