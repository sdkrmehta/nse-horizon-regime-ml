"""
robustness_analysis.py — Robustness & Sensitivity Analysis
Optimised: pre-compute signals once, sweep costs analytically.
"""
import json, logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED5_PATH   = PROJECT_ROOT / "artifacts/kaggle/tabular_ml_models/predictions_all_splits.csv"
PRED6_PATH   = PROJECT_ROOT / "artifacts/kaggle/lstm_sequence_models/lstm_predictions_all_splits.csv"
ML_RES_PATH  = PROJECT_ROOT / "artifacts/kaggle/tabular_ml_models/results_all_splits.csv"
LSTM_RES_PATH= PROJECT_ROOT / "artifacts/kaggle/lstm_sequence_models/lstm_results_all_splits.csv"
RESULTS_DIR  = PROJECT_ROOT / "results/tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRADING_DAYS = 252

# ── helpers ───────────────────────────────────────────────────────────────────
def portfolio_metrics(r_arr):
    r = pd.Series(r_arr).dropna()
    if len(r) < 5:
        return dict(cagr=np.nan, sharpe=np.nan, max_dd=np.nan)
    cum   = float((1+r).prod()-1)
    nyrs  = len(r)/TRADING_DAYS
    cagr  = (1+cum)**(1/max(nyrs,1/252))-1
    vol   = float(r.std()*np.sqrt(TRADING_DAYS))
    sharpe= float(r.mean()*TRADING_DAYS/vol) if vol>0 else 0.0
    peak  = (1+r).cumprod().cummax()
    mdd   = float(((1+r).cumprod()/peak-1).min())
    return dict(cagr=round(cagr,4), sharpe=round(sharpe,4), max_dd=round(mdd,4))

# ── pre-compute arrays for one (model, split, horizon) ───────────────────────
def build_daily_arrays_1d(sub: pd.DataFrame):
    """
    Returns (dates, gross_arr, turnover_arr) so we can sweep costs without
    re-pivoting. gross_arr[i] = equal-weight gross return on day i.
    turnover_arr[i] = fraction of positions that changed on day i.
    """
    sig_p = sub.pivot_table(index="Date", columns="Ticker",
                             values="y_pred", aggfunc="first").fillna(0)
    ret_p = sub.pivot_table(index="Date", columns="Ticker",
                             values="fwd_return_1d", aggfunc="first")
    dates = sig_p.index.sort_values()
    prev  = pd.Series(0.0, index=sig_p.columns)
    gross_arr, turn_arr = [], []
    for date in dates:
        sig  = sig_p.loc[date]
        n    = sig.sum()
        fwd  = ret_p.loc[date] if date in ret_p.index else pd.Series(0.0, index=sig.index)
        gross= float((sig/n * fwd.fillna(0)).sum()) if n>0 else 0.0
        turn = float((sig!=prev).sum()/len(sig))
        gross_arr.append(gross)
        turn_arr.append(turn)
        prev = sig
    return dates, np.array(gross_arr), np.array(turn_arr)


def build_period_arrays_5d(sub: pd.DataFrame):
    """Returns (gross_5d, turnover_5d) for non-overlapping 5-day windows."""
    dates   = sorted(sub["Date"].unique())
    windows = [dates[i:i+5] for i in range(0, len(dates), 5)]
    prev    = {}
    gross_5d, turn_5d = [], []
    for w in windows:
        if not w: continue
        wp   = sub[sub["Date"].isin(w)]
        sigs = wp.groupby("Ticker")["y_pred"].agg(lambda x: int(x.mode().iloc[0]))
        rets = wp[wp["fwd_return_5d"].notna()].groupby("Ticker")["fwd_return_5d"].first()
        n    = int(sigs.sum())
        gross= float(rets.reindex(sigs[sigs==1].index).fillna(0).sum()/n) if n>0 else 0.0
        curr  = set(sigs[sigs==1].index); prev_s = set(k for k,v in prev.items() if v==1)
        turn  = (len(curr-prev_s)+len(prev_s-curr))/max(len(sigs),1)
        gross_5d.append(gross/5)
        turn_5d.append(turn)
        prev = sigs.to_dict()
    return np.array(gross_5d), np.array(turn_5d)


# ─────────────────────────────────────────────────────────────────────────────
# A. Transaction Cost Sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def cost_sensitivity(all_preds, splits_cfg):
    logger.info("=== A. Transaction Cost Sensitivity ===")
    COST_BPS = [0, 5, 10, 20, 30, 50]

    # Pre-build arrays for every (model, split, horizon) combo
    logger.info("  Pre-building signal/return arrays...")
    cache_1d, cache_5d = {}, {}
    for model in all_preds["model"].unique():
        for sc in splits_cfg:
            label = sc["label"]
            # 1d
            sub = all_preds[(all_preds["model"]==model) &
                            (all_preds["split"]==label) &
                            (all_preds["horizon"]==1)]
            if len(sub) >= 20:
                cache_1d[(model, label)] = build_daily_arrays_1d(sub)
            # 5d
            sub5 = all_preds[(all_preds["model"]==model) &
                             (all_preds["split"]==label) &
                             (all_preds["horizon"]==5)]
            if len(sub5) >= 20:
                cache_5d[(model, label)] = build_period_arrays_5d(sub5)
    logger.info("  Cached %d 1d and %d 5d arrays.", len(cache_1d), len(cache_5d))

    # Sweep costs analytically
    rows = []
    for rt_bps in COST_BPS:
        half_rt = (rt_bps/10000)/2
        for model in all_preds["model"].unique():
            # 1d
            cagrs_1d = []
            for sc in splits_cfg:
                key = (model, sc["label"])
                if key not in cache_1d: continue
                _, gross_arr, turn_arr = cache_1d[key]
                net_arr = gross_arr - half_rt * turn_arr
                cagrs_1d.append(portfolio_metrics(net_arr)["cagr"])
            if cagrs_1d:
                rows.append({"rt_bps": rt_bps, "model": model, "horizon": 1,
                             "mean_net_cagr": round(np.nanmean(cagrs_1d), 4)})
            # 5d
            cagrs_5d = []
            for sc in splits_cfg:
                key = (model, sc["label"])
                if key not in cache_5d: continue
                gross_arr, turn_arr = cache_5d[key]
                net_arr = gross_arr - half_rt * turn_arr
                cagrs_5d.append(portfolio_metrics(net_arr)["cagr"])
            if cagrs_5d:
                rows.append({"rt_bps": rt_bps, "model": model, "horizon": 5,
                             "mean_net_cagr": round(np.nanmean(cagrs_5d), 4)})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR/"cost_sensitivity.csv", index=False)
    logger.info("  Saved cost_sensitivity.csv (%d rows)", len(df))

    for h in [1, 5]:
        pivot = df[df["horizon"]==h].pivot_table(
            index="model", columns="rt_bps", values="mean_net_cagr").round(4)
        logger.info("Mean Net CAGR h=%dd by RT cost (bps):\n%s", h, pivot.to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# B. Regime-Conditional vs Global
# ─────────────────────────────────────────────────────────────────────────────
def regime_conditional(ml_preds, splits_cfg):
    logger.info("=== B. Regime-Conditional Model Selection ===")
    ml_res = pd.read_csv(ML_RES_PATH)
    SPLITS  = [s["label"] for s in splits_cfg]
    RC_MAP  = {s: f"regime_{s}" for s in SPLITS}
    REGIMES = {0.0:"low", 1.0:"medium", 2.0:"high"}
    half_rt = 0.002/2

    rows = []
    for i, split_label in enumerate(SPLITS):
        if i < 1: continue  # need prior splits for regime accuracy
        prev_splits = SPLITS[:i]
        regime_col  = RC_MAP[split_label]

        # Best model per regime from PRIOR splits only (leakage-safe)
        best = {}
        for rid, rname in REGIMES.items():
            acc_col = f"regime_{rname}_accuracy"
            prior = ml_res[(ml_res["split"].isin(prev_splits)) &
                           (ml_res["horizon"]==1) &
                           ml_res[acc_col].notna()]
            best[rid] = prior.groupby("model")[acc_col].mean().idxmax() if not prior.empty else "XGB"

        logger.info("  %s: regime_best=%s", split_label,
                    {REGIMES[k]: v for k,v in best.items()})

        # Build conditional prediction frame
        test_all = ml_preds[(ml_preds["split"]==split_label) &
                            (ml_preds["horizon"]==1)].copy()
        if regime_col not in test_all.columns: continue

        cond_rows = []
        for (date, ticker), grp in test_all.groupby(["Date","Ticker"]):
            rid = grp[regime_col].iloc[0]
            mdl = best.get(rid, "XGB")
            sel = grp[grp["model"]==mdl]
            if sel.empty: sel = grp[grp["model"]=="XGB"]
            if sel.empty: continue
            cond_rows.append(sel.iloc[0])

        if not cond_rows: continue
        cond_df = pd.DataFrame(cond_rows)
        cond_df["Date"] = pd.to_datetime(cond_df["Date"])

        _, cg, ct = build_daily_arrays_1d(cond_df)
        cond_m = portfolio_metrics(cg - half_rt*ct)

        xgb_df  = test_all[test_all["model"]=="XGB"].copy()
        xgb_df["Date"] = pd.to_datetime(xgb_df["Date"])
        _, xg, xt = build_daily_arrays_1d(xgb_df)
        xgb_m = portfolio_metrics(xg - half_rt*xt)

        naive_df = test_all[test_all["model"]=="Naive"].copy()
        naive_df["Date"] = pd.to_datetime(naive_df["Date"])
        _, ng, nt = build_daily_arrays_1d(naive_df)
        naive_m = portfolio_metrics(ng - half_rt*nt)

        rows.append({"split":split_label,
                     "best_models": str({REGIMES[k]:v for k,v in best.items()}),
                     "cond_cagr":  cond_m["cagr"], "cond_sharpe": cond_m["sharpe"],
                     "xgb_cagr":   xgb_m["cagr"],  "naive_cagr": naive_m["cagr"],
                     "vs_xgb":     round((cond_m["cagr"] or 0)-(xgb_m["cagr"] or 0), 4),
                     "vs_naive":   round((cond_m["cagr"] or 0)-(naive_m["cagr"] or 0), 4)})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR/"regime_conditional.csv", index=False)
    logger.info("  Saved regime_conditional.csv")
    if not df.empty:
        logger.info("\n%s", df[["split","cond_cagr","xgb_cagr","naive_cagr","vs_xgb","vs_naive"]].to_string(index=False))
        logger.info("Mean: cond=%.4f  vs_xgb=%+.4f  vs_naive=%+.4f",
                    df["cond_cagr"].mean(), df["vs_xgb"].mean(), df["vs_naive"].mean())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# C. Prediction Stability
# ─────────────────────────────────────────────────────────────────────────────
def prediction_stability():
    logger.info("=== C. Prediction Stability ===")
    ml   = pd.read_csv(ML_RES_PATH)[["split","horizon","model","accuracy","roc_auc"]]
    lstm = pd.read_csv(LSTM_RES_PATH)[["split","horizon","model","accuracy","roc_auc"]]
    combined = pd.concat([ml, lstm], ignore_index=True)
    rows = []
    for (model, horizon), g in combined.groupby(["model","horizon"]):
        accs = g["accuracy"].dropna(); aucs = g["roc_auc"].dropna()
        rows.append({"model": model, "horizon": horizon,
                     "acc_mean": round(accs.mean(),4), "acc_std": round(accs.std(),4),
                     "acc_cv_pct": round(accs.std()/accs.mean()*100,2) if accs.mean()!=0 else None,
                     "acc_min": round(accs.min(),4), "acc_max": round(accs.max(),4),
                     "auc_mean": round(aucs.mean(),4) if len(aucs)>0 else None,
                     "auc_std":  round(aucs.std(),4) if len(aucs)>0 else None,
                     "n_splits": len(accs)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR/"prediction_stability.csv", index=False)
    logger.info("  Saved prediction_stability.csv")
    logger.info("\n%s", df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# D. Ranking Stability — Kendall's W
# ─────────────────────────────────────────────────────────────────────────────
def ranking_stability():
    logger.info("=== D. Ranking Stability (Kendall's W) ===")
    ml   = pd.read_csv(ML_RES_PATH)[["split","horizon","model","accuracy"]]
    lstm = pd.read_csv(LSTM_RES_PATH)[["split","horizon","model","accuracy"]]
    combined = pd.concat([ml, lstm], ignore_index=True)
    rows = []
    for horizon in [1, 5]:
        sub = combined[combined["horizon"]==horizon]
        splits = sorted(sub["split"].unique())
        models = sorted(sub["model"].unique())
        mat = []
        for sp in splits:
            sp_acc = sub[sub["split"]==sp].set_index("model")["accuracy"].reindex(models)
            mat.append(sp_acc.rank(ascending=False).values)
        mat = np.array(mat)
        m, n = mat.shape
        S = np.sum((mat.sum(axis=0) - m*(n+1)/2)**2)
        W = 12*S / (m**2*(n**3-n))
        chi2 = m*(n-1)*W
        p    = 1-stats.chi2.cdf(chi2, df=n-1)
        interp = ("Strong" if W>0.7 else "Moderate" if W>0.5 else "Weak")
        rows.append({"horizon":horizon,"kendalls_W":round(W,4),"chi2":round(chi2,4),
                     "p_value":round(p,4),"agreement":interp})
        logger.info("  h=%dd  W=%.4f  chi2=%.4f  p=%.4f  [%s agreement]",
                    horizon, W, chi2, p, interp)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR/"ranking_stability.csv", index=False)
    logger.info("  Saved ranking_stability.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("="*60)
    logger.info("STAGE 9 — Robustness & Sensitivity Analysis")
    logger.info("="*60)
    with open(PROJECT_ROOT/"configs/experiment.yaml") as f:
        splits_cfg = yaml.safe_load(f)["walk_forward"]["splits"]

    logger.info("Loading predictions...")
    ml_preds   = pd.read_csv(PRED5_PATH, parse_dates=["Date"])
    lstm_preds = pd.read_csv(PRED6_PATH, parse_dates=["Date"])
    ml_preds["horizon"]   = ml_preds["horizon"].astype(int)
    lstm_preds["horizon"] = lstm_preds["horizon"].astype(int)
    all_preds = pd.concat([ml_preds, lstm_preds], ignore_index=True)
    logger.info("All preds: %s", all_preds.shape)

    cost_df      = cost_sensitivity(all_preds, splits_cfg)
    regime_df    = regime_conditional(ml_preds, splits_cfg)
    stability_df = prediction_stability()
    ranking_df   = ranking_stability()

    summary = {
        "cost_sensitivity_rows": len(cost_df),
        "regime_conditional_splits": len(regime_df),
        "cond_vs_xgb_mean":   round(regime_df["vs_xgb"].mean(),4)   if not regime_df.empty else None,
        "cond_vs_naive_mean": round(regime_df["vs_naive"].mean(),4) if not regime_df.empty else None,
        "kendalls_W_1d": float(ranking_df[ranking_df["horizon"]==1]["kendalls_W"].iloc[0]) if len(ranking_df)>0 else None,
        "kendalls_W_5d": float(ranking_df[ranking_df["horizon"]==5]["kendalls_W"].iloc[0]) if len(ranking_df)>1 else None,
    }
    with open(RESULTS_DIR/"robustness_summary.json","w") as f:
        json.dump(summary, f, indent=2)

    logger.info("="*60)
    logger.info("STAGE 9 COMPLETE — outputs in results/tables/")
    logger.info("="*60)


if __name__ == "__main__":
    main()
