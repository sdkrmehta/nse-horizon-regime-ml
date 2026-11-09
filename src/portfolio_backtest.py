"""
portfolio_backtest.py — Trading Strategy & Backtest Evaluation
Project: Horizon- and Regime-Specific ML for Stock Return Prediction
Market: Indian Equity (NSE) exclusively

Strategy (Long-Only, Equal-Weight):
  - Signal = y_pred from ML/LSTM model
  - If y_pred == 1 (predict UP) for a stock on day t → go long that stock
  - If y_pred == 0 → stay in cash for that stock
  - Equal weight across all "buy" positions on each day
  - Cash earns 0 (no risk-free rate assumption — conservative)

Transaction Costs (from configs/costs.yaml — fixed BEFORE results seen):
  - Round-trip cost: 20 bps (0.20%)
  - Slippage: 5 bps (0.05%)
  - Applied on TURNOVER only: cost = (entering_stocks + exiting_stocks) / total × half_RT

For 5-day horizon:
  - Non-overlapping windows: rebalance every 5 trading days
  - Signal = mode of daily predictions over each 5-day window per stock
  - Return = 5-day cumulative return for held stocks

Metrics computed per model × split × horizon:
  - Cumulative return
  - Annualised return (CAGR)
  - Annualised Sharpe ratio (daily returns, no risk-free rate)
  - Maximum drawdown
  - Calmar ratio (CAGR / |MaxDD|)
  - Annualised turnover
  - Net metrics (after transaction costs)
  - Per-regime breakdown

HARD CONSTRAINT: Trading signals use only data available at trade time.
  All signals come from pre-computed predictions (tabular ML and LSTM sequence models).
  No look-ahead used here.

Usage:
    python src/portfolio_backtest.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH  = PROJECT_ROOT / "configs" / "experiment.yaml"
COSTS_PATH   = PROJECT_ROOT / "configs" / "costs.yaml"
RESULTS_DIR  = PROJECT_ROOT / "results" / "tables"
PRED5_PATH   = PROJECT_ROOT / "artifacts" / "kaggle" / "tabular_ml_models" / "predictions_all_splits.csv"
PRED6_PATH   = PROJECT_ROOT / "artifacts" / "kaggle" / "lstm_sequence_models" / "lstm_predictions_all_splits.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Load configs
# ────────────────────────────────────────────────────────────
def load_configs():
    with open(CONFIG_PATH) as f:
        exp_cfg = yaml.safe_load(f)
    with open(COSTS_PATH) as f:
        cost_cfg = yaml.safe_load(f)
    return exp_cfg, cost_cfg


# ────────────────────────────────────────────────────────────
# Portfolio Metrics
# ────────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252

def compute_portfolio_metrics(daily_returns: pd.Series, label: str = "") -> dict:
    """
    Compute trading performance metrics from a series of daily portfolio returns.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily gross portfolio returns (as fractions, e.g. 0.01 = 1%).
    label : str
        Optional prefix for result keys.
    """
    r = daily_returns.dropna()
    if len(r) == 0:
        return {}

    cum_ret      = (1 + r).prod() - 1
    n_days       = len(r)
    n_years      = n_days / TRADING_DAYS_PER_YEAR
    cagr         = (1 + cum_ret) ** (1 / max(n_years, 1/252)) - 1
    vol          = r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe       = (r.mean() * TRADING_DAYS_PER_YEAR) / vol if vol > 0 else 0.0

    # Maximum drawdown
    cum_curve = (1 + r).cumprod()
    peak      = cum_curve.cummax()
    dd        = (cum_curve - peak) / peak
    max_dd    = dd.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    prefix = f"{label}_" if label else ""
    return {
        f"{prefix}cum_return":   round(cum_ret,   4),
        f"{prefix}cagr":         round(cagr,      4),
        f"{prefix}sharpe":       round(sharpe,    4),
        f"{prefix}max_dd":       round(max_dd,    4),
        f"{prefix}calmar":       round(calmar,    4) if not np.isnan(calmar) else None,
        f"{prefix}volatility":   round(vol,       4),
        f"{prefix}n_days":       n_days,
        f"{prefix}n_years":      round(n_years,   3),
    }


# ────────────────────────────────────────────────────────────
# 1-Day Horizon Backtest
# ────────────────────────────────────────────────────────────
def backtest_1d(
    preds: pd.DataFrame,
    half_rt_cost: float,
) -> tuple[dict, pd.Series, pd.Series]:
    """
    Run daily-rebalancing backtest for the 1-day horizon.

    Returns
    -------
    metrics : dict
    gross_daily : pd.Series — daily gross returns
    net_daily   : pd.Series — daily net returns (after transaction costs)
    """
    # Pivot: rows=Date, cols=Ticker, values=y_pred
    signal_pivot = preds.pivot_table(
        index="Date", columns="Ticker", values="y_pred", aggfunc="first"
    ).fillna(0)

    # Pivot: rows=Date, cols=Ticker, values=fwd_return_1d
    return_pivot = preds.pivot_table(
        index="Date", columns="Ticker", values="fwd_return_1d", aggfunc="first"
    )

    dates = signal_pivot.index.sort_values()
    gross_rets, net_rets = [], []
    prev_positions = pd.Series(0.0, index=signal_pivot.columns)

    for date in dates:
        signals  = signal_pivot.loc[date]
        n_long   = signals.sum()
        fwd_rets = return_pivot.loc[date] if date in return_pivot.index else pd.Series(0.0, index=signal_pivot.columns)

        if n_long > 0:
            # Equal-weight portfolio of all "buy" signals
            weights = signals / n_long
            gross_ret = (weights * fwd_rets.fillna(0)).sum()

            # Transaction cost = half-RT cost × turnover fraction
            turnover  = (signals != prev_positions).sum() / len(signals)
            cost      = half_rt_cost * turnover
            net_ret   = gross_ret - cost
        else:
            # All in cash
            gross_ret = 0.0
            net_ret   = 0.0
            turnover  = 0.0

        gross_rets.append(gross_ret)
        net_rets.append(net_ret)
        prev_positions = signals

    gross_daily = pd.Series(gross_rets, index=dates, name="gross")
    net_daily   = pd.Series(net_rets,   index=dates, name="net")

    gross_m = compute_portfolio_metrics(gross_daily, "gross")
    net_m   = compute_portfolio_metrics(net_daily,   "net")
    metrics = {**gross_m, **net_m}

    # Annualised turnover (average daily turnover × 252)
    avg_daily_turnover = pd.Series([
        (signal_pivot.loc[dates[i]] != signal_pivot.loc[dates[i-1]]).sum() / len(signal_pivot.columns)
        for i in range(1, len(dates))
    ]).mean()
    metrics["annual_turnover"] = round(avg_daily_turnover * TRADING_DAYS_PER_YEAR, 2)

    return metrics, gross_daily, net_daily


# ────────────────────────────────────────────────────────────
# 5-Day Horizon Backtest
# ────────────────────────────────────────────────────────────
def backtest_5d(
    preds: pd.DataFrame,
    half_rt_cost: float,
) -> tuple[dict, pd.Series, pd.Series]:
    """
    Non-overlapping 5-day rebalancing backtest.

    On each rebalancing date (every 5 trading days):
      - Signal = majority vote (mode) of daily y_pred across 5-day window per stock
      - Held position earns the 5-day fwd_return for each held stock
      - Transaction cost applied on turnover
    """
    dates = sorted(preds["Date"].unique())
    # Create non-overlapping 5-day windows
    windows = [dates[i:i+5] for i in range(0, len(dates), 5)]

    gross_rets, net_rets, window_dates = [], [], []
    prev_positions = {}

    for window in windows:
        if len(window) == 0:
            continue

        rebal_date = window[0]
        window_preds = preds[preds["Date"].isin(window)]

        # Per-ticker majority signal in this window
        ticker_signals = (
            window_preds.groupby("Ticker")["y_pred"]
            .agg(lambda x: int(x.mode().iloc[0]))
        )

        # 5-day forward return: take first available fwd_return_5d per ticker
        ticker_rets = (
            window_preds[window_preds["fwd_return_5d"].notna()]
            .groupby("Ticker")["fwd_return_5d"]
            .first()
        )

        n_long = ticker_signals.sum()

        if n_long > 0:
            long_tickers = ticker_signals[ticker_signals == 1].index
            weights  = 1.0 / n_long
            gross_ret = ticker_rets.reindex(long_tickers).fillna(0).sum() * weights

            # Turnover cost
            prev_set   = set(k for k, v in prev_positions.items() if v == 1)
            curr_set   = set(long_tickers)
            entered    = len(curr_set - prev_set)
            exited     = len(prev_set - curr_set)
            total_pos  = max(len(ticker_signals), 1)
            turnover   = (entered + exited) / total_pos
            cost       = half_rt_cost * turnover
            net_ret    = gross_ret - cost
        else:
            gross_ret = net_ret = 0.0

        gross_rets.append(gross_ret)
        net_rets.append(net_ret)
        window_dates.append(rebal_date)
        prev_positions = ticker_signals.to_dict()

    # Convert 5-day period returns to approximate daily returns for comparability
    gross_daily = pd.Series(
        [r / 5 for r in gross_rets], index=window_dates, name="gross"
    )
    net_daily = pd.Series(
        [r / 5 for r in net_rets], index=window_dates, name="net"
    )

    gross_m = compute_portfolio_metrics(gross_daily, "gross")
    net_m   = compute_portfolio_metrics(net_daily,   "net")
    metrics = {**gross_m, **net_m}
    metrics["annual_turnover"] = round(len(windows) * 2 / len(dates) * TRADING_DAYS_PER_YEAR, 2)

    return metrics, gross_daily, net_daily


# ────────────────────────────────────────────────────────────
# Benchmark: Buy-and-Hold equal-weight portfolio
# ────────────────────────────────────────────────────────────
def compute_benchmark(preds_1d: pd.DataFrame) -> dict:
    """
    Equal-weight buy-and-hold of all NSE stocks in universe for the test period.
    Uses 1-day fwd_return_1d (daily return of each stock).
    """
    daily_ret = (
        preds_1d
        .groupby("Date")["fwd_return_1d"]
        .mean()  # equal-weight average of all stocks
    )
    return compute_portfolio_metrics(daily_ret, "bah")


# ────────────────────────────────────────────────────────────
# Per-Regime Metrics
# ────────────────────────────────────────────────────────────
REGIME_COL_MAP = {
    "split_1":       "regime_split_1",
    "split_2":       "regime_split_2",
    "split_3":       "regime_split_3",
    "split_4":       "regime_split_4",
    "split_5":       "regime_split_5",
    "split_6_final": "regime_split_6_final",
}
REGIME_NAMES = {0: "low", 1: "medium", 2: "high"}


def backtest_per_regime(
    preds: pd.DataFrame,
    horizon: int,
    half_rt_cost: float,
    split_label: str,
) -> dict:
    """Run separate backtest for each volatility regime."""
    regime_col = REGIME_COL_MAP.get(split_label)
    if regime_col not in preds.columns:
        # LSTM predictions use generic 'regime' column
        regime_col = "regime" if "regime" in preds.columns else None

    if regime_col is None:
        return {}

    result = {}
    for regime_id, regime_name in REGIME_NAMES.items():
        mask = preds[regime_col] == regime_id
        sub  = preds[mask]
        if len(sub) < 10:
            continue

        if horizon == 1:
            m, _, _ = backtest_1d(sub, half_rt_cost)
        else:
            m, _, _ = backtest_5d(sub, half_rt_cost)

        for key, val in m.items():
            result[f"regime_{regime_name}_{key}"] = val

    return result


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("BACKTEST — Trading Strategy Evaluation")
    logger.info("=" * 60)

    exp_cfg, cost_cfg = load_configs()
    splits_cfg   = exp_cfg["walk_forward"]["splits"]

    # Transaction cost: half round-trip (entry or exit)
    rt_cost      = cost_cfg["round_trip_cost_pct"]           # 0.002 = 20 bps (root level)
    slippage     = cost_cfg.get("slippage_bps", 5) / 10000  # 5 bps
    half_rt_cost = (rt_cost + slippage) / 2                 # applied per leg
    logger.info("Round-trip cost: %.1f bps | Slippage: %.0f bps | Half-RT: %.2f bps",
                rt_cost * 10000, slippage * 10000, half_rt_cost * 10000)

    # ── Load predictions ─────────────────────────────────────
    logger.info("Loading tabular ML model predictions...")
    ml_preds = pd.read_csv(PRED5_PATH, parse_dates=["Date"])

    logger.info("Loading LSTM sequence model predictions...")
    lstm_preds = pd.read_csv(PRED6_PATH, parse_dates=["Date"])

    # Standardise LSTM column name to match ML predictions
    if "y_true" in lstm_preds.columns and "fwd_label_1d" not in lstm_preds.columns:
        lstm_preds = lstm_preds.rename(columns={"y_true": "fwd_label"})

    # Combine all predictions
    all_preds = pd.concat([ml_preds, lstm_preds], ignore_index=True)
    models    = all_preds["model"].unique()
    logger.info("Models: %s", list(models))

    # ── Run backtests ─────────────────────────────────────────
    all_results   = []
    all_equity    = {}   # equity curves per model×split×horizon

    for split_cfg in splits_cfg:
        label      = split_cfg["label"]
        test_start = split_cfg["test_start"]
        test_end   = split_cfg["test_end"]

        logger.info("-" * 40)
        logger.info("Split: %s | Test: %s → %s", label, test_start, test_end)

        # Benchmark for this split
        bah_preds = ml_preds[
            (ml_preds["split"] == label) &
            (ml_preds["horizon"] == 1) &
            (ml_preds["model"] == "Naive")
        ]
        bah_metrics = compute_benchmark(bah_preds)
        bah_row = {
            "split": label, "horizon": "all", "model": "BuyAndHold",
            "test_start": test_start, "test_end": test_end,
            **bah_metrics,
        }
        all_results.append(bah_row)
        logger.info("  B&H gross_cagr=%.4f  sharpe=%.4f  max_dd=%.4f",
                    bah_metrics.get("bah_cagr", 0),
                    bah_metrics.get("bah_sharpe", 0),
                    bah_metrics.get("bah_max_dd", 0))

        for horizon in [1, 5]:
            for model in models:
                sub = all_preds[
                    (all_preds["split"] == label) &
                    (all_preds["horizon"] == horizon) &
                    (all_preds["model"] == model)
                ].copy()

                if len(sub) < 20:
                    continue

                # Run backtest
                if horizon == 1:
                    m, gross_eq, net_eq = backtest_1d(sub, half_rt_cost)
                else:
                    m, gross_eq, net_eq = backtest_5d(sub, half_rt_cost)

                # Per-regime breakdown
                regime_m = backtest_per_regime(sub, horizon, half_rt_cost, label)

                row = {
                    "split":      label,
                    "horizon":    horizon,
                    "model":      model,
                    "test_start": test_start,
                    "test_end":   test_end,
                    **m,
                    **regime_m,
                }
                all_results.append(row)

                # Store equity curves
                key = f"{model}|{label}|{horizon}d"
                all_equity[key] = {"gross": gross_eq, "net": net_eq}

                logger.info(
                    "  %-10s h=%dd  gross_cagr=%+.4f  net_cagr=%+.4f  sharpe=%.4f  dd=%.4f",
                    model, horizon,
                    m.get("gross_cagr", 0), m.get("net_cagr", 0),
                    m.get("gross_sharpe", 0), m.get("gross_max_dd", 0),
                )

    # ── Save results ──────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    out_path   = RESULTS_DIR / "backtest_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Saved backtest results: %s (%d rows)", out_path, len(results_df))

    # ── Summary pivot: net CAGR by model × horizon ─────────────
    logger.info("=" * 60)
    logger.info("SUMMARY — Net CAGR (mean across splits)")
    model_results = results_df[results_df["model"] != "BuyAndHold"].copy()

    for h in [1, 5]:
        sub = model_results[model_results["horizon"] == h]
        if sub.empty:
            continue
        pivot = sub.groupby("model")[["gross_cagr", "net_cagr", "gross_sharpe", "gross_max_dd"]].mean()
        logger.info("Horizon %dd:", h)
        logger.info("\n%s", pivot.round(4).to_string())

    # ── Save equity curves as JSON ────────────────────────────
    equity_summary = {}
    for key, curves in all_equity.items():
        equity_summary[key] = {
            "gross": curves["gross"].to_dict(),
            "net":   curves["net"].to_dict(),
        }
    eq_path = RESULTS_DIR / "equity_curves.json"
    with open(eq_path, "w") as f:
        json.dump({k: {"gross": {str(d): v for d, v in c["gross"].items()},
                       "net":   {str(d): v for d, v in c["net"].items()}}
                   for k, c in all_equity.items()}, f)
    logger.info("Equity curves saved: %s", eq_path)

    logger.info("=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 60)

    return results_df


if __name__ == "__main__":
    main()
