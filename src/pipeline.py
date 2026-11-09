"""
pipeline.py — Master Pipeline Runner
Project: Horizon- and Regime-Specific ML for Stock Return Prediction
Market: Indian Equity (NSE) exclusively

Executes the local analysis pipeline in logical chronological order:
  1. Data Loader & Audit       (src/data_loader.py)
  2. Target Construction       (src/target_engineering.py)
  3. Feature Engineering       (src/feature_engineering.py)
  4. Market Regime Detection   (src/regime_detection.py)
  5. Validation Framework      (src/walk_forward_validation.py)
  6. Backtesting & Costs       (src/portfolio_backtest.py)
  7. Robustness & Sensitivity  (src/robustness_analysis.py)
  8. SHAP Explainability       (src/shap_explainability.py)
  9. Regression Metrics        (src/regression_metrics.py)
  10. Reports & Visualizations (src/report_generator.py)

Usage:
  python src/pipeline.py                          # Run entire pipeline
  python src/pipeline.py --step portfolio_backtest # Run single step
"""

import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")

STEPS = [
    ("data_loader",             "src.data_loader",             "Stage 1: Data Acquisition & Audit"),
    ("target_engineering",      "src.target_engineering",      "Stage 2a: Target Construction"),
    ("feature_engineering",     "src.feature_engineering",     "Stage 2b: Feature Engineering"),
    ("regime_detection",        "src.regime_detection",        "Stage 3: Market Regime Detection"),
    ("walk_forward_validation", "src.walk_forward_validation", "Stage 4: Walk-Forward Validation Setup"),
    ("portfolio_backtest",      "src.portfolio_backtest",      "Stage 5: Backtesting & Trading Costs"),
    ("robustness_analysis",     "src.robustness_analysis",     "Stage 6: Robustness & Sensitivity"),
    ("shap_explainability",     "src.shap_explainability",     "Stage 7: SHAP Feature Importance"),
    ("regression_metrics",      "src.regression_metrics",      "Stage 8: Regression Metrics"),
    ("report_generator",        "src.report_generator",        "Stage 9: Reports & Visualizations"),
]

# Aliases for backward compatibility
STEP_ALIASES = {
    "targets": "target_engineering",
    "features": "feature_engineering",
    "regimes": "regime_detection",
    "validation": "walk_forward_validation",
    "backtest": "portfolio_backtest",
    "robustness": "robustness_analysis",
    "explainability": "shap_explainability",
}


def run_step(step_name: str, module_path: str, description: str):
    logger.info("=" * 70)
    logger.info("STARTING STEP: %s (%s)", step_name, description)
    logger.info("=" * 70)
    try:
        module = __import__(module_path, fromlist=["main"])
        if hasattr(module, "main"):
            module.main()
        else:
            logger.warning("Module %s does not define main(), skipping direct invocation.", module_path)
        logger.info("SUCCESS: Step %s completed.", step_name)
    except Exception as e:
        logger.error("FAILED: Step %s raised exception: %s", step_name, e, exc_info=True)
        raise


def main():
    step_choices = [s[0] for s in STEPS] + list(STEP_ALIASES.keys())
    parser = argparse.ArgumentParser(description="Run the NSE ML project pipeline.")
    parser.add_argument(
        "--step",
        choices=step_choices,
        help="Run a specific pipeline step instead of the full pipeline.",
    )
    args = parser.parse_args()

    if args.step:
        target_step = STEP_ALIASES.get(args.step, args.step)
        for name, mod, desc in STEPS:
            if name == target_step:
                run_step(name, mod, desc)
                break
    else:
        logger.info("Starting full pipeline execution (%d steps)...", len(STEPS))
        for name, mod, desc in STEPS:
            run_step(name, mod, desc)
        logger.info("All pipeline steps completed successfully!")


if __name__ == "__main__":
    main()
