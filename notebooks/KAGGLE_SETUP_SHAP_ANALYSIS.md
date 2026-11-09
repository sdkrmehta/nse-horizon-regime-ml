# Kaggle Setup — SHAP Feature Importance Analysis

## Notebook
`notebooks/03_shap_explainability_analysis.ipynb`

## What it does
- Re-trains XGBoost (frozen hyperparams from `models.yaml`) on each of 6 walk-forward splits × 2 horizons
- Computes SHAP TreeExplainer values on test data only
- Reports global + per-regime feature importance, and cross-split stability metrics

## Steps

### 1. Upload Notebook
Go to **Kaggle → Code → New Notebook** → **File → Import Notebook** → upload `03_shap_explainability_analysis.ipynb`

### 2. Add Dataset
Click **Add Data** → search for `surajdeepkumarmehta/nse-ml-project` → Add

### 3. Accelerator
Settings → Accelerator → **GPU T4 x2** recommended (XGBoost + SHAP faster on GPU)

> Note: XGBoost SHAP is CPU-bound — **None** is also fine, runtime ~10-15 min

### 4. Run All
Click **Run All** → wait for completion (~10-20 min)

### 5. Download & Extract
- Output tab → Download the artifacts zip
- Extract to: `artifacts/kaggle/shap_explainability_models/`

## Expected Outputs (inside ZIP)
| File | Description |
|------|-------------|
| `shap_importance_by_split.csv` | Mean \|SHAP\| per feature, per split × horizon |
| `shap_importance_by_regime.csv` | Mean \|SHAP\| per feature, per regime (Low/Med/High) |
| `shap_stability.csv` | Pairwise Spearman r + Kendall's W across splits |
| `feature_rankings.csv` | Mean rank + per-split importance for all 44 features |

## Research Question Answered
**RQ6**: Are important features stable across regimes and retraining windows?
