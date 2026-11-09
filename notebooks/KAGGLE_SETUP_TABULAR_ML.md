# Tabular ML Models — Kaggle Setup Instructions

## What to Upload

Create a **Kaggle Dataset** with the following folder structure:

```
nse-ml-project/
├── processed/
│   ├── features_targets.csv      ← data/processed/features_targets.csv
│   ├── feature_list.txt          ← data/processed/feature_list.txt
│   └── validation_split_summary.csv
├── configs/
│   ├── experiment.yaml           ← configs/experiment.yaml
│   └── models.yaml               ← configs/models.yaml
```

## Step-by-Step

1. **Go to** https://www.kaggle.com/datasets → **New Dataset**
2. **Name it** `nse-ml-project`
3. **Upload** the files from the table above (drag & drop)
4. **Make it Private** → Save

## Run the Notebook

1. **Go to** https://www.kaggle.com/code → **New Notebook**
2. **Upload** `notebooks/01_tabular_ml_training.ipynb` (File → Import Notebook)
3. **Add data**: on the right panel → Add Data → search `nse-ml-project` → Add
4. **Settings** → Accelerator: **None** (CPU is fine for these models)
5. **Run All** — expected runtime: ~10-15 minutes
6. **Session options** → Persist Session to save output

## Download Results

After completion:
1. Open the **Output** tab in the notebook
2. Download the output artifacts zip
3. Extract it to: `artifacts/kaggle/tabular_ml_models/`

## Files You'll Get

| File | Contents |
|------|---------|
| `results_all_splits.csv` | Metrics for every model × split × horizon |
| `predictions_all_splits.csv` | Row-level predictions (needed for Phase 7 portfolio backtest) |
| `accuracy_pivot.csv` | Mean accuracy by model × horizon |
| `auc_pivot.csv` | Mean AUC-ROC by model × horizon |

## After Downloading

Share the key metrics (accuracy pivot table or paste the checkpoint summary) and we'll:
- Update RESULTS.md with actual numbers
- Proceed to Phase 6 (LSTM) or Phase 7 (backtest) depending on results
