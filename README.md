# ML Internalization Model (LATAM)

This repository now includes an end-to-end, dependency-light pipeline to estimate treaty internalization capacity in LATAM using the datasets in `data/`.

## What the pipeline does

- Builds a panel from SPAR, political, V-Dem, WGI, GDP, health spending, and UHC data.
- Creates a dynamic outcome: annual change in SPAR legal capacity (`spar_delta`).
- Converts the outcome to classification target (`spar_delta > 0`).
- Trains two supervised models with 70/30 train-test split and 5-fold CV hyperparameter selection:
  - Random Forest
  - Gradient Boosting Machine (GBM)
- Reports: accuracy, precision, recall, F1, ROC-AUC.
- Computes SHAP-like feature contributions (Monte-Carlo Shapley approximation).
- Runs external validation against tobacco treaty implementation dynamics.

## Run

```bash
python treaty_model_pipeline.py
```

## Outputs

- `outputs/model_results.json`: machine-readable metrics, hyperparameters, top factors.
- `outputs/model_results.md`: human-readable summary.
