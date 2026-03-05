# Treaty Internalization Capacity Model (LATAM)

- Observations: 237 (train=165, test=72)
- Target: annual improvement in SPAR legal capacity (`spar_delta > 0`)
- Selected model: **gbm**

## Performance
- random_forest: {'accuracy': 0.75, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.4754}
- gbm: {'accuracy': 0.8056, 'precision': 0.5, 'recall': 0.0714, 'f1': 0.125, 'roc_auc': 0.5234}

## Top SHAP-like factors (mean |contribution|)
- yrsoffc: 0.03679
- health_exp_gdp: 0.03604
- rl_est: 0.02505
- gov_seat_share: 0.02098
- uhc: 0.01999
- checks: 0.01413
- log_gdp_pc: 0.00634
- cc_est: 0.00560
- rq_est: 0.00249
- state_capacity_wgi: 0.00215

## External validation
- Tobacco implementation transfer AUC: 0.351
