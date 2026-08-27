# Stage 2 baselines (no model trained yet)

Both baselines evaluated with the exact same test rows, sort order, and tiebreak as the feature table — `previous_build_status` here is the literal column from `data/processed/model_features.parquet`, not a recomputation.

## Temporal (train < 2015-11-01, test >= 2015-11-01)

| Baseline | N (test) | N fail | Precision (fail) | Recall (fail) | F1 (fail) | ROC-AUC | PR-AUC | AUC note |
|---|---|---|---|---|---|---|---|---|
| Majority class (always predict pass) | 53,974 | 14,052 | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a — constant predictor, no ranking signal |
| Previous build failed -> predict fail (0 first-builds defaulted to pass) | 53,974 | 14,052 | 0.697 | 0.697 | 0.697 | 0.795 | 0.565 | coarse — single-threshold hard rule, not a probability curve |

## Held-out-projects (49/243 projects never in training)

| Baseline | N (test) | N fail | Precision (fail) | Recall (fail) | F1 (fail) | ROC-AUC | PR-AUC | AUC note |
|---|---|---|---|---|---|---|---|---|
| Majority class (always predict pass) | 60,032 | 15,137 | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a — constant predictor, no ranking signal |
| Previous build failed -> predict fail (49 first-builds defaulted to pass) | 60,032 | 15,137 | 0.608 | 0.608 | 0.608 | 0.738 | 0.468 | coarse — single-threshold hard rule, not a probability curve |

## Notes

- Majority class has no ROC-AUC/PR-AUC: a constant prediction carries no ranking information, so those metrics are undefined, not merely low. For reference, a *no-skill* (random-ranking) classifier's PR-AUC would equal the fail-class prevalence (~26-29% depending on split).
- The previous-build-failed rule's ROC-AUC/PR-AUC come from a single hard threshold (it only ever outputs 0 or 1), so they're a coarse, single-point estimate, not a full probability-ranked curve like a real model produces.