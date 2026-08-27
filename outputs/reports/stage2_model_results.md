# Stage 2 model results: HistGradientBoostingClassifier vs. both baselines

Features used (33 columns, `language` one-hot encoded from 1 categorical into 4 binary columns; everything else as listed in stage2_feature_list.md).

No SMOTE, no resampling. No hyperparameter tuning against either test set — the only data-driven choice is class_weight (None vs 'balanced'), selected on a temporal validation slice (last 10% of each split's training portion by build time), never on test data. All other HistGradientBoostingClassifier settings are scikit-learn defaults with a fixed random_state.

## Temporal split (train < 2015-11-01, test >= 2015-11-01)

**Class weight selection** (validation PR-AUC, 20,716 rows carved from the end of the 207,165-row training portion): `None` -> 0.8068, `'balanced'` -> 0.8070 (margin 0.0002 — a negligible margin, essentially a tie; class weighting made no real difference here). **Chosen: `class_weight=balanced`** (final model refit on the full 207,165-row training portion with this setting).

Test set: 53,974 builds, 14,052 failed.

| Method | Precision (fail) | Recall (fail) | F1 (fail) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Majority class (always pass) | 0.000 | 0.000 | 0.000 | n/a | n/a |
| Previous build failed -> predict fail | 0.697 | 0.697 | 0.697 | 0.795 | 0.565 |
| **HistGradientBoostingClassifier (class_weight=balanced)** | **0.611** | **0.755** | **0.675** | **0.876** | **0.804** |

**Model beats the previous-build baseline on PR-AUC**: 0.804 vs 0.565 (+0.239).

## Held-out-projects split (49/243 projects never in training)

**Class weight selection** (validation PR-AUC, 20,110 rows carved from the end of the 201,107-row training portion): `None` -> 0.7977, `'balanced'` -> 0.7997 (margin 0.0020). **Chosen: `class_weight=balanced`** (final model refit on the full 201,107-row training portion with this setting).

Test set: 60,032 builds, 15,137 failed.

| Method | Precision (fail) | Recall (fail) | F1 (fail) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Majority class (always pass) | 0.000 | 0.000 | 0.000 | n/a | n/a |
| Previous build failed -> predict fail | 0.608 | 0.608 | 0.608 | 0.738 | 0.468 |
| **HistGradientBoostingClassifier (class_weight=balanced)** | **0.545** | **0.676** | **0.603** | **0.830** | **0.690** |

**Model beats the previous-build baseline on PR-AUC**: 0.690 vs 0.468 (+0.221).

## Temporal vs. held-out-projects gap: baseline vs. model

- Previous-build baseline PR-AUC: 0.565 (temporal) -> 0.468 (held-out) — gap of +0.096.
- Model PR-AUC: 0.804 (temporal) -> 0.690 (held-out) — gap of +0.114.

**The model's split gap is WIDER than the baseline's** (+0.114 vs +0.096) — it relies more heavily on project-specific patterns that don't transfer to unseen projects than the simple rule does.