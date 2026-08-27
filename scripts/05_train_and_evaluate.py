"""
Stage 2, step 2b: train HistGradientBoostingClassifier and build the full
model-vs-baselines comparison table.

Design, per explicit instructions this round:

  - No SMOTE, no resampling. Class weights are the only imbalance handling
    considered, and only kept if they actually help.
  - The two splits are evaluated completely independently: a model fit on
    the temporal split's training data is never touched by the held-out-
    projects split, and vice versa. Two separate models, two separate
    fits.
  - No hyperparameter tuning against either test set. The one choice this
    script makes from data — class_weight=None vs class_weight='balanced'
    — is selected on a validation slice carved from the TRAINING portion
    only: the most recent 10% of that training portion by build time,
    never the real test set. Nothing else about the model (learning rate,
    tree depth, number of iterations, etc.) is tuned; scikit-learn's
    defaults are used as-is (with a fixed random_state), on the view that
    tuning those without a proper nested validation loop, on a two-week
    timeline, would risk more overfitting-to-us than it would buy in
    accuracy. Flagged here rather than silently doing a grid search.
  - Precision/recall/F1 use the default 0.5 probability threshold — not
    tuned. PR-AUC/ROC-AUC (the metrics that matter most per instructions)
    are threshold-independent and unaffected by this choice.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, SEED  # noqa: E402

IN_PATH = PROCESSED_DATA_DIR / "model_features.parquet"
REPORT_PATH = REPORTS_DIR / "stage2_model_results.md"

VALIDATION_FRACTION = 0.10  # last 10% of the training portion, by time, carved for class-weight selection only

SPLITS = [
    ("split_temporal", "Temporal", "train < 2015-11-01, test >= 2015-11-01"),
    ("split_project_holdout", "Held-out-projects", "49/243 projects never in training"),
]


def load_data():
    df = pd.read_parquet(IN_PATH)
    # One-hot encode the one categorical feature (4 values: ruby/python/java/go).
    # Everything else is already numeric/boolean/int.
    lang_dummies = pd.get_dummies(df["language"], prefix="language")
    df = pd.concat([df, lang_dummies], axis=1)
    return df, lang_dummies.columns.tolist()


def make_baseline_predictions(test: pd.DataFrame):
    majority = np.zeros(len(test), dtype=int)
    prev_status = test["previous_build_status"].fillna(0).to_numpy().astype(int)
    return {"Majority class (always pass)": majority,
            "Previous build failed -> predict fail": prev_status}


def compute_metrics(y_true, y_score_or_pred, is_probability: bool) -> dict:
    if is_probability:
        y_pred = (y_score_or_pred >= 0.5).astype(int)
        roc = roc_auc_score(y_true, y_score_or_pred)
        pr = average_precision_score(y_true, y_score_or_pred)
    else:
        y_pred = y_score_or_pred
        if y_pred.std() == 0:
            roc, pr = float("nan"), float("nan")
        else:
            roc = roc_auc_score(y_true, y_pred)
            pr = average_precision_score(y_true, y_pred)
    return {
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc,
        "pr_auc": pr,
    }


def select_class_weight(train_fit, val, feature_cols):
    """Pick class_weight in {None, 'balanced'} using PR-AUC on a temporal
    validation slice carved from the training portion. Returns the chosen
    weight plus both validation scores for full transparency."""
    scores = {}
    for cw in [None, "balanced"]:
        model = HistGradientBoostingClassifier(random_state=SEED, class_weight=cw)
        model.fit(train_fit[feature_cols], train_fit["failed"])
        val_proba = model.predict_proba(val[feature_cols])[:, 1]
        scores[cw] = average_precision_score(val["failed"], val_proba)
    chosen = max(scores, key=scores.get)
    return chosen, scores


def run_split(df, split_col, feature_cols):
    train_full = df[df[split_col] == "train"].sort_values("build_ts")
    test = df[df[split_col] == "test"].sort_values("build_ts")

    n_val = int(len(train_full) * VALIDATION_FRACTION)
    train_fit, val = train_full.iloc[:-n_val], train_full.iloc[-n_val:]

    chosen_cw, val_scores = select_class_weight(train_fit, val, feature_cols)

    # Refit on the FULL training portion (train_fit + val) with the chosen
    # class-weight setting, then evaluate once on the real test set.
    final_model = HistGradientBoostingClassifier(random_state=SEED, class_weight=chosen_cw)
    final_model.fit(train_full[feature_cols], train_full["failed"])
    test_proba = final_model.predict_proba(test[feature_cols])[:, 1]

    model_metrics = compute_metrics(test["failed"], test_proba, is_probability=True)

    baseline_preds = make_baseline_predictions(test)
    baseline_metrics = {
        name: compute_metrics(test["failed"], pred, is_probability=False)
        for name, pred in baseline_preds.items()
    }

    return {
        "chosen_class_weight": chosen_cw,
        "val_pr_auc_scores": val_scores,
        "n_train": len(train_full),
        "n_val_carved": len(val),
        "n_test": len(test),
        "n_fail_test": int(test["failed"].sum()),
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "model": final_model,
    }


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) and not np.isnan(x) else ("n/a" if isinstance(x, float) else x)


def write_report(results, feature_cols):
    lines = ["# Stage 2 model results: HistGradientBoostingClassifier vs. both baselines", "",
              f"Features used ({len(feature_cols)} columns, `language` one-hot encoded from 1 categorical "
              "into 4 binary columns; everything else as listed in stage2_feature_list.md).",
              "", "No SMOTE, no resampling. No hyperparameter tuning against either test set — "
              "the only data-driven choice is class_weight (None vs 'balanced'), selected on a "
              "temporal validation slice (last 10% of each split's training portion by build time), "
              "never on test data. All other HistGradientBoostingClassifier settings are "
              "scikit-learn defaults with a fixed random_state.", ""]

    for split_col, split_name, split_desc in [(s[0], s[1], s[2]) for s in SPLITS]:
        r = results[split_col]
        lines.append(f"## {split_name} split ({split_desc})")
        lines.append("")
        margin = abs(r["val_pr_auc_scores"]["balanced"] - r["val_pr_auc_scores"][None])
        margin_note = (" — a negligible margin, essentially a tie; class weighting made no "
                        "real difference here" if margin < 0.001 else "")
        lines.append(f"**Class weight selection** (validation PR-AUC, {r['n_val_carved']:,} rows carved "
                      f"from the end of the {r['n_train']:,}-row training portion): "
                      f"`None` -> {r['val_pr_auc_scores'][None]:.4f}, "
                      f"`'balanced'` -> {r['val_pr_auc_scores']['balanced']:.4f} "
                      f"(margin {margin:.4f}{margin_note}). "
                      f"**Chosen: `class_weight={r['chosen_class_weight']}`** "
                      f"(final model refit on the full {r['n_train']:,}-row training portion with this setting).")
        lines.append("")
        lines.append(f"Test set: {r['n_test']:,} builds, {r['n_fail_test']:,} failed.")
        lines.append("")
        lines.append("| Method | Precision (fail) | Recall (fail) | F1 (fail) | ROC-AUC | PR-AUC |")
        lines.append("|---|---|---|---|---|---|")
        for name, m in r["baseline_metrics"].items():
            lines.append(f"| {name} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | "
                          f"{fmt(m['roc_auc'])} | {fmt(m['pr_auc'])} |")
        m = r["model_metrics"]
        lines.append(f"| **HistGradientBoostingClassifier (class_weight={r['chosen_class_weight']})** | "
                      f"**{fmt(m['precision'])}** | **{fmt(m['recall'])}** | **{fmt(m['f1'])}** | "
                      f"**{fmt(m['roc_auc'])}** | **{fmt(m['pr_auc'])}** |")
        lines.append("")

        prev_pr = r["baseline_metrics"]["Previous build failed -> predict fail"]["pr_auc"]
        model_pr = m["pr_auc"]
        delta = model_pr - prev_pr
        beats = "beats" if delta > 0 else "does NOT beat"
        lines.append(f"**Model {beats} the previous-build baseline on PR-AUC**: "
                      f"{model_pr:.3f} vs {prev_pr:.3f} ({delta:+.3f}).")
        lines.append("")

    # Temporal-vs-held-out gap, for both the baseline and the model
    prev_pr_temporal = results["split_temporal"]["baseline_metrics"]["Previous build failed -> predict fail"]["pr_auc"]
    prev_pr_holdout = results["split_project_holdout"]["baseline_metrics"]["Previous build failed -> predict fail"]["pr_auc"]
    model_pr_temporal = results["split_temporal"]["model_metrics"]["pr_auc"]
    model_pr_holdout = results["split_project_holdout"]["model_metrics"]["pr_auc"]
    baseline_gap = prev_pr_temporal - prev_pr_holdout
    model_gap = model_pr_temporal - model_pr_holdout

    lines += ["## Temporal vs. held-out-projects gap: baseline vs. model", "",
              f"- Previous-build baseline PR-AUC: {prev_pr_temporal:.3f} (temporal) -> "
              f"{prev_pr_holdout:.3f} (held-out) — gap of {baseline_gap:+.3f}.",
              f"- Model PR-AUC: {model_pr_temporal:.3f} (temporal) -> {model_pr_holdout:.3f} "
              f"(held-out) — gap of {model_gap:+.3f}.",
              ""]
    if abs(model_gap) > abs(baseline_gap):
        lines.append(f"**The model's split gap is WIDER than the baseline's** "
                      f"({model_gap:+.3f} vs {baseline_gap:+.3f}) — it relies more heavily on "
                      "project-specific patterns that don't transfer to unseen projects than "
                      "the simple rule does.")
    else:
        lines.append(f"**The model's split gap is NARROWER than or equal to the baseline's** "
                      f"({model_gap:+.3f} vs {baseline_gap:+.3f}) — it generalizes to unseen "
                      "projects at least as well as the simple rule does.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    df, lang_cols = load_data()

    base_feature_cols = [
        "src_churn", "files_added", "files_deleted", "files_modified", "total_files_changed",
        "src_files_changed", "doc_files_changed", "other_files_changed", "tests_added",
        "tests_deleted", "test_file_ratio", "num_commits_in_build", "commits_on_touched_files",
        "previous_build_status", "project_prior_failure_rate", "project_prior_build_count",
        "consecutive_failure_streak", "author_prior_builds_in_project",
        "author_prior_failure_rate_in_project", "author_days_since_last_build_in_project",
        "team_size", "repo_age_days", "repo_num_commits", "sloc", "test_lines_per_kloc",
        "test_cases_per_kloc", "asserts_per_kloc", "is_pr", "by_core_team_member",
        "is_main_branch",
    ]
    feature_cols = base_feature_cols + lang_cols

    results = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for split_col, split_name, _ in SPLITS:
        r = run_split(df, split_col, feature_cols)
        results[split_col] = r
        model_path = MODELS_DIR / f"hgb_{split_col}.joblib"
        joblib.dump({"model": r["model"], "feature_cols": feature_cols}, model_path)
        print(f"[{split_name}] class_weight={r['chosen_class_weight']} -> saved {model_path}")

    write_report(results, feature_cols)


if __name__ == "__main__":
    main()
