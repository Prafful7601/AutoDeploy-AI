"""
Stage 2, step 2a: baselines, alone, before any model exists.

Two baselines, evaluated on the test side of both splits from
data/processed/model_features.parquet:

  1. Majority class — always predict "pass" (0). The floor: anything that
     can't beat this isn't a classifier.
  2. Previous-build-failed rule — predict fail (1) iff the immediately
     preceding build in this project failed, else predict pass (0). This
     uses the `previous_build_status` column exactly as computed in
     scripts/03_build_features.py — same sort key (project, build start
     time, build id) and same tiebreak, because it IS that column, not a
     separately recomputed one. Builds with no previous build in their
     project (first build ever, ~0.1% of rows) default to predicting pass,
     matching the majority-class default.

No model is trained here. This is deliberately its own script/step so the
baseline numbers stand on their own before any comparison table exists.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR, REPORTS_DIR  # noqa: E402

IN_PATH = PROCESSED_DATA_DIR / "model_features.parquet"
REPORT_PATH = REPORTS_DIR / "stage2_baselines.md"

SPLITS = [
    ("split_temporal", "Temporal (train < 2015-11-01, test >= 2015-11-01)"),
    ("split_project_holdout", "Held-out-projects (49/243 projects never in training)"),
]


def evaluate(y_true: np.ndarray, y_score: np.ndarray, label: str) -> dict:
    """y_score is a hard 0/1 prediction for both baselines (neither produces
    a probability) — ROC-AUC/PR-AUC on a two-valued score are mathematically
    well-defined but coarse (a single decision threshold, not a curve).
    Flagged explicitly rather than silently presented as equivalent to a
    model's probabilistic AUC."""
    n_pos = int(y_true.sum())
    row = {"baseline": label, "n": len(y_true), "n_fail": n_pos}
    if y_score.std() == 0:
        # Constant prediction (majority class): no ranking capability at all.
        row["roc_auc"] = float("nan")
        row["pr_auc"] = float("nan")
        row["auc_note"] = "n/a — constant predictor, no ranking signal"
    else:
        row["roc_auc"] = roc_auc_score(y_true, y_score)
        row["pr_auc"] = average_precision_score(y_true, y_score)
        row["auc_note"] = "coarse — single-threshold hard rule, not a probability curve"
    row["precision"] = precision_score(y_true, y_score, pos_label=1, zero_division=0)
    row["recall"] = recall_score(y_true, y_score, pos_label=1, zero_division=0)
    row["f1"] = f1_score(y_true, y_score, pos_label=1, zero_division=0)
    return row


def main():
    df = pd.read_parquet(IN_PATH)
    results = []

    for split_col, split_desc in SPLITS:
        test = df[df[split_col] == "test"]
        y_true = test["failed"].to_numpy()

        majority_pred = np.zeros(len(test), dtype=int)
        results.append({"split": split_desc, **evaluate(y_true, majority_pred, "Majority class (always predict pass)")})

        prev_status = test["previous_build_status"]
        n_no_prior = int(prev_status.isna().sum())
        prev_pred = prev_status.fillna(0).to_numpy().astype(int)
        row = evaluate(y_true, prev_pred, "Previous build failed -> predict fail")
        row["split"] = split_desc
        row["n_no_prior_defaulted_to_pass"] = n_no_prior
        results.append(row)

    lines = ["# Stage 2 baselines (no model trained yet)", "",
             "Both baselines evaluated with the exact same test rows, sort order, and "
             "tiebreak as the feature table — `previous_build_status` here is the literal "
             "column from `data/processed/model_features.parquet`, not a recomputation.",
             ""]
    for split_col, split_desc in SPLITS:
        lines.append(f"## {split_desc}")
        lines.append("")
        lines.append("| Baseline | N (test) | N fail | Precision (fail) | Recall (fail) | F1 (fail) | ROC-AUC | PR-AUC | AUC note |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in results:
            if r["split"] != split_desc:
                continue
            roc = f"{r['roc_auc']:.3f}" if not np.isnan(r["roc_auc"]) else "n/a"
            pr = f"{r['pr_auc']:.3f}" if not np.isnan(r["pr_auc"]) else "n/a"
            extra = f" ({r['n_no_prior_defaulted_to_pass']} first-builds defaulted to pass)" if "n_no_prior_defaulted_to_pass" in r else ""
            lines.append(
                f"| {r['baseline']}{extra} | {r['n']:,} | {r['n_fail']:,} | "
                f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {roc} | {pr} | {r['auc_note']} |"
            )
        lines.append("")

    lines += ["## Notes", "",
              "- Majority class has no ROC-AUC/PR-AUC: a constant prediction carries no "
              "ranking information, so those metrics are undefined, not merely low. For "
              "reference, a *no-skill* (random-ranking) classifier's PR-AUC would equal "
              "the fail-class prevalence (~26-29% depending on split).",
              "- The previous-build-failed rule's ROC-AUC/PR-AUC come from a single hard "
              "threshold (it only ever outputs 0 or 1), so they're a coarse, single-point "
              "estimate, not a full probability-ranked curve like a real model produces.",
              ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
