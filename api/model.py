"""
Model loading, feature-vector assembly, and prediction. Kept separate from
main.py so the FastAPI route handlers stay thin and this logic is testable
without spinning up the app.

Loads the TEMPORALLY-trained HistGradientBoostingClassifier
(class_weight='balanced') saved by scripts/05_train_and_evaluate.py — not
the held-out-projects model. That's the model Stage 2's SHAP analysis was
run on, and the one whose behavior is documented; using it here keeps the
served predictions consistent with what's been analyzed and reported on.

Layer 1b adds the cold-start branch to `predict()`. The rule itself lives
in api/coldstart.py (dependency-free, so it's testable and reusable by the
Layer 3 Action) and is documented canonically in api/schema.py. Nothing in
this file imputes a missing value — that's deliberate and load-bearing.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap

from .coldstart import (
    COLD_START_MESSAGE,
    COLD_START_UPGRADE_PATH,
    CONFIDENCE_LOW,
    CONFIDENCE_NORMAL,
    CONTRIBUTORS_SCOPE_ALL,
    CONTRIBUTORS_SCOPE_NON_NULL,
    PROJECT_HISTORY_FEATURES,
    STATUS_COLD_START,
    STATUS_OK,
    cold_start_reason,
    cold_start_trigger_features,
    is_cold_start,
    null_history_features,
)
from .schema import KNOWN_LANGUAGES, NAN_ALLOWED_FEATURES, BuildFeatures

logger = logging.getLogger("autodeploy_ai.model")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "hgb_split_temporal.joblib"

# Risk tier thresholds — tunable. Not derived from any formal cost-benefit
# analysis; picked as round, defensible cutpoints. Change here if the
# consuming workflow needs a different appetite for false positives.
RISK_TIER_LOW_MAX = 0.3
RISK_TIER_HIGH_MIN = 0.6

TOP_K_CONTRIBUTORS = 5


class ModelNotLoadedError(RuntimeError):
    pass


class PredictionService:
    """Loads (or fails loudly about) the trained model, and turns a
    BuildFeatures payload into a prediction + SHAP explanation."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.feature_cols: Optional[list] = None
        self.explainer = None
        self.load_error: Optional[str] = None
        # Fallback used only while no model is loaded (e.g. /health before
        # training has run); overwritten with the real, model-derived set
        # the moment a model loads successfully.
        self.known_languages = sorted(KNOWN_LANGUAGES)
        self._load()

    def _load(self):
        if not self.model_path.exists():
            self.load_error = (
                f"Model artifact not found at {self.model_path}. "
                "Run Stage 2 training first: `python scripts/05_train_and_evaluate.py` "
                "(from the project root, with the venv activated). This trains and saves "
                "the temporal-split HistGradientBoostingClassifier this API serves."
            )
            logger.error("MODEL NOT LOADED — %s", self.load_error)
            return
        try:
            bundle = joblib.load(self.model_path)
            self.model = bundle["model"]
            self.feature_cols = bundle["feature_cols"]
            self.explainer = shap.TreeExplainer(self.model)
            # Derived from the actual trained columns, not the hardcoded
            # KNOWN_LANGUAGES guess in schema.py — this is what protects
            # against exactly the bug that guess had (claiming 'python' was
            # a recognized language when the final training set had zero
            # python rows and no language_python column at all).
            self.known_languages = sorted(
                c[len("language_"):] for c in self.feature_cols if c.startswith("language_")
            )
            logger.info("Loaded model from %s (%d features, languages=%s)",
                        self.model_path, len(self.feature_cols), self.known_languages)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any load failure must be surfaced clearly
            self.load_error = f"Failed to load model artifact at {self.model_path}: {exc!r}"
            logger.error("MODEL NOT LOADED — %s", self.load_error)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def feature_schema(self) -> dict:
        schema = {}
        for name, field in BuildFeatures.model_fields.items():
            schema[name] = {
                "required": field.is_required(),  # False exactly for the 5 NAN_ALLOWED_FEATURES
                "nan_allowed": name in NAN_ALLOWED_FEATURES,
                # The distinction Layer 1b adds: of the 5 nullable features,
                # only the 2 project-history ones withhold the risk tier.
                "cold_start_trigger": name in PROJECT_HISTORY_FEATURES,
                "type": "int" if field.annotation is int else "string" if field.annotation is str else "float",
                "description": field.description,
            }
        # Ground truth for "language", derived from the loaded model itself
        # rather than the static KNOWN_LANGUAGES guess — see the __init__
        # comment on self.known_languages for why that distinction matters.
        schema["language"]["known_values"] = self.known_languages
        return schema

    def cold_start_behavior(self) -> dict:
        """The cold-start contract, served from /health so a caller can
        discover it without reading the source."""
        return {
            "rule": (
                "cold_start if and only if at least one PROJECT-history feature is "
                "absent (key omitted, null, or NaN): "
                + " or ".join(PROJECT_HISTORY_FEATURES)
                + ". Absence of author-history features alone does NOT trigger it."
            ),
            "trigger_features": sorted(PROJECT_HISTORY_FEATURES),
            "non_trigger_nullable_features": sorted(NAN_ALLOWED_FEATURES - set(PROJECT_HISTORY_FEATURES)),
            "zero_is_not_absent": (
                "previous_build_status=0 means 'the previous build passed' — a real "
                "value and a negative risk signal. Only null/omitted/NaN is absence."
            ),
            "states": {
                STATUS_OK: (
                    "risk_tier is Low/Medium/High, probability_confidence='normal', "
                    "contributors ranked over all 31 features."
                ),
                STATUS_COLD_START: (
                    "risk_tier is null — no tier is issued. failure_probability is "
                    "still returned with probability_confidence='low' and a "
                    "plain-language message. Contributors are ranked over non-null "
                    "features only. HTTP status is still 200."
                ),
            },
            "why": (
                "Stage 2 SHAP: previous_build_status + project_prior_failure_rate + "
                "consecutive_failure_streak carry 65.8% of total attribution. When "
                "project history is null that block is gone, and the model does not "
                "treat the absence neutrally — in training null meant 'first build of "
                "a young unstable project', so it learned null as a failure signal "
                "(+1.35 SHAP). A synthetic brand-new-repo vector scored 0.706 (High) "
                "before the repo had done anything wrong. At serve time null also "
                "means 'established repo, tool just installed' and the model cannot "
                "tell the two apart."
            ),
            "imputation": (
                "None. Nulls are passed to the model as NaN exactly as in training. "
                "Imputing a neutral prior or base rate would hide the uncertainty "
                "rather than surface it."
            ),
            "upgrade_path": COLD_START_UPGRADE_PATH,
        }

    def _row_from_features(self, features: BuildFeatures) -> pd.DataFrame:
        data = features.model_dump()
        language = data.pop("language").strip().lower()
        for lang in self.known_languages:
            data[f"language_{lang}"] = 1 if language == lang else 0
        # None -> NaN for the cold-start-allowed fields; pydantic already
        # guarantees every other key is present and non-null.
        for k, v in data.items():
            if v is None:
                data[k] = np.nan
        row = pd.DataFrame([data])
        # Reindex to the exact training column order — this is what makes
        # "same names, same order handled internally" true regardless of
        # JSON key order or dict iteration order.
        missing = set(self.feature_cols) - set(row.columns)
        if missing:
            # Should only ever happen for an unrecognized language producing
            # none of the 4 known dummy columns being touched above, which
            # already sets all 4 to 0 - so this is a defensive check, not an
            # expected path.
            raise ModelNotLoadedError(f"Internal schema mismatch, missing columns: {missing}")
        return row[self.feature_cols]

    def classify_risk(self, probability: float) -> str:
        if probability < RISK_TIER_LOW_MAX:
            return "Low"
        if probability < RISK_TIER_HIGH_MIN:
            return "Medium"
        return "High"

    def _top_contributors(self, row: pd.DataFrame, exclude_null: bool) -> list:
        """SHAP-rank this request's features.

        `exclude_null=True` drops any column that is NaN before ranking. On a
        cold-start vector the null history features carry the largest SHAP
        magnitudes in the whole row (`previous_build_status=null` alone was
        +1.35 on the Layer 1 test vector), but that magnitude describes an
        *absence* — "this repo has no history" — not a property of the commit
        being scored. Listing it as the top "driver" of a prediction we are
        simultaneously declining to turn into a risk tier would be actively
        misleading. So on cold_start we rank only the features that have
        values, which surfaces the change-size and project-context signal
        that genuinely does apply.
        """
        shap_row = self.explainer.shap_values(row)[0]

        candidates = range(len(self.feature_cols))
        if exclude_null:
            candidates = [i for i in candidates if not pd.isna(row.iloc[0, i])]

        order = sorted(candidates, key=lambda i: -abs(shap_row[i]))[:TOP_K_CONTRIBUTORS]
        contributors = []
        for i in order:
            fval = row.iloc[0, i]
            contributors.append({
                "feature": self.feature_cols[i],
                "feature_value": None if pd.isna(fval) else float(fval),
                "shap_value": float(shap_row[i]),
                "direction": "increases risk" if shap_row[i] > 0 else "decreases risk",
            })
        return contributors

    def predict(self, features: BuildFeatures) -> dict:
        """Score one build.

        Returns either a normal tiered prediction or the cold-start state.
        The branch is decided by the rule in api/coldstart.py, documented
        canonically in api/schema.py. Note what does NOT happen here: no null
        is filled in, before or after the branch. The model sees the same
        NaNs the caller sent, exactly as in training.
        """
        if not self.is_loaded:
            raise ModelNotLoadedError(self.load_error or "Model not loaded.")

        payload = features.model_dump()
        row = self._row_from_features(features)
        probability = float(self.model.predict_proba(row)[0, 1])

        common = {
            "failure_probability": probability,
            "null_history_features": null_history_features(payload),
        }

        if is_cold_start(payload):
            return {
                **common,
                "status": STATUS_COLD_START,
                "probability_confidence": CONFIDENCE_LOW,
                # Withheld on purpose. This is the whole feature.
                "risk_tier": None,
                "risk_tier_thresholds": None,
                "message": COLD_START_MESSAGE,
                "cold_start": {
                    "reason": cold_start_reason(payload),
                    "triggered_by": cold_start_trigger_features(payload),
                    "history_scoring_active": False,
                    "upgrade_path": COLD_START_UPGRADE_PATH,
                },
                "top_contributors": self._top_contributors(row, exclude_null=True),
                "contributors_scope": CONTRIBUTORS_SCOPE_NON_NULL,
            }

        return {
            **common,
            "status": STATUS_OK,
            "probability_confidence": CONFIDENCE_NORMAL,
            "risk_tier": self.classify_risk(probability),
            "risk_tier_thresholds": {"low_max": RISK_TIER_LOW_MAX, "high_min": RISK_TIER_HIGH_MIN},
            "message": None,
            "cold_start": None,
            "top_contributors": self._top_contributors(row, exclude_null=False),
            "contributors_scope": CONTRIBUTORS_SCOPE_ALL,
        }
