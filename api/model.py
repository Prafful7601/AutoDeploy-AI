"""
Model loading, feature-vector assembly, and prediction. Kept separate from
main.py so the FastAPI route handlers stay thin and this logic is testable
without spinning up the app.

Loads the TEMPORALLY-trained HistGradientBoostingClassifier
(class_weight='balanced') saved by scripts/05_train_and_evaluate.py — not
the held-out-projects model. That's the model Stage 2's SHAP analysis was
run on, and the one whose behavior is documented; using it here keeps the
served predictions consistent with what's been analyzed and reported on.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap

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
            logger.info("Loaded model from %s (%d features)", self.model_path, len(self.feature_cols))
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
                "type": "int" if field.annotation is int else "string" if field.annotation is str else "float",
            }
        return schema

    def _row_from_features(self, features: BuildFeatures) -> pd.DataFrame:
        data = features.model_dump()
        language = data.pop("language").strip().lower()
        for lang in KNOWN_LANGUAGES:
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

    def predict(self, features: BuildFeatures) -> dict:
        if not self.is_loaded:
            raise ModelNotLoadedError(self.load_error or "Model not loaded.")

        row = self._row_from_features(features)
        probability = float(self.model.predict_proba(row)[0, 1])
        tier = self.classify_risk(probability)

        shap_row = self.explainer.shap_values(row)[0]
        order = np.argsort(-np.abs(shap_row))[:TOP_K_CONTRIBUTORS]
        contributors = []
        for i in order:
            fname = self.feature_cols[i]
            fval = row.iloc[0, i]
            contributors.append({
                "feature": fname,
                "feature_value": None if pd.isna(fval) else float(fval),
                "shap_value": float(shap_row[i]),
                "direction": "increases risk" if shap_row[i] > 0 else "decreases risk",
            })

        return {
            "failure_probability": probability,
            "risk_tier": tier,
            "risk_tier_thresholds": {"low_max": RISK_TIER_LOW_MAX, "high_min": RISK_TIER_HIGH_MIN},
            "top_contributors": contributors,
        }
