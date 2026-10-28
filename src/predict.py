"""
Clinical Inference Service, Feature Range Validation, and Risk Classification Handler.
"""

import logging
from typing import Dict, Any, Union, Optional
import pandas as pd
import numpy as np

from src.config import (
    FEATURE_NAMES,
    FEATURE_METADATA,
    DEFAULT_PIPELINE_PATH,
    DEFAULT_CALIBRATED_PIPELINE_PATH
)
from src.model import load_production_pipeline, load_calibrated_pipeline
from src.explain import HeartDiseaseExplainer
from src.utils import analyze_risk_factors

logger = logging.getLogger(__name__)

# Global singletons for fast inference
_CACHED_PIPELINE = None
_CACHED_CALIBRATED = None
_CACHED_EXPLAINER = None


def get_inference_pipeline():
    """Retrieve or initialize the production inference pipeline."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        _CACHED_PIPELINE = load_production_pipeline(DEFAULT_PIPELINE_PATH)
    return _CACHED_PIPELINE


def get_calibrated_pipeline():
    """Retrieve or initialize the calibrated inference pipeline."""
    global _CACHED_CALIBRATED
    if _CACHED_CALIBRATED is None:
        _CACHED_CALIBRATED = load_calibrated_pipeline(DEFAULT_CALIBRATED_PIPELINE_PATH)
    return _CACHED_CALIBRATED


def get_explainer():
    """Retrieve or initialize the SHAP explainer."""
    global _CACHED_EXPLAINER
    if _CACHED_EXPLAINER is None:
        pipe = get_inference_pipeline()
        _CACHED_EXPLAINER = HeartDiseaseExplainer(pipe)
    return _CACHED_EXPLAINER


def validate_patient_features(data: Dict[str, Any]) -> Dict[str, Union[int, float]]:
    """
    Validate incoming clinical payload against medical ranges and categories.
    Raises ValueError with descriptive clinical details if input is malformed.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Input payload must be a JSON dictionary, got {type(data).__name__}")

    clean = {}
    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        raise ValueError(f"Incomplete clinical profile. Missing required features: {', '.join(missing)}")

    for f in FEATURE_NAMES:
        raw_val = data[f]
        if raw_val is None or str(raw_val).strip() == "":
            raise ValueError(f"Feature '{f}' cannot be null or empty.")

        try:
            num_val = float(raw_val)
        except (ValueError, TypeError):
            raise ValueError(f"Feature '{f}' must be numeric, received: '{raw_val}'")

        meta = FEATURE_METADATA[f]
        # Range checks
        min_val = meta.get("min")
        max_val = meta.get("max")
        if min_val is not None and num_val < min_val:
            raise ValueError(f"{meta['label']} ({num_val}) is below valid clinical minimum ({min_val} {meta.get('unit', '')}).")
        if max_val is not None and num_val > max_val:
            raise ValueError(f"{meta['label']} ({num_val}) exceeds valid clinical maximum ({max_val} {meta.get('unit', '')}).")

        # Categorical choices check
        choices = meta.get("choices")
        if choices is not None:
            int_val = int(round(num_val))
            if int_val not in choices and int_val not in [c for c in range(min_val or 0, (max_val or 3) + 1)]:
                valid_keys = list(choices.keys())
                raise ValueError(f"{meta['label']} must be one of {valid_keys}, received: {num_val}")
            clean[f] = int_val
        elif meta["type"] == int:
            clean[f] = int(round(num_val))
        else:
            clean[f] = float(num_val)

    return clean


def predict_patient_risk(
    data: Dict[str, Any],
    include_explanation: bool = False
) -> Dict[str, Any]:
    """
    Executes clinical prediction pipeline, applies probability calibration,
    evaluates AHA heuristic flags, and optionally computes SHAP attributions.
    """
    clean_data = validate_patient_features(data)
    df_patient = pd.DataFrame([clean_data])[FEATURE_NAMES]

    prod_pipe = get_inference_pipeline()
    calib_pipe = get_calibrated_pipeline()

    # Raw model prediction
    raw_pred = int(prod_pipe.predict(df_patient)[0])
    raw_prob = float(prod_pipe.predict_proba(df_patient)[0][1])

    # Calibrated probability
    try:
        calib_prob = float(calib_pipe.predict_proba(df_patient)[0][1])
    except Exception as e:
        logger.warning(f"Calibrated prediction fallback: {e}")
        calib_prob = raw_prob

    # Risk tier categorization
    if calib_prob >= 0.70:
        risk_level = "High Risk"
        risk_badge = "HIGH CARDIAC RISK"
    elif calib_prob >= 0.45:
        risk_level = "Moderate Risk"
        risk_badge = "ELEVATED RISK"
    elif calib_prob >= 0.20:
        risk_level = "Mild Risk"
        risk_badge = "LOW-MODERATE RISK"
    else:
        risk_level = "Low Risk"
        risk_badge = "MINIMAL RISK"

    # Clinical heuristic evaluation (AHA thresholds)
    risk_factors = analyze_risk_factors(clean_data)

    result = {
        "prediction": 1 if calib_prob >= 0.50 else 0,
        "label": "Elevated Risk (Heart Disease Indicated)" if calib_prob >= 0.50 else "Low Risk (Healthy Baseline)",
        "calibrated_probability": round(calib_prob, 4),
        "raw_probability": round(raw_prob, 4),
        "confidence_percentage": round(calib_prob * 100, 2),
        "risk_level": risk_level,
        "risk_badge": risk_badge,
        "risk_factors": risk_factors,
        "patient_data": clean_data
    }

    if include_explanation:
        explainer = get_explainer()
        result["explanation"] = explainer.explain_instance(clean_data)

    return result
