"""
Inference engine and clinical prediction handler for Heart Disease classification.
"""

from typing import Dict, Any, Union
import pandas as pd
import numpy as np

from src.utils import FEATURE_NAMES, analyze_risk_factors
from src.model import load_trained_pipeline


def validate_patient_input(data: Dict[str, Any]) -> Dict[str, Union[int, float]]:
    """
    Validate and cast incoming clinical feature dictionary.
    Ensures all 13 required features exist and are within sane medical bounds.
    """
    clean_data = {}
    for feature in FEATURE_NAMES:
        if feature not in data:
            raise ValueError(f"Missing required clinical feature: '{feature}'")
        try:
            val = float(data[feature])
        except (ValueError, TypeError):
            raise ValueError(f"Feature '{feature}' must be numeric, got {data[feature]}")
        
        # Cast integer features
        if feature in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "age", "trestbps", "chol", "thalach"]:
            clean_data[feature] = int(round(val))
        else:
            clean_data[feature] = float(val)

    # Range checks
    if not (10 <= clean_data["age"] <= 125):
        raise ValueError(f"Age {clean_data['age']} is outside plausible range (10-125).")
    if clean_data["sex"] not in [0, 1]:
        raise ValueError(f"Sex must be 0 (Female) or 1 (Male), got {clean_data['sex']}.")
    if not (50 <= clean_data["trestbps"] <= 300):
        raise ValueError(f"Resting BP {clean_data['trestbps']} mm Hg outside plausible clinical range.")
    if not (50 <= clean_data["chol"] <= 800):
        raise ValueError(f"Cholesterol {clean_data['chol']} mg/dl outside plausible clinical range.")

    return clean_data


def predict_patient_risk(data: Dict[str, Any], pipeline=None) -> Dict[str, Any]:
    """
    Run machine learning inference on patient features and generate clinical risk evaluation.
    """
    clean = validate_patient_input(data)
    df_input = pd.DataFrame([clean])[FEATURE_NAMES]

    if pipeline is None:
        pipeline = load_trained_pipeline()

    # Predict class
    prediction = int(pipeline.predict(df_input)[0])

    # Predict probabilities if supported
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df_input)[0]
        # Probability of class 1 (heart disease)
        probability = float(probs[1])
    else:
        probability = 1.0 if prediction == 1 else 0.0

    # Categorize clinical risk tier
    if probability >= 0.75:
        risk_level = "High Risk"
        badge = "CRITICAL / HIGH RISK"
    elif probability >= 0.50:
        risk_level = "Moderate Risk"
        badge = "ELEVATED RISK"
    elif probability >= 0.25:
        risk_level = "Mild Risk"
        badge = "LOW-MODERATE RISK"
    else:
        risk_level = "Low Risk"
        badge = "HEALTHY / MINIMAL RISK"

    risk_factors = analyze_risk_factors(clean)

    return {
        "prediction": prediction,
        "label": "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected",
        "risk_level": risk_level,
        "risk_badge": badge,
        "disease_probability": round(probability, 4),
        "confidence_percentage": round(probability * 100, 2),
        "risk_factors": risk_factors,
        "patient_data": clean
    }
