"""
Centralized Configuration and Clinical Metadata for Heart Disease Prediction System.
"""

import os
from pathlib import Path
from typing import Dict, Any, List

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "minor" / "data"
MODEL_DIR = PROJECT_ROOT / "minor" / "model"
RESULT_DIR = PROJECT_ROOT / "minor" / "result"
REPORTS_DIR = PROJECT_ROOT / "minor" / "reports"

# Ensure runtime directories exist
for directory in [DATA_DIR, MODEL_DIR, RESULT_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASET_PATH = DATA_DIR / "heart.csv"
DEFAULT_PIPELINE_PATH = MODEL_DIR / "heart_disease_pipeline.pkl"
DEFAULT_CALIBRATED_PIPELINE_PATH = MODEL_DIR / "calibrated_pipeline.pkl"
DEFAULT_METRICS_PATH = RESULT_DIR / "model_comparison.json"

# Server / Deployment Config
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8050"))
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MODEL_VERSION = "2.1.0"

# ML Pipeline Settings
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.20
CV_FOLDS = 5

FEATURE_NAMES: List[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "age": {
        "label": "Age",
        "description": "Patient biological age in years",
        "unit": "years",
        "type": int,
        "default": 55,
        "min": 18,
        "max": 120,
        "reference_range": "18 - 120 years"
    },
    "sex": {
        "label": "Biological Sex",
        "description": "Biological sex at birth (1: Male, 0: Female)",
        "unit": "category",
        "type": int,
        "choices": {0: "Female", 1: "Male"},
        "default": 1
    },
    "cp": {
        "label": "Chest Pain Type",
        "description": "Symptomatic presentation of chest discomfort",
        "unit": "grade (0-3)",
        "type": int,
        "choices": {
            0: "0: Typical Angina (Exertional chest discomfort relieved by rest/nitro)",
            1: "1: Atypical Angina (Non-exertional or atypical presentation)",
            2: "2: Non-Anginal Pain (Musculoskeletal/gastrointestinal origin)",
            3: "3: Asymptomatic (Absence of typical anginal discomfort)"
        },
        "default": 0
    },
    "trestbps": {
        "label": "Resting Blood Pressure",
        "description": "Resting arterial systolic blood pressure upon clinical intake",
        "unit": "mm Hg",
        "type": int,
        "default": 130,
        "min": 80,
        "max": 250,
        "reference_range": "< 120 mm Hg (Normal), 130-139 (Stage 1), >= 140 (Stage 2)"
    },
    "chol": {
        "label": "Serum Cholesterol",
        "description": "Total serum cholesterol level",
        "unit": "mg/dl",
        "type": int,
        "default": 240,
        "min": 100,
        "max": 600,
        "reference_range": "< 200 mg/dl (Desirable), 200-239 (Borderline), >= 240 (High)"
    },
    "fbs": {
        "label": "Fasting Blood Sugar > 120 mg/dl",
        "description": "Surrogate marker for impaired fasting glucose / diabetic status",
        "unit": "binary",
        "type": int,
        "choices": {0: "No (<= 120 mg/dl, Normal)", 1: "Yes (> 120 mg/dl, Elevated)"},
        "default": 0
    },
    "restecg": {
        "label": "Resting Electrocardiogram (ECG)",
        "description": "Resting 12-lead electrocardiographic evaluation",
        "unit": "category (0-2)",
        "type": int,
        "choices": {
            0: "0: Normal ECG",
            1: "1: ST-T Wave Abnormality (T-wave inversions / ST depression > 0.05 mV)",
            2: "2: Left Ventricular Hypertrophy (Estes' criteria or voltage criteria)"
        },
        "default": 1
    },
    "thalach": {
        "label": "Maximum Heart Rate Achieved",
        "description": "Peak heart rate recorded during treadmill/ergometer exercise stress testing",
        "unit": "bpm",
        "type": int,
        "default": 150,
        "min": 60,
        "max": 230,
        "reference_range": "Age-predicted max: 220 - age (bpm)"
    },
    "exang": {
        "label": "Exercise-Induced Angina",
        "description": "Appearance of classic ischemic angina during exertion",
        "unit": "binary",
        "type": int,
        "choices": {0: "No (Negative)", 1: "Yes (Positive for Angina)"},
        "default": 0
    },
    "oldpeak": {
        "label": "ST Depression (oldpeak)",
        "description": "ST-segment depression induced by exercise relative to resting baseline",
        "unit": "mm / mV",
        "type": float,
        "default": 1.0,
        "min": 0.0,
        "max": 10.0,
        "reference_range": "< 1.0 mm (Normal), >= 1.0 mm (Equivocal), >= 2.0 mm (Marked Ischemia)"
    },
    "slope": {
        "label": "Slope of Peak Exercise ST Segment",
        "description": "Geometric configuration of the ST segment during peak exercise stress",
        "unit": "grade (0-2)",
        "type": int,
        "choices": {
            0: "0: Upsloping (Often normal/benign physiologic response)",
            1: "1: Flat (Equivocal / early ischemic response)",
            2: "2: Downsloping (Classic high-risk myocardial ischemia)"
        },
        "default": 1
    },
    "ca": {
        "label": "Major Vessels Colored (Fluoroscopy)",
        "description": "Number of major epicardial coronary arteries (LAD, LCx, RCA) colored by fluoroscopy",
        "unit": "vessels (0-3)",
        "type": int,
        "choices": {0: "0 vessels blocked", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels (Multi-vessel CAD)"},
        "default": 0,
        "min": 0,
        "max": 4
    },
    "thal": {
        "label": "Thalassemia / Perfusion Scintigraphy",
        "description": "Myocardial perfusion scintigraphy status (nuclear stress defect pattern)",
        "unit": "status (1-3)",
        "type": int,
        "choices": {
            1: "1: Normal Perfusion",
            2: "2: Fixed Defect (Prior myocardial infarction / non-viable scar)",
            3: "3: Reversible Defect (Stress-induced ischemia resolving at rest)"
        },
        "default": 2,
        "min": 0,
        "max": 3
    }
}

def get_serializable_feature_metadata() -> Dict[str, Dict[str, Any]]:
    """Return JSON-serializable copy of feature metadata with types as strings."""
    out = {}
    for k, v in FEATURE_METADATA.items():
        item = v.copy()
        if "type" in item and hasattr(item["type"], "__name__"):
            item["type"] = item["type"].__name__
        out[k] = item
    return out


CLINICAL_DISCLAIMER = (
    "DISCLAIMER: This software is an academic machine learning research and educational demonstration prototype. "
    "It is NOT a certified medical diagnostic device (CE-IVD / FDA approved) and must NEVER be used as a substitute "
    "for professional clinical assessment, angiographic evaluation, or emergency medical judgment. "
    "Always consult a licensed cardiologist for individual patient diagnoses and treatment decisions."
)
