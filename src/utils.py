"""
Utility functions and configuration constants for the Heart Disease Prediction system.
"""

from pathlib import Path
from typing import Dict, Any, List

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "age": {
        "label": "Age",
        "description": "Patient age in years",
        "type": int,
        "default": 55,
        "min": 18,
        "max": 120
    },
    "sex": {
        "label": "Sex",
        "description": "Biological sex (1: Male, 0: Female)",
        "type": int,
        "choices": {0: "Female", 1: "Male"},
        "default": 1
    },
    "cp": {
        "label": "Chest Pain Type (cp)",
        "description": "Type of chest pain reported",
        "type": int,
        "choices": {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-Anginal Pain",
            3: "Asymptomatic"
        },
        "default": 0
    },
    "trestbps": {
        "label": "Resting Blood Pressure (trestbps)",
        "description": "Resting blood pressure in mm Hg (normal: 90-120)",
        "type": int,
        "default": 130,
        "min": 80,
        "max": 250
    },
    "chol": {
        "label": "Serum Cholesterol (chol)",
        "description": "Serum cholesterol in mg/dl (desirable < 200)",
        "type": int,
        "default": 240,
        "min": 100,
        "max": 600
    },
    "fbs": {
        "label": "Fasting Blood Sugar > 120 mg/dl (fbs)",
        "description": "Fasting blood sugar > 120 mg/dl (1: True, 0: False)",
        "type": int,
        "choices": {0: "False (<= 120 mg/dl)", 1: "True (> 120 mg/dl)"},
        "default": 0
    },
    "restecg": {
        "label": "Resting ECG (restecg)",
        "description": "Resting electrocardiographic results",
        "type": int,
        "choices": {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy (Estes' criteria)"
        },
        "default": 1
    },
    "thalach": {
        "label": "Max Heart Rate Achieved (thalach)",
        "description": "Maximum heart rate achieved during exercise stress test (bpm)",
        "type": int,
        "default": 150,
        "min": 60,
        "max": 230
    },
    "exang": {
        "label": "Exercise Induced Angina (exang)",
        "description": "Exercise induced angina (1: Yes, 0: No)",
        "type": int,
        "choices": {0: "No", 1: "Yes"},
        "default": 0
    },
    "oldpeak": {
        "label": "ST Depression (oldpeak)",
        "description": "ST depression induced by exercise relative to rest",
        "type": float,
        "default": 1.0,
        "min": 0.0,
        "max": 10.0
    },
    "slope": {
        "label": "Slope of Peak Exercise ST (slope)",
        "description": "The slope of the peak exercise ST segment",
        "type": int,
        "choices": {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        },
        "default": 1
    },
    "ca": {
        "label": "Major Vessels Colored (ca)",
        "description": "Number of major vessels (0-3) colored by fluoroscopy",
        "type": int,
        "choices": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
        "default": 0
    },
    "thal": {
        "label": "Thalassemia (thal)",
        "description": "Thalassemia blood disorder status",
        "type": int,
        "choices": {
            1: "Normal",
            2: "Fixed Defect",
            3: "Reversible Defect"
        },
        "default": 2
    }
}


def find_project_root() -> Path:
    """Locate the root directory of the Heart Disease project."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "minor" / "data" / "heart.csv").exists():
            return parent
        if (parent / "requirements.txt").exists():
            return parent
    return Path.cwd()


def get_dataset_path() -> Path:
    """Find path to heart.csv dataset."""
    root = find_project_root()
    candidates = [
        root / "minor" / "data" / "heart.csv",
        root / "data" / "heart.csv",
        Path("minor/data/heart.csv"),
        Path("data/heart.csv")
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find heart.csv in project directory.")


def get_model_path() -> Path:
    """Return path for primary saved model pipeline."""
    root = find_project_root()
    model_dir = root / "minor" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "heart_disease_pipeline.pkl"


def analyze_risk_factors(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Perform clinical heuristics check on key cardiac indicators to provide
    contextual risk insights alongside model predictions.
    """
    alerts = []
    
    # Blood pressure check (AHA guidelines)
    trestbps = data.get("trestbps", 120)
    if trestbps >= 140:
        alerts.append({
            "feature": "Resting Blood Pressure",
            "status": "High (Hypertension Stage 2)",
            "value": f"{trestbps} mm Hg",
            "clinical_note": "Significantly elevated resting blood pressure strains arterial walls."
        })
    elif trestbps >= 130:
        alerts.append({
            "feature": "Resting Blood Pressure",
            "status": "Elevated (Hypertension Stage 1)",
            "value": f"{trestbps} mm Hg",
            "clinical_note": "Prehypertensive reading; monitoring lifestyle and sodium intake advised."
        })

    # Cholesterol check
    chol = data.get("chol", 200)
    if chol >= 240:
        alerts.append({
            "feature": "Serum Cholesterol",
            "status": "High Risk (> 240 mg/dl)",
            "value": f"{chol} mg/dl",
            "clinical_note": "High cholesterol accelerates atherosclerotic plaque accumulation."
        })
    elif chol >= 200:
        alerts.append({
            "feature": "Serum Cholesterol",
            "status": "Borderline High (200-239 mg/dl)",
            "value": f"{chol} mg/dl",
            "clinical_note": "Borderline cholesterol; lipid panel monitoring recommended."
        })

    # Exercise induced angina
    if data.get("exang") == 1:
        alerts.append({
            "feature": "Exercise Induced Angina",
            "status": "Present",
            "value": "Yes (Positive)",
            "clinical_note": "Chest pain during physical exertion strongly correlates with myocardial ischemia."
        })

    # ST depression
    oldpeak = data.get("oldpeak", 0.0)
    if oldpeak >= 2.0:
        alerts.append({
            "feature": "ST Depression (oldpeak)",
            "status": "Marked Depression (>= 2.0)",
            "value": f"{oldpeak}",
            "clinical_note": "Significant ST depression suggests exercise-induced coronary insufficiency."
        })

    # Fluoroscopy vessel count
    ca = data.get("ca", 0)
    if ca > 0:
        alerts.append({
            "feature": "Major Vessels Colored (Fluoroscopy)",
            "status": f"Vessel Involvement ({ca})",
            "value": f"{ca} major vessel(s)",
            "clinical_note": "Fluoroscopy opacification indicates narrowed or blocked coronary vessels."
        })

    # Thalassemia defect
    thal = data.get("thal")
    if thal == 3:
        alerts.append({
            "feature": "Thalassemia Test",
            "status": "Reversible Defect",
            "value": "Reversible Defect",
            "clinical_note": "Perfusion defect present during stress that resolves at rest; classic ischemia indicator."
        })

    return alerts
