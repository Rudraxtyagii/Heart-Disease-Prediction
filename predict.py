#!/usr/bin/env python3
"""
CLI Tool for Heart Disease Prediction & Risk Analysis.
Supports single CLI flag predictions, interactive prompt mode, and sample demo mode.

Usage:
    python predict.py --demo
    python predict.py --interactive
    python predict.py --age 58 --sex 1 --cp 2 --trestbps 140 --chol 260 --fbs 0 \
                      --restecg 1 --thalach 165 --exang 0 --oldpeak 0.8 --slope 2 --ca 0 --thal 2
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import FEATURE_METADATA, FEATURE_NAMES
from src.predict import predict_patient_risk


def print_prediction_report(res: dict):
    """Print beautifully formatted cardiac evaluation report to terminal."""
    print("\n" + "=" * 65)
    print("           CARDIAC RISK EVALUATION REPORT")
    print("=" * 65)
    
    label = res["label"]
    conf = res["confidence_percentage"]
    badge = res["risk_badge"]
    
    print(f"  Diagnosis Result : {label.upper()}")
    print(f"  Risk Category    : [{badge}]")
    print(f"  Disease Risk     : {conf}% (probability: {res['disease_probability']})")
    print("-" * 65)
    
    print("  Patient Vitals & Indicators:")
    p = res["patient_data"]
    sex_str = "Male" if p["sex"] == 1 else "Female"
    print(f"    - Demographics : Age {p['age']}, {sex_str}")
    print(f"    - Vitals       : Resting BP {p['trestbps']} mm Hg | Max HR {p['thalach']} bpm")
    print(f"    - Blood Work   : Cholesterol {p['chol']} mg/dl | FBS > 120: {'Yes' if p['fbs'] else 'No'}")
    print(f"    - ECG / Stress : Chest Pain Type {p['cp']} | ST Depression {p['oldpeak']} | ExAngina: {'Yes' if p['exang'] else 'No'}")
    print(f"    - Anatomy      : Fluoroscopy Vessels: {p['ca']} | Thalassemia: {p['thal']}")
    print("-" * 65)

    alerts = res["risk_factors"]
    if alerts:
        print("  Clinical Risk Factors Detected:")
        for a in alerts:
            print(f"    [!] {a['feature']}: {a['value']} ({a['status']})")
            print(f"        -> {a['clinical_note']}")
    else:
        print("  Clinical Risk Factors: No critical flags identified.")

    print("=" * 65 + "\n")


def run_interactive_wizard():
    """Prompt clinician / user step-by-step for the 13 clinical features."""
    print("=" * 65)
    print("  INTERACTIVE CARDIAC ASSESSMENT WIZARD")
    print("  Press Enter to accept recommended default value shown in [brackets].")
    print("=" * 65)

    data = {}
    for feat in FEATURE_NAMES:
        meta = FEATURE_METADATA[feat]
        label = meta["label"]
        default_val = meta["default"]
        
        choices = meta.get("choices")
        if choices:
            choice_str = ", ".join([f"{k}={v}" for k, v in choices.items()])
            prompt = f"  {label} ({choice_str}) [{default_val}]: "
        else:
            prompt = f"  {label} [{default_val}]: "

        while True:
            raw = input(prompt).strip()
            if not raw:
                data[feat] = default_val
                break
            try:
                if meta["type"] == int:
                    data[feat] = int(raw)
                else:
                    data[feat] = float(raw)
                break
            except ValueError:
                print(f"    Invalid input. Please enter a valid {meta['type'].__name__}.")

    res = predict_patient_risk(data)
    print_prediction_report(res)


def run_demo():
    """Demonstrate inference on two contrasting clinical profiles."""
    print("\nRunning demonstration with sample clinical profiles...\n")

    # Sample Profile A: High Risk
    patient_high_risk = {
        "age": 62, "sex": 1, "cp": 0, "trestbps": 160, "chol": 290, "fbs": 1,
        "restecg": 2, "thalach": 115, "exang": 1, "oldpeak": 2.8, "slope": 1,
        "ca": 2, "thal": 3
    }

    # Sample Profile B: Healthy / Low Risk
    patient_low_risk = {
        "age": 41, "sex": 0, "cp": 2, "trestbps": 118, "chol": 195, "fbs": 0,
        "restecg": 0, "thalach": 172, "exang": 0, "oldpeak": 0.2, "slope": 2,
        "ca": 0, "thal": 2
    }

    print("--- PROFILE 1: High-Risk Profile ---")
    res1 = predict_patient_risk(patient_high_risk)
    print_prediction_report(res1)

    print("--- PROFILE 2: Healthy Profile ---")
    res2 = predict_patient_risk(patient_low_risk)
    print_prediction_report(res2)


def main():
    parser = argparse.ArgumentParser(description="Heart Disease Risk Inference CLI")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive assessment wizard")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample patient profiles")

    # Feature arguments
    for feat in FEATURE_NAMES:
        meta = FEATURE_METADATA[feat]
        parser.add_argument(f"--{feat}", type=meta["type"], help=meta["description"])

    args = parser.parse_args()

    if args.interactive:
        run_interactive_wizard()
        return

    if args.demo:
        run_demo()
        return

    # Check if flags were passed directly
    cli_dict = {f: getattr(args, f) for f in FEATURE_NAMES if getattr(args, f) is not None}
    if len(cli_dict) == len(FEATURE_NAMES):
        res = predict_patient_risk(cli_dict)
        print_prediction_report(res)
    elif len(cli_dict) > 0:
        missing = [f for f in FEATURE_NAMES if getattr(args, f) is None]
        print(f"Error: Incomplete feature set passed. Missing: {', '.join(missing)}")
        print("Run with --interactive or pass all 13 features.")
        sys.exit(1)
    else:
        # Default behavior: run demo
        print("No input arguments specified. Running default demo mode.")
        print("Tip: Use 'python predict.py --interactive' for guided input, or '--help' for flags.\n")
        run_demo()


if __name__ == "__main__":
    main()
