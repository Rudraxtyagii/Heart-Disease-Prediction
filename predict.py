#!/usr/bin/env python3
"""
CLI Tool for Cardiac Risk Inference and Explainable AI (SHAP) Analysis.
Usage:
    python predict.py --demo
    python predict.py --demo --explain
    python predict.py --interactive
    python predict.py --age 62 --sex 1 --cp 0 --trestbps 160 --chol 290 --fbs 1 \
                      --restecg 2 --thalach 115 --exang 1 --oldpeak 2.8 --slope 1 --ca 2 --thal 3 --explain
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_METADATA, FEATURE_NAMES
from src.predict import predict_patient_risk


def print_prediction_report(res: dict, show_explanation: bool = False):
    """Print formatted clinical evaluation report to terminal."""
    print("\n" + "=" * 68)
    print("           CARDIAC RISK EVALUATION REPORT")
    print("=" * 68)
    
    label = res["label"]
    conf = res["confidence_percentage"]
    badge = res["risk_badge"]
    tier = res["risk_level"]
    
    print(f"  Diagnostic Result     : {label.upper()}")
    print(f"  Risk Category Tier    : [{badge}] ({tier})")
    print(f"  Calibrated Probability: {conf}% (Raw ensemble: {res.get('raw_probability', 0.0):.4f})")
    print("-" * 68)
    
    print("  Patient Vitals & Indicators:")
    p = res["patient_data"]
    sex_str = "Male" if p["sex"] == 1 else "Female"
    print(f"    - Demographics : Age {p['age']}, {sex_str}")
    print(f"    - Vitals       : Resting BP {p['trestbps']} mm Hg | Max HR {p['thalach']} bpm")
    print(f"    - Blood Work   : Cholesterol {p['chol']} mg/dl | FBS > 120: {'Yes' if p['fbs'] else 'No'}")
    print(f"    - Stress/ECG   : Chest Pain Type {p['cp']} | ST Depression {p['oldpeak']} mm | ExAngina: {'Yes' if p['exang'] else 'No'}")
    print(f"    - Anatomy      : Fluoroscopy Vessels: {p['ca']} | Thalassemia: {p['thal']}")
    print("-" * 68)

    alerts = res.get("risk_factors", [])
    if alerts:
        print("  Clinical Risk Flags (AHA Guidelines):")
        for a in alerts:
            print(f"    [!] {a['feature']}: {a['value']} ({a['status']})")
            print(f"        -> {a['clinical_note']}")
    else:
        print("  Clinical Risk Flags: No abnormal threshold alerts identified.")

    # Explainable AI section
    if show_explanation and "explanation" in res:
        xai = res["explanation"]
        print("-" * 68)
        print("  Explainable AI (SHAP Local Feature Contributions):")
        top_inc = xai.get("top_risk_drivers", [])
        top_dec = xai.get("top_protective_factors", [])
        
        if top_inc:
            print("    [+] Factors Increasing Cardiac Risk:")
            for item in top_inc[:3]:
                bar = "#" * int(min(item["abs_magnitude"] * 50, 25))
                print(f"        + {item['label']:<32}: +{item['shap_value']:.4f} | {bar}")
        if top_dec:
            print("    [-] Factors Decreasing Cardiac Risk (Protective):")
            for item in top_dec[:3]:
                bar = "-" * int(min(item["abs_magnitude"] * 50, 25))
                print(f"        - {item['label']:<32}: {item['shap_value']:.4f} | {bar}")

        print(f"\n    Narrative Synthesis:")
        print(f"    {xai.get('clinical_narrative', '')}")

    print("=" * 68 + "\n")


def run_interactive_wizard(show_explanation: bool = True):
    """Prompt clinician / user step-by-step for the 13 clinical features."""
    print("=" * 68)
    print("  INTERACTIVE CARDIAC ASSESSMENT WIZARD")
    print("  Press Enter to accept recommended default value shown in [brackets].")
    print("=" * 68)

    data = {}
    for feat in FEATURE_NAMES:
        meta = FEATURE_METADATA[feat]
        label = meta["label"]
        default_val = meta["default"]
        unit = meta.get("unit", "")
        
        choices = meta.get("choices")
        if choices:
            choice_str = ", ".join([f"{k}={v[:25]}" for k, v in choices.items()])
            prompt = f"  {label} ({choice_str}) [{default_val}]: "
        else:
            prompt = f"  {label} ({unit}) [{default_val}]: "

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

    res = predict_patient_risk(data, include_explanation=show_explanation)
    print_prediction_report(res, show_explanation=show_explanation)


def run_demo(show_explanation: bool = True):
    """Demonstrate inference on contrasting clinical profiles."""
    print("\nExecuting Demonstration Mode with 3 Clinical Profiles...\n")

    profiles = [
        (
            "PROFILE 1: High-Risk Profile (Severe Angina, Blocked Vessels, Ischemic ST)",
            {
                "age": 62, "sex": 1, "cp": 0, "trestbps": 160, "chol": 290, "fbs": 1,
                "restecg": 2, "thalach": 115, "exang": 1, "oldpeak": 2.8, "slope": 1,
                "ca": 2, "thal": 3
            }
        ),
        (
            "PROFILE 2: Borderline / Intermediate Profile",
            {
                "age": 56, "sex": 0, "cp": 1, "trestbps": 135, "chol": 235, "fbs": 0,
                "restecg": 1, "thalach": 142, "exang": 0, "oldpeak": 1.2, "slope": 1,
                "ca": 1, "thal": 2
            }
        ),
        (
            "PROFILE 3: Healthy / Low-Risk Profile",
            {
                "age": 41, "sex": 0, "cp": 2, "trestbps": 118, "chol": 195, "fbs": 0,
                "restecg": 0, "thalach": 172, "exang": 0, "oldpeak": 0.2, "slope": 2,
                "ca": 0, "thal": 2
            }
        )
    ]

    for title, patient in profiles:
        print(f"--- {title} ---")
        res = predict_patient_risk(patient, include_explanation=show_explanation)
        print_prediction_report(res, show_explanation=show_explanation)


def main():
    parser = argparse.ArgumentParser(description="Cardiac Risk Inference & Explainable AI (SHAP) CLI")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive assessment wizard")
    parser.add_argument("--demo", action="store_true", help="Run demonstration with sample patient profiles")
    parser.add_argument("--explain", action="store_true", help="Include SHAP feature attribution breakdown")

    # Feature arguments
    for feat in FEATURE_NAMES:
        meta = FEATURE_METADATA[feat]
        parser.add_argument(f"--{feat}", type=meta["type"], help=meta["description"])

    args = parser.parse_args()

    if args.interactive:
        run_interactive_wizard(show_explanation=args.explain or True)
        return

    if args.demo:
        run_demo(show_explanation=args.explain or True)
        return

    cli_dict = {f: getattr(args, f) for f in FEATURE_NAMES if getattr(args, f) is not None}
    if len(cli_dict) == len(FEATURE_NAMES):
        res = predict_patient_risk(cli_dict, include_explanation=args.explain)
        print_prediction_report(res, show_explanation=args.explain)
    elif len(cli_dict) > 0:
        missing = [f for f in FEATURE_NAMES if getattr(args, f) is None]
        print(f"Error: Incomplete feature set. Missing: {', '.join(missing)}")
        print("Tip: Run with --interactive or pass all 13 features.")
        sys.exit(1)
    else:
        print("No feature flags passed. Running demonstration mode with SHAP explanations...\n")
        run_demo(show_explanation=True)


if __name__ == "__main__":
    main()
