#!/usr/bin/env python3
"""
Production Training, Calibration, Benchmarking, and Explainable AI Pipeline.
Usage:
    python train.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import HeartDiseaseModelTrainer
from src.explain import HeartDiseaseExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    print("=" * 72)
    print("  CARDIOPREDICT - MODEL TRAINING, BENCHMARKING & XAI PIPELINE")
    print("=" * 72)

    trainer = HeartDiseaseModelTrainer()
    
    print("\n[1/4] Loading dataset, dropping duplicates & verifying target labels...")
    print("[2/4] Executing Stratified 5-Fold CV Across 5 Candidate Classifiers...")
    summary = trainer.train_and_benchmark()

    print("\n" + "=" * 72)
    print("  CLINICAL BENCHMARK EVALUATION (HELD-OUT TEST SET)")
    print("=" * 72)
    print(f"{'Model':<28} | {'Acc':<6} | {'Recall':<6} | {'Spec':<6} | {'F1':<6} | {'AUC':<6} | {'CV Mean':<7}")
    print("-" * 72)
    for b in summary["benchmarks"]:
        print(
            f"{b['model_name']:<28} | "
            f"{b['accuracy']:.4f} | "
            f"{b['sensitivity_recall']:.4f} | "
            f"{b['specificity']:.4f} | "
            f"{b['f1_score']:.4f} | "
            f"{b['roc_auc']:.4f} | "
            f"{b['cv_5fold_mean']:.4f} +- {b['cv_5fold_std']:.3f}"
        )

    print("-" * 72)
    print(f"Probability Calibration (Random Forest Brier Score): {summary['calibrated_rf_brier_score']:.4f}")

    print("\n" + "=" * 72)
    print("  TOP PREDICTIVE CARDIAC RISK FEATURES")
    print("=" * 72)
    for feat, imp in list(summary["feature_importances"].items())[:7]:
        bar = "#" * int(imp * 45)
        print(f"  {feat:<12} : {imp:.4f} | {bar}")

    print("\n[3/4] Generating Global SHAP Explainable AI Summary Plot...")
    explainer = HeartDiseaseExplainer(trainer.production_pipeline)
    xai_res = explainer.generate_global_explanations()
    print(f"  [SUCCESS] SHAP Summary Plot generated: {xai_res['plot_saved_path']}")

    print("\n[4/4] Serializing production model artifacts and benchmark records...")
    paths = trainer.save_artifacts()
    print(f"  [SUCCESS] Production Pipeline : {paths['pipeline']}")
    print(f"  [SUCCESS] Calibrated Pipeline : {paths['calibrated']}")
    print(f"  [SUCCESS] Evaluation Summary  : {paths['metrics']}")
    print("=" * 72)
    print("ALL TRAINING & BENCHMARKING PHASES COMPLETED SUCCESSFULLY.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
