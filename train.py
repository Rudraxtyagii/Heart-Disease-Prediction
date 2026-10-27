#!/usr/bin/env python3
"""
CLI Script to train, benchmark, and persist the Heart Disease Prediction pipeline.
Usage:
    python train.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import HeartDiseaseModelPipeline


def main():
    print("=" * 65)
    print("  HEART DISEASE PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 65)

    trainer = HeartDiseaseModelPipeline()
    print("\n[1/3] Loading and preprocessing dataset from:", trainer.data_path)
    
    print("[2/3] Benchmarking classification algorithms (Stratified 5-Fold CV)...")
    summary = trainer.train_final_pipeline()

    print("\n" + "-" * 65)
    print("MODEL COMPARISON BENCHMARK:")
    print("-" * 65)
    bench_df = summary["benchmarks"]
    print(bench_df.to_string(index=False))

    print("\n" + "-" * 65)
    print("PRODUCTION PIPELINE EVALUATION (Random Forest with Scaling):")
    print("-" * 65)
    metrics = summary["metrics"]
    print(f"Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")

    print("\nCLASSIFICATION REPORT:")
    print(summary["classification_report"])

    print("-" * 65)
    print("TOP PREDICTIVE CARDIAC FEATURES:")
    print("-" * 65)
    for feat, imp in list(summary["feature_importances"].items())[:6]:
        bar = "#" * int(imp * 40)
        print(f"  {feat:<12} : {imp:.4f} | {bar}")

    print("\n[3/3] Saving production artifacts...")
    saved_model_path = trainer.save_artifacts()
    print(f"[SUCCESS] Trained model pipeline serialized to: {saved_model_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
