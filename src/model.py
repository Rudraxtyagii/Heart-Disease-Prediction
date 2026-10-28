"""
Machine Learning Training, Multi-Model Benchmarking, Probability Calibration, and Artifact Serialization.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from src.config import (
    DEFAULT_PIPELINE_PATH,
    DEFAULT_CALIBRATED_PIPELINE_PATH,
    DEFAULT_METRICS_PATH,
    RESULT_DIR,
    FEATURE_NAMES,
    RANDOM_STATE,
    CV_FOLDS,
    MODEL_VERSION
)
from src.dataset import DatasetHandler

logger = logging.getLogger(__name__)


class HeartDiseaseModelTrainer:
    """End-to-end Machine Learning training, calibration, and benchmarking suite."""

    def __init__(self, data_path: Path = None, random_state: int = RANDOM_STATE):
        self.dataset_handler = DatasetHandler(data_path)
        self.random_state = random_state
        self.production_pipeline: Pipeline = None
        self.calibrated_pipeline: CalibratedClassifierCV = None
        self.benchmark_results: List[Dict[str, Any]] = []
        self.feature_importances: Dict[str, float] = {}
        self.evaluation_summary: Dict[str, Any] = {}

    def get_candidate_models(self) -> Dict[str, Pipeline]:
        """Construct standard Scikit-Learn pipelines for all candidate algorithms."""
        return {
            "Random Forest": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=150,
                    max_depth=6,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=self.random_state
                ))
            ]),
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=0.5,
                    max_iter=1000,
                    random_state=self.random_state
                ))
            ]),
            "Support Vector Machine (SVM)": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    C=1.0,
                    kernel="rbf",
                    probability=True,
                    random_state=self.random_state
                ))
            ]),
            "K-Nearest Neighbors (KNN)": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))
            ]),
            "Decision Tree": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_split=6,
                    random_state=self.random_state
                ))
            ])
        }

    def train_and_benchmark(self) -> Dict[str, Any]:
        """
        Executes full stratified benchmarking across all 5 candidate models,
        computes comprehensive clinical performance metrics, fits calibration,
        and generates visual benchmark charts.
        """
        X_train, X_test, y_train, y_test, split_meta = self.dataset_handler.get_stratified_split()
        models = self.get_candidate_models()
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=self.random_state)

        benchmark_records = []
        roc_data = {}
        pr_data = {}
        confusion_data = {}
        calibration_data = {}

        for name, pipe in models.items():
            logger.info(f"Training and evaluating model: {name}")
            
            # 5-fold cross-validation on training fold
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())

            # Train on entire training set
            pipe.fit(X_train, y_train)

            # Test set predictions
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            # Confusion Matrix: TN, FP, FN, TP
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            roc_auc = float(roc_auc_score(y_test, y_prob))
            pr_auc = float(average_precision_score(y_test, y_prob))
            brier = float(brier_score_loss(y_test, y_prob))

            # Curves
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=5, strategy="uniform")

            roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
            pr_data[name] = {"precision": prec_curve.tolist(), "recall": rec_curve.tolist(), "auc": pr_auc}
            confusion_data[name] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
            calibration_data[name] = {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist(), "brier": brier}

            record = {
                "model_name": name,
                "accuracy": acc,
                "precision": prec,
                "sensitivity_recall": rec,
                "specificity": specificity,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_score": brier,
                "cv_5fold_mean": cv_mean,
                "cv_5fold_std": cv_std
            }
            benchmark_records.append(record)

        # Primary production model: Random Forest
        self.production_pipeline = models["Random Forest"]

        # Train a calibrated classifier wrapper using 5-fold cross-validation
        logger.info("Fitting CalibratedClassifierCV (isotonic) on Random Forest pipeline...")
        self.calibrated_pipeline = CalibratedClassifierCV(
            estimator=self.production_pipeline,
            method="sigmoid",
            cv=5
        )
        self.calibrated_pipeline.fit(X_train, y_train)
        cal_prob = self.calibrated_pipeline.predict_proba(X_test)[:, 1]
        cal_brier = float(brier_score_loss(y_test, cal_prob))
        logger.info(f"Probability Calibration: Uncalibrated RF Brier: {calibration_data['Random Forest']['brier']:.4f} -> Calibrated RF Brier: {cal_brier:.4f}")

        # Extract feature importances
        rf_step: RandomForestClassifier = self.production_pipeline.named_steps["clf"]
        importances = pd.Series(rf_step.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
        self.feature_importances = {k: float(v) for k, v in importances.items()}

        self.benchmark_results = sorted(benchmark_records, key=lambda x: x["accuracy"], reverse=True)

        self.evaluation_summary = {
            "model_version": MODEL_VERSION,
            "split_info": split_meta,
            "benchmarks": self.benchmark_results,
            "calibrated_rf_brier_score": cal_brier,
            "feature_importances": self.feature_importances,
            "roc_curves": roc_data,
            "pr_curves": pr_data,
            "confusion_matrices": confusion_data,
            "calibration_data": calibration_data
        }

        # Generate and save comparison plots
        self._generate_benchmark_plots(y_test, roc_data, pr_data, calibration_data)

        return self.evaluation_summary

    def _generate_benchmark_plots(
        self,
        y_test: pd.Series,
        roc_data: Dict[str, Any],
        pr_data: Dict[str, Any],
        calibration_data: Dict[str, Any]
    ) -> None:
        """Render high-resolution benchmark visualization figures."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Benchmark Bar Chart Comparison
        df_bench = pd.DataFrame(self.benchmark_results).set_index("model_name")
        metrics_to_plot = ["accuracy", "sensitivity_recall", "specificity", "f1_score", "roc_auc"]
        
        fig, ax = plt.subplots(figsize=(11, 6))
        df_bench[metrics_to_plot].plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
        ax.set_title("Clinical Model Benchmark Comparison (Held-Out Test Set)", fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel("Metric Score (0.0 to 1.0)", fontsize=12)
        ax.set_xlabel("Classification Algorithm", fontsize=12)
        ax.set_ylim(0.5, 1.02)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=20, ha="right", fontsize=10)
        plt.legend(["Accuracy", "Sensitivity (Recall)", "Specificity", "F1 Score", "ROC-AUC"], loc="lower right", framealpha=0.9)
        plt.tight_layout()
        fig.savefig(RESULT_DIR / "model_comparison.png", dpi=200)
        plt.close(fig)

        # 2. ROC Curves Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, data in roc_data.items():
            ax.plot(data["fpr"], data["tpr"], label=f"{name} (AUC = {data['auc']:.3f})", lw=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance (AUC = 0.500)")
        ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
        plt.tight_layout()
        fig.savefig(RESULT_DIR / "roc_curves.png", dpi=200)
        plt.close(fig)

        # 3. Precision-Recall Curves Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, data in pr_data.items():
            ax.plot(data["recall"], data["precision"], label=f"{name} (PR-AUC = {data['auc']:.3f})", lw=2)
        base_rate = float(y_test.mean())
        ax.axhline(y=base_rate, linestyle="--", color="gray", label=f"Baseline Prevalence ({base_rate:.3f})")
        ax.set_title("Precision-Recall (PR) Curves", fontsize=13, fontweight="bold")
        ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
        ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
        ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
        plt.tight_layout()
        fig.savefig(RESULT_DIR / "pr_curves.png", dpi=200)
        plt.close(fig)

        # 4. Calibration Curve Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, data in calibration_data.items():
            ax.plot(data["prob_pred"], data["prob_true"], marker="o", label=f"{name} (Brier = {data['brier']:.3f})", lw=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect Calibration")
        ax.set_title("Probability Calibration Curves (Reliability Diagram)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax.set_ylabel("Empirical Fraction of Positives", fontsize=11)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
        plt.tight_layout()
        fig.savefig(RESULT_DIR / "calibration_curve.png", dpi=200)
        plt.close(fig)

        logger.info(f"Benchmark visualization plots saved to directory: {RESULT_DIR}")

    def save_artifacts(
        self,
        pipeline_path: Path = DEFAULT_PIPELINE_PATH,
        calibrated_path: Path = DEFAULT_CALIBRATED_PIPELINE_PATH,
        metrics_path: Path = DEFAULT_METRICS_PATH
    ) -> Dict[str, Path]:
        """Serialize trained pipelines and benchmark evaluation data."""
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.production_pipeline, pipeline_path)
        joblib.dump(self.calibrated_pipeline, calibrated_path)

        # Save metrics as JSON
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.evaluation_summary, f, indent=2)

        # Save metrics as CSV
        df_bench = pd.DataFrame(self.benchmark_results)
        df_bench.to_csv(RESULT_DIR / "model_comparison.csv", index=False)

        logger.info(f"Artifacts successfully saved to:\n  - {pipeline_path}\n  - {calibrated_path}\n  - {metrics_path}")
        return {
            "pipeline": pipeline_path,
            "calibrated": calibrated_path,
            "metrics": metrics_path
        }


def load_production_pipeline(model_path: Path = DEFAULT_PIPELINE_PATH) -> Pipeline:
    """Load serialized primary production Pipeline."""
    target = Path(model_path) if model_path else DEFAULT_PIPELINE_PATH
    if not target.exists():
        raise FileNotFoundError(f"Production model pipeline not found at {target}. Run training first.")
    return joblib.load(target)


def load_calibrated_pipeline(model_path: Path = DEFAULT_CALIBRATED_PIPELINE_PATH) -> CalibratedClassifierCV:
    """Load serialized calibrated model Pipeline (or fall back to standard production pipeline)."""
    target = Path(model_path) if model_path else DEFAULT_CALIBRATED_PIPELINE_PATH
    if target.exists():
        return joblib.load(target)
    return load_production_pipeline()
