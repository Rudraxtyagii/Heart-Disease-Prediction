"""
Machine Learning Model Training, Cross-Validation, and Evaluation Pipeline.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.utils import get_dataset_path, get_model_path, find_project_root, FEATURE_NAMES


class HeartDiseaseModelPipeline:
    """End-to-end Machine Learning pipeline for Heart Disease risk classification."""

    def __init__(self, data_path: Path = None, random_state: int = 42):
        self.data_path = Path(data_path) if data_path else get_dataset_path()
        self.random_state = random_state
        self.pipeline: Pipeline = None
        self.benchmark_results: pd.DataFrame = None
        self.feature_importances: pd.Series = None
        self.metrics: Dict[str, float] = {}

    def load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load dataset, clean duplicates, and separate features/target."""
        df = pd.read_csv(self.data_path)
        # Drop duplicate records for clean validation
        df = df.drop_duplicates().reset_index(drop=True)
        
        if "target" not in df.columns:
            raise ValueError("Dataset missing 'target' column.")
            
        X = df[FEATURE_NAMES]
        y = df["target"]
        return X, y

    def benchmark_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
        """Benchmark candidate classification algorithms."""
        models = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=self.random_state))
            ]),
            "Decision Tree": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", DecisionTreeClassifier(random_state=self.random_state))
            ]),
            "Random Forest": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=100, random_state=self.random_state))
            ]),
            "KNN": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5))
            ]),
            "SVM": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, random_state=self.random_state))
            ])
        }

        records = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for name, pipe in models.items():
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            
            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            
            # Cross-validation score on whole training set
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
            
            records.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "CV_Mean_Accuracy": cv_scores.mean(),
                "CV_Std": cv_scores.std()
            })

        self.benchmark_results = pd.DataFrame(records).sort_values(by="Accuracy", ascending=False)
        return self.benchmark_results

    def train_final_pipeline(self) -> Dict[str, Any]:
        """Train the full production pipeline with StandardScaler + RandomForestClassifier."""
        X, y = self.load_and_preprocess()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state, stratify=y
        )

        # Benchmark candidate models
        self.benchmark_models(X_train, X_test, y_train, y_test)

        # Build production pipeline
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                random_state=self.random_state
            ))
        ])

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        # Metrics
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred))
        }

        # Feature importances from the RandomForest step
        rf_step: RandomForestClassifier = self.pipeline.named_steps["classifier"]
        self.feature_importances = pd.Series(
            rf_step.feature_importances_,
            index=FEATURE_NAMES
        ).sort_values(ascending=False)

        # Generate reports
        clf_report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        return {
            "metrics": self.metrics,
            "classification_report": clf_report,
            "confusion_matrix": conf_matrix,
            "benchmarks": self.benchmark_results,
            "feature_importances": self.feature_importances.to_dict()
        }

    def save_artifacts(self, model_save_path: Path = None) -> Path:
        """Persist model pipeline and update benchmark result files."""
        save_path = Path(model_save_path) if model_save_path else get_model_path()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, save_path)

        # Save benchmark and report if available
        root = find_project_root()
        result_dir = root / "minor" / "result"
        result_dir.mkdir(parents=True, exist_ok=True)

        if self.benchmark_results is not None:
            self.benchmark_results.to_csv(result_dir / "model_comparison.csv", index=False)

        return save_path


def load_trained_pipeline(model_path: Path = None) -> Pipeline:
    """Load the serialized inference pipeline."""
    target_path = Path(model_path) if model_path else get_model_path()
    if not target_path.exists():
        # Fallback to older pickle if pipeline hasn't been re-trained yet
        legacy_path = target_path.parent / "random_forest_model.pkl"
        if legacy_path.exists():
            return joblib.load(legacy_path)
        raise FileNotFoundError(f"Model file not found at {target_path}. Please run train.py first.")
    return joblib.load(target_path)
