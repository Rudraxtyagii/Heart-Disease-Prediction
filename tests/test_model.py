"""
Unit test suite verifying dataset loading, ML training pipeline, inference, and error handling.
"""

import unittest
import pandas as pd
from pathlib import Path

from src.utils import get_dataset_path, get_model_path, FEATURE_NAMES
from src.model import HeartDiseaseModelPipeline, load_trained_pipeline
from src.predict import validate_patient_input, predict_patient_risk


class TestHeartDiseaseSystem(unittest.TestCase):
    """Test cases for Heart Disease ML system."""

    @classmethod
    def setUpClass(cls):
        cls.valid_patient = {
            "age": 52,
            "sex": 1,
            "cp": 1,
            "trestbps": 125,
            "chol": 212,
            "fbs": 0,
            "restecg": 1,
            "thalach": 168,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 2,
            "ca": 2,
            "thal": 3
        }

    def test_dataset_integrity(self):
        """Verify dataset file exists and has correct columns and rows."""
        path = get_dataset_path()
        self.assertTrue(path.exists(), f"Dataset file not found at {path}")
        df = pd.read_csv(path)
        self.assertGreaterEqual(len(df), 300, "Dataset contains insufficient rows")
        for col in FEATURE_NAMES + ["target"]:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

    def test_training_pipeline_execution(self):
        """Test that model training runs end-to-end and satisfies baseline performance."""
        trainer = HeartDiseaseModelPipeline()
        res = trainer.train_final_pipeline()
        
        self.assertIn("metrics", res)
        self.assertGreater(res["metrics"]["accuracy"], 0.70, "Accuracy below 70% threshold")
        self.assertIn("benchmarks", res)
        self.assertGreaterEqual(len(res["benchmarks"]), 4, "Benchmarks should contain candidate models")
        self.assertEqual(len(res["feature_importances"]), len(FEATURE_NAMES))

    def test_prediction_output_schema(self):
        """Verify inference returns the expected schema, types, and bounded probabilities."""
        pred_res = predict_patient_risk(self.valid_patient)
        self.assertIn(pred_res["prediction"], [0, 1])
        self.assertIsInstance(pred_res["label"], str)
        self.assertIsInstance(pred_res["risk_badge"], str)
        self.assertGreaterEqual(pred_res["disease_probability"], 0.0)
        self.assertLessEqual(pred_res["disease_probability"], 1.0)
        self.assertIsInstance(pred_res["risk_factors"], list)

    def test_input_validation_errors(self):
        """Verify validator raises ValueError on missing features and invalid ranges."""
        # Missing feature
        incomplete = self.valid_patient.copy()
        del incomplete["age"]
        with self.assertRaises(ValueError):
            validate_patient_input(incomplete)

        # Invalid non-numeric value
        bad_type = self.valid_patient.copy()
        bad_type["chol"] = "invalid_string"
        with self.assertRaises(ValueError):
            validate_patient_input(bad_type)

        # Out-of-bounds age
        bad_age = self.valid_patient.copy()
        bad_age["age"] = -5
        with self.assertRaises(ValueError):
            validate_patient_input(bad_age)

    def test_model_artifact_load(self):
        """Verify serialized model pipeline can be successfully loaded and queried."""
        pipe = load_trained_pipeline()
        self.assertIsNotNone(pipe)
        df_sample = pd.DataFrame([self.valid_patient])[FEATURE_NAMES]
        preds = pipe.predict(df_sample)
        self.assertEqual(len(preds), 1)


if __name__ == "__main__":
    unittest.main()
