"""
Unit tests for Machine Learning Pipeline, multi-model evaluation, and calibration.
"""

import unittest
from pathlib import Path
import pandas as pd
from src.config import FEATURE_NAMES, DEFAULT_PIPELINE_PATH, DEFAULT_CALIBRATED_PIPELINE_PATH
from src.model import (
    HeartDiseaseModelTrainer,
    load_production_pipeline,
    load_calibrated_pipeline
)


class TestModelPipeline(unittest.TestCase):
    """Test suite for pipeline training, benchmarking, and calibration."""

    @classmethod
    def setUpClass(cls):
        cls.trainer = HeartDiseaseModelTrainer()
        cls.sample_patient = {
            "age": 60, "sex": 1, "cp": 0, "trestbps": 145, "chol": 270, "fbs": 0,
            "restecg": 1, "thalach": 130, "exang": 1, "oldpeak": 2.0, "slope": 1,
            "ca": 2, "thal": 3
        }

    def test_candidate_models_structure(self):
        """Verify candidate models dictionary contains all required classifiers."""
        models = self.trainer.get_candidate_models()
        expected = ["Random Forest", "Logistic Regression", "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)", "Decision Tree"]
        for exp in expected:
            self.assertIn(exp, models)

    def test_model_serialization_and_reload(self):
        """Verify production and calibrated pipelines serialize and deserialize cleanly."""
        prod = load_production_pipeline(DEFAULT_PIPELINE_PATH)
        calib = load_calibrated_pipeline(DEFAULT_CALIBRATED_PIPELINE_PATH)
        
        self.assertIsNotNone(prod)
        self.assertIsNotNone(calib)

        df = pd.DataFrame([self.sample_patient])[FEATURE_NAMES]
        pred_prod = prod.predict(df)
        prob_prod = prod.predict_proba(df)
        self.assertEqual(len(pred_prod), 1)
        self.assertEqual(prob_prod.shape, (1, 2))

        prob_calib = calib.predict_proba(df)
        self.assertEqual(prob_calib.shape, (1, 2))
        self.assertGreaterEqual(prob_calib[0][1], 0.0)
        self.assertLessEqual(prob_calib[0][1], 1.0)


if __name__ == "__main__":
    unittest.main()
