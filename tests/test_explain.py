"""
Unit tests for Explainable AI (SHAP) module and clinical narrative generation.
"""

import unittest
from src.config import FEATURE_NAMES
from src.explain import HeartDiseaseExplainer


class TestExplainableAI(unittest.TestCase):
    """Test suite for SHAP global and local attribution calculations."""

    @classmethod
    def setUpClass(cls):
        cls.explainer = HeartDiseaseExplainer()
        cls.high_risk_patient = {
            "age": 62, "sex": 1, "cp": 0, "trestbps": 160, "chol": 290, "fbs": 1,
            "restecg": 2, "thalach": 115, "exang": 1, "oldpeak": 2.8, "slope": 1,
            "ca": 2, "thal": 3
        }

    def test_explainer_initialization_and_base_value(self):
        """Verify SHAP TreeExplainer initializes with valid base rate probability."""
        self.assertIsNotNone(self.explainer.explainer)
        self.assertGreater(self.explainer.base_value, 0.20)
        self.assertLess(self.explainer.base_value, 0.80)

    def test_local_instance_explanation_schema(self):
        """Verify instance explanation returns all required attribution fields."""
        res = self.explainer.explain_instance(self.high_risk_patient)
        self.assertIn("base_value", res)
        self.assertIn("feature_contributions", res)
        self.assertIn("top_risk_drivers", res)
        self.assertIn("top_protective_factors", res)
        self.assertIn("clinical_narrative", res)

        # 13 features must be evaluated
        self.assertEqual(len(res["feature_contributions"]), len(FEATURE_NAMES))

        # Check top driver structure
        if res["top_risk_drivers"]:
            top = res["top_risk_drivers"][0]
            self.assertIn("feature", top)
            self.assertIn("shap_value", top)
            self.assertGreater(top["shap_value"], 0.0)

    def test_clinical_narrative_generation(self):
        """Verify plain-English explanation contains medical context and disclaimer."""
        res = self.explainer.explain_instance(self.high_risk_patient)
        narrative = res["clinical_narrative"]
        self.assertIsInstance(narrative, str)
        self.assertIn("Note: SHAP feature attributions", narrative)


if __name__ == "__main__":
    unittest.main()
