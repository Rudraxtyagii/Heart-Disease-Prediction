"""
Integration tests for FastAPI endpoints: health, metadata, prediction, XAI, PDF, and error validation.
"""

import unittest
from fastapi.testclient import TestClient
from app import app


class TestFastAPIEndpoints(unittest.TestCase):
    """Test suite for FastAPI REST interface."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.valid_payload = {
            "age": 62, "sex": 1, "cp": 0, "trestbps": 160, "chol": 290, "fbs": 1,
            "restecg": 2, "thalach": 115, "exang": 1, "oldpeak": 2.8, "slope": 1,
            "ca": 2, "thal": 3
        }

    def test_health_endpoint(self):
        """GET /health must return 200 and healthy status."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["pipeline_loaded"])

    def test_features_metadata_endpoint(self):
        """GET /api/features must return 13 clinical indicators."""
        res = self.client.get("/api/features")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("features", data)
        self.assertEqual(len(data["feature_order"]), 13)

    def test_model_info_endpoint(self):
        """GET /api/model-info must return benchmarks and disclaimer."""
        res = self.client.get("/api/model-info")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("model_name", data)
        self.assertIn("benchmarks", data)
        self.assertIn("disclaimer", data)

    def test_predict_endpoint_success(self):
        """POST /api/predict must calculate probability and risk tier."""
        res = self.client.post("/api/predict", json=self.valid_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("prediction", data)
        self.assertIn("confidence_percentage", data)
        self.assertIn("risk_level", data)
        self.assertIn("risk_factors", data)
        self.assertGreaterEqual(data["confidence_percentage"], 0.0)
        self.assertLessEqual(data["confidence_percentage"], 100.0)

    def test_explain_endpoint_success(self):
        """POST /api/explain must return prediction with SHAP feature attributions."""
        res = self.client.post("/api/explain", json=self.valid_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("explanation", data)
        xai = data["explanation"]
        self.assertIn("feature_contributions", xai)
        self.assertIn("clinical_narrative", xai)
        self.assertEqual(len(xai["feature_contributions"]), 13)

    def test_pdf_report_endpoint_success(self):
        """POST /api/report/pdf must stream PDF file."""
        res = self.client.post("/api/report/pdf", json=self.valid_payload)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "application/pdf")
        self.assertTrue(res.content.startswith(b"%PDF-"))

    def test_validation_error_on_missing_field(self):
        """POST /api/predict must reject incomplete payload with HTTP 422."""
        incomplete = self.valid_payload.copy()
        del incomplete["age"]
        res = self.client.post("/api/predict", json=incomplete)
        self.assertEqual(res.status_code, 422)

    def test_validation_error_on_out_of_range_field(self):
        """POST /api/predict must reject physiologically implausible age with HTTP 422."""
        out_of_range = self.valid_payload.copy()
        out_of_range["age"] = 150  # Max is 120
        res = self.client.post("/api/predict", json=out_of_range)
        self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()
