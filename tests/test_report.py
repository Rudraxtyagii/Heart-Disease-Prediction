"""
Unit tests for ReportLab PDF clinical evaluation report generation.
"""

import unittest
from src.report import generate_patient_pdf_report


class TestReportGenerator(unittest.TestCase):
    """Test suite for PDF report rendering."""

    def setUp(self):
        self.patient = {
            "age": 55, "sex": 1, "cp": 0, "trestbps": 140, "chol": 250, "fbs": 0,
            "restecg": 1, "thalach": 145, "exang": 0, "oldpeak": 1.0, "slope": 1,
            "ca": 1, "thal": 2
        }
        self.prediction = {
            "prediction": 1,
            "label": "Elevated Risk (Heart Disease Indicated)",
            "confidence_percentage": 78.5,
            "risk_level": "Moderate Risk",
            "risk_factors": [
                {
                    "feature": "Resting Blood Pressure",
                    "value": "140 mm Hg",
                    "status": "High (Hypertension Stage 2)",
                    "clinical_note": "Elevated resting blood pressure strains arterial walls."
                }
            ]
        }
        self.explanation = {
            "top_risk_drivers": [
                {"label": "Resting Blood Pressure", "shap_value": 0.082}
            ],
            "top_protective_factors": [
                {"label": "Exercise-Induced Angina", "shap_value": -0.045}
            ],
            "clinical_narrative": "Hypertension is the primary risk driver."
        }

    def test_pdf_generation_produces_valid_bytes(self):
        """Verify report returns non-empty byte buffer matching PDF specification."""
        pdf_bytes = generate_patient_pdf_report(
            patient_data=self.patient,
            prediction_result=self.prediction,
            explanation_result=self.explanation
        )
        self.assertIsInstance(pdf_bytes, bytes)
        # PDF files must begin with header '%PDF-'
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"), "Generated file does not match PDF binary header")
        self.assertGreater(len(pdf_bytes), 2000, "Generated PDF is unexpectedly small")


if __name__ == "__main__":
    unittest.main()
