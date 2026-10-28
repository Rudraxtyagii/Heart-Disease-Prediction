"""
Explainable AI (XAI) Engine utilizing SHAP (SHapley Additive exPlanations).
Provides global dataset-level feature importances and local per-patient decision attributions.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.config import (
    FEATURE_NAMES,
    FEATURE_METADATA,
    RESULT_DIR,
    DEFAULT_PIPELINE_PATH
)
from src.dataset import DatasetHandler

logger = logging.getLogger(__name__)


class HeartDiseaseExplainer:
    """SHAP-based explanation engine for Random Forest cardiac risk models."""

    def __init__(self, pipeline=None, background_data: pd.DataFrame = None):
        if pipeline is None:
            pipeline = joblib.load(DEFAULT_PIPELINE_PATH)
        self.pipeline = pipeline
        self.scaler = pipeline.named_steps["scaler"]
        self.classifier = pipeline.named_steps["clf"]

        # Background training data for SHAP reference
        if background_data is None:
            handler = DatasetHandler()
            X_train, _, _, _, _ = handler.get_stratified_split()
            self.background_df = X_train
        else:
            self.background_df = background_data

        self.background_scaled = pd.DataFrame(
            self.scaler.transform(self.background_df),
            columns=FEATURE_NAMES
        )

        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.classifier)
        
        # Expected value (base rate) for class 1 (Heart Disease)
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            self.base_value = float(self.explainer.expected_value[1])
        else:
            self.base_value = float(self.explainer.expected_value)

    def generate_global_explanations(self) -> Dict[str, Any]:
        """
        Calculates global SHAP values across background cohort,
        saves summary plot to disk, and returns feature attribution rankings.
        """
        shap_values = self.explainer.shap_values(self.background_scaled)
        
        # Extract class 1 (disease) SHAP values
        if isinstance(shap_values, list):
            sv_class1 = shap_values[1]
        elif len(shap_values.shape) == 3:
            sv_class1 = shap_values[:, :, 1]
        else:
            sv_class1 = shap_values

        mean_abs_shap = np.abs(sv_class1).mean(axis=0)
        global_importance = pd.Series(mean_abs_shap, index=FEATURE_NAMES).sort_values(ascending=False)

        # Generate and save SHAP Beeswarm Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv_class1,
            self.background_df,
            feature_names=[FEATURE_METADATA[f]["label"] for f in FEATURE_NAMES],
            show=False
        )
        plt.title("Global SHAP Feature Attribution (Impact on Cardiac Risk)", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        summary_path = RESULT_DIR / "shap_summary.png"
        plt.savefig(summary_path, dpi=200, bbox_inches="tight")
        plt.close()

        logger.info(f"Global SHAP summary plot saved to: {summary_path}")

        return {
            "base_value": self.base_value,
            "global_importance_ranking": global_importance.to_dict(),
            "plot_saved_path": str(summary_path)
        }

    def explain_instance(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate local SHAP explanations for an individual patient instance.
        Decomposes model prediction into positive (risk-increasing) and negative
        (risk-decreasing) contributions.
        """
        df_patient = pd.DataFrame([patient_data])[FEATURE_NAMES]
        scaled_patient = pd.DataFrame(self.scaler.transform(df_patient), columns=FEATURE_NAMES)

        shap_values = self.explainer.shap_values(scaled_patient)
        
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        contributions = []
        for feat_name, shap_val in zip(FEATURE_NAMES, sv):
            val = patient_data[feat_name]
            meta = FEATURE_METADATA[feat_name]
            direction = "increases_risk" if shap_val > 0 else "decreases_risk"
            
            contributions.append({
                "feature": feat_name,
                "label": meta["label"],
                "patient_value": val,
                "unit": meta.get("unit", ""),
                "shap_value": float(round(shap_val, 4)),
                "abs_magnitude": float(round(abs(shap_val), 4)),
                "direction": direction,
                "impact": "Elevates Risk" if shap_val > 0 else "Lowers Risk"
            })

        # Sort features by absolute contribution magnitude
        contributions_sorted = sorted(contributions, key=lambda x: x["abs_magnitude"], reverse=True)
        top_increasing = [c for c in contributions_sorted if c["direction"] == "increases_risk"][:4]
        top_decreasing = [c for c in contributions_sorted if c["direction"] == "decreases_risk"][:4]

        # Plain-English narrative synthesis
        narrative = self._generate_plain_english_narrative(top_increasing, top_decreasing)

        return {
            "base_value": round(self.base_value, 4),
            "feature_contributions": contributions_sorted,
            "top_risk_drivers": top_increasing,
            "top_protective_factors": top_decreasing,
            "clinical_narrative": narrative
        }

    def _generate_plain_english_narrative(
        self,
        top_increasing: List[Dict[str, Any]],
        top_decreasing: List[Dict[str, Any]]
    ) -> str:
        """Translate mathematical SHAP attributions into clear, professional clinical prose."""
        parts = []

        if top_increasing:
            inc_phrases = [f"{c['label']} ({c['patient_value']} {c['unit']})".strip() for c in top_increasing[:3]]
            parts.append(
                f"The primary factors elevating cardiac risk above cohort baseline are: {', '.join(inc_phrases)}."
            )

        if top_decreasing:
            dec_phrases = [f"{c['label']} ({c['patient_value']} {c['unit']})".strip() for c in top_decreasing[:3]]
            parts.append(
                f"Conversely, favorable indicators mitigating overall risk score include: {', '.join(dec_phrases)}."
            )

        parts.append(
            "Note: SHAP feature attributions reflect statistical associations within the trained model "
            "and do not constitute definitive biological proof of individual medical etiology."
        )

        return " ".join(parts)
