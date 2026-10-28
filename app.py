#!/usr/bin/env python3
"""
Production FastAPI Medical Web Service & REST API for Heart Disease Prediction.
Features Explainable AI (SHAP), Calibrated Cardiac Risk Probability, and PDF Reporting.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.config import (
    HOST,
    PORT,
    MODEL_VERSION,
    FEATURE_NAMES,
    FEATURE_METADATA,
    get_serializable_feature_metadata,
    CLINICAL_DISCLAIMER,
    DEFAULT_METRICS_PATH
)
from src.predict import (
    predict_patient_risk,
    validate_patient_features,
    get_inference_pipeline,
    get_calibrated_pipeline
)
from src.explain import HeartDiseaseExplainer
from src.report import generate_patient_pdf_report

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cardio_api")

app = FastAPI(
    title="Heart Disease Prediction & Explainable AI Service",
    description=(
        "Production-grade clinical decision support prototype utilizing calibrated machine learning "
        "(Random Forest) and SHAP (SHapley Additive exPlanations) for transparent cardiac risk evaluation."
    ),
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for secure integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientInputSchema(BaseModel):
    age: int = Field(..., ge=18, le=120, description="Age in years (18-120)")
    sex: int = Field(..., ge=0, le=1, description="Biological Sex (1: Male, 0: Female)")
    cp: int = Field(..., ge=0, le=3, description="Chest Pain Type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)")
    trestbps: int = Field(..., ge=80, le=250, description="Resting Blood Pressure in mm Hg (80-250)")
    chol: int = Field(..., ge=100, le=600, description="Serum Cholesterol in mg/dl (100-600)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting Blood Sugar > 120 mg/dl (1: True, 0: False)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG Results (0: Normal, 1: ST-T anomaly, 2: LV hypertrophy)")
    thalach: int = Field(..., ge=60, le=230, description="Maximum Heart Rate Achieved (60-230 bpm)")
    exang: int = Field(..., ge=0, le=1, description="Exercise-Induced Angina (1: Yes, 0: No)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST Depression (0.0 - 10.0 mm)")
    slope: int = Field(..., ge=0, le=2, description="Slope of Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)")
    ca: int = Field(..., ge=0, le=4, description="Major Coronary Vessels Colored by Fluoroscopy (0-3, 4 unknown)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia / Perfusion Defect (1: Normal, 2: Fixed, 3: Reversible)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 62,
                "sex": 1,
                "cp": 0,
                "trestbps": 160,
                "chol": 290,
                "fbs": 1,
                "restecg": 2,
                "thalach": 115,
                "exang": 1,
                "oldpeak": 2.8,
                "slope": 1,
                "ca": 2,
                "thal": 3
            }
        }
    }


# =====================================================================
# REST API Endpoints
# =====================================================================

@app.get("/health", tags=["System"])
def health_check():
    """System health verification and pipeline status."""
    try:
        pipe = get_inference_pipeline()
        calib = get_calibrated_pipeline()
        return {
            "status": "healthy",
            "service": "Heart Disease Prediction System",
            "model_version": MODEL_VERSION,
            "pipeline_loaded": pipe is not None,
            "calibration_active": calib is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "degraded", "error": str(e)}


@app.get("/api/features", tags=["Metadata"])
def get_features_metadata():
    """Retrieve full clinical metadata, reference ranges, and bounds for all 13 indicators."""
    return {
        "features": get_serializable_feature_metadata(),
        "feature_order": FEATURE_NAMES,
        "disclaimer": CLINICAL_DISCLAIMER
    }


@app.get("/api/model-info", tags=["Metadata"])
def get_model_information():
    """Returns model architecture, validation benchmarks, and training metadata."""
    import json
    metrics_data = {}
    if DEFAULT_METRICS_PATH.exists():
        try:
            with open(DEFAULT_METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read metrics file: {e}")

    return {
        "model_name": "Calibrated Random Forest Classifier",
        "version": MODEL_VERSION,
        "framework": "Scikit-Learn & SHAP",
        "dataset": "Cleveland Heart Disease Dataset (Deduplicated 302 Patients)",
        "benchmarks": metrics_data.get("benchmarks", []),
        "split_info": metrics_data.get("split_info", {}),
        "feature_importances": metrics_data.get("feature_importances", {}),
        "disclaimer": CLINICAL_DISCLAIMER
    }


@app.post("/api/predict", tags=["Inference"])
def predict_risk(patient: PatientInputSchema):
    """
    Run patient risk inference.
    Returns diagnostic class, calibrated probability, risk tier, and AHA heuristic flags.
    """
    try:
        result = predict_patient_risk(patient.model_dump(), include_explanation=False)
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err))
    except Exception as err:
        logger.error(f"Prediction failed: {err}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred during inference.")


@app.post("/api/explain", tags=["Inference"])
def explain_risk(patient: PatientInputSchema):
    """
    Run patient risk inference with Explainable AI.
    Returns prediction plus local SHAP feature attributions and a plain-English clinical narrative.
    """
    try:
        result = predict_patient_risk(patient.model_dump(), include_explanation=True)
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err))
    except Exception as err:
        logger.error(f"Explanation failed: {err}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred during explanation generation.")


@app.post("/api/report/pdf", tags=["Reports"])
def download_pdf_report(patient: PatientInputSchema):
    """Generate and stream a downloadable multi-section clinical assessment PDF report."""
    try:
        patient_dict = patient.model_dump()
        result = predict_patient_risk(patient_dict, include_explanation=True)
        explanation = result.get("explanation", {})

        pdf_bytes = generate_patient_pdf_report(
            patient_data=patient_dict,
            prediction_result=result,
            explanation_result=explanation
        )

        filename = f"Cardiac_Risk_Report_{patient.age}yo_{patient.sex}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as err:
        logger.error(f"PDF generation failed: {err}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate clinical PDF report.")


# =====================================================================
# Integrated Healthcare Analytics Web Dashboard
# =====================================================================

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def serve_dashboard():
    """Serves the complete responsive medical analytics dashboard UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CardioPredict AI | Clinical Decision Support & Explainable AI</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b1120;
      --card-bg: rgba(15, 23, 42, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --danger: #ef4444;
      --warning: #f59e0b;
      --success: #10b981;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
    body {
      background: radial-gradient(circle at 15% 15%, #1e293b 0%, #0b1120 100%);
      color: var(--text);
      min-height: 100vh;
      padding: 30px 20px 60px;
    }
    .container { max-width: 1220px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 30px; }
    .badge-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      background: rgba(59, 130, 246, 0.12);
      border: 1px solid rgba(59, 130, 246, 0.25);
      color: #93c5fd;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 600;
      margin-bottom: 12px;
      letter-spacing: 0.04em;
    }
    h1 { font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 8px; }
    .subtitle { color: var(--text-muted); font-size: 1.0rem; max-width: 700px; margin: 0 auto; line-height: 1.5; }

    .main-grid {
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 25px;
    }
    @media (max-width: 960px) { .main-grid { grid-template-columns: 1fr; } }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 26px;
      backdrop-filter: blur(16px);
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
    }
    .card-title {
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .quick-actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .btn-sm {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 0.78rem;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.2s;
    }
    .btn-sm:hover { color: #fff; background: rgba(59, 130, 246, 0.25); border-color: rgba(59, 130, 246, 0.4); }

    .form-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 14px;
    }
    @media (max-width: 580px) { .form-grid { grid-template-columns: 1fr; } }

    .input-group label {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 5px;
    }
    .input-group label span.unit { color: #64748b; font-weight: 400; }
    .input-group input, .input-group select {
      width: 100%;
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      padding: 9px 12px;
      color: #fff;
      font-size: 0.9rem;
      transition: border 0.2s;
    }
    .input-group input:focus, .input-group select:focus {
      outline: none;
      border-color: var(--primary);
    }
    .full-width { grid-column: span 2; }
    @media (max-width: 580px) { .full-width { grid-column: span 1; } }

    .btn-primary {
      width: 100%;
      margin-top: 18px;
      background: linear-gradient(135deg, #2563eb, #3b82f6);
      border: none;
      color: #fff;
      font-size: 1.0rem;
      font-weight: 700;
      padding: 13px;
      border-radius: 12px;
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.15s;
    }
    .btn-primary:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 25px rgba(37, 99, 235, 0.4);
    }

    /* Result Panel */
    .result-box { text-align: center; }
    .risk-circle {
      width: 130px;
      height: 130px;
      border-radius: 50%;
      margin: 0 auto 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      border: 6px solid rgba(255, 255, 255, 0.08);
      position: relative;
      transition: all 0.3s;
    }
    .risk-circle.high { border-color: var(--danger); background: rgba(239, 68, 68, 0.12); color: var(--danger); }
    .risk-circle.low { border-color: var(--success); background: rgba(16, 185, 129, 0.12); color: var(--success); }
    .risk-circle .score { font-size: 2.1rem; font-weight: 800; line-height: 1; }
    .risk-circle .label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700; margin-top: 4px; }

    .result-badge {
      display: inline-block;
      padding: 5px 14px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 700;
      margin-bottom: 10px;
      letter-spacing: 0.04em;
    }
    .result-badge.high { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }
    .result-badge.low { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }

    .result-title { font-size: 1.4rem; font-weight: 800; margin-bottom: 6px; }
    .result-desc { color: var(--text-muted); font-size: 0.88rem; line-height: 1.4; margin-bottom: 16px; }

    /* Explainable AI Bar chart */
    .shap-section {
      text-align: left;
      margin-top: 20px;
      border-top: 1px solid var(--card-border);
      padding-top: 16px;
    }
    .shap-title { font-size: 0.92rem; font-weight: 700; margin-bottom: 8px; color: #cbd5e1; }
    .shap-bar-row {
      display: flex;
      align-items: center;
      margin-bottom: 8px;
      font-size: 0.78rem;
    }
    .shap-label { width: 150px; font-weight: 600; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .shap-bar-container { flex: 1; height: 16px; background: rgba(255, 255, 255, 0.05); border-radius: 4px; position: relative; margin: 0 10px; }
    .shap-bar { height: 100%; border-radius: 4px; transition: width 0.3s; }
    .shap-bar.inc { background: linear-gradient(90deg, #f87171, #ef4444); }
    .shap-bar.dec { background: linear-gradient(90deg, #34d399, #10b981); }
    .shap-val { width: 45px; text-align: right; font-family: var(--mono); font-size: 0.75rem; }

    .narrative-box {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 0.82rem;
      color: #cbd5e1;
      line-height: 1.45;
      margin-top: 12px;
    }

    /* Alerts */
    .alerts-list { margin-top: 16px; text-align: left; }
    .alert-card {
      background: rgba(239, 68, 68, 0.08);
      border-left: 3px solid var(--danger);
      padding: 8px 12px;
      border-radius: 6px;
      margin-bottom: 8px;
      font-size: 0.82rem;
    }
    .alert-card strong { color: #fca5a5; display: block; margin-bottom: 2px; }
    .alert-card span { color: var(--text-muted); font-size: 0.75rem; }

    .btn-download {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      width: 100%;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 10px;
      border-radius: 10px;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
      margin-top: 18px;
      transition: all 0.2s;
    }
    .btn-download:hover { background: rgba(59, 130, 246, 0.2); border-color: var(--primary); }

    .disclaimer-banner {
      margin-top: 35px;
      padding: 14px 18px;
      background: rgba(245, 158, 11, 0.08);
      border: 1px solid rgba(245, 158, 11, 0.25);
      border-radius: 12px;
      font-size: 0.78rem;
      color: #fde68a;
      line-height: 1.45;
      text-align: center;
    }
    footer { text-align: center; margin-top: 25px; color: var(--text-muted); font-size: 0.82rem; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge-pill">&bull; Calibrated Random Forest &bull; SHAP Explainable AI</div>
      <h1>CardioPredict Analytics Dashboard</h1>
      <p class="subtitle">Clinical decision support system utilizing supervised machine learning and game-theoretic SHAP attributions for cardiac risk assessment.</p>
    </header>

    <div class="main-grid">
      <!-- Input Parameters -->
      <div class="card">
        <div class="card-title">
          <span>Patient Clinical Profile</span>
          <div class="quick-actions">
            <button type="button" class="btn-sm" onclick="loadSample('high')">High Risk Demo</button>
            <button type="button" class="btn-sm" onclick="loadSample('border')">Borderline Demo</button>
            <button type="button" class="btn-sm" onclick="loadSample('low')">Healthy Demo</button>
          </div>
        </div>

        <form id="cardioForm" onsubmit="handleAnalyze(event)">
          <div class="form-grid">
            <div class="input-group">
              <label>Age <span class="unit">years (18-120)</span></label>
              <input type="number" id="age" value="55" min="18" max="120" required>
            </div>
            <div class="input-group">
              <label>Biological Sex <span class="unit">at birth</span></label>
              <select id="sex"><option value="1">1: Male</option><option value="0">0: Female</option></select>
            </div>
            <div class="input-group full-width">
              <label>Chest Pain Symptom Type (cp)</label>
              <select id="cp">
                <option value="0">0: Typical Angina (Exertional discomfort relieved by rest)</option>
                <option value="1">1: Atypical Angina (Non-exertional presentation)</option>
                <option value="2">2: Non-Anginal Pain (Musculoskeletal origin)</option>
                <option value="3">3: Asymptomatic (No typical anginal discomfort)</option>
              </select>
            </div>
            <div class="input-group">
              <label>Resting Blood Pressure <span class="unit">mm Hg (80-250)</span></label>
              <input type="number" id="trestbps" value="130" min="80" max="250" required>
            </div>
            <div class="input-group">
              <label>Serum Cholesterol <span class="unit">mg/dl (100-600)</span></label>
              <input type="number" id="chol" value="240" min="100" max="600" required>
            </div>
            <div class="input-group">
              <label>Fasting Blood Sugar > 120 <span class="unit">diabetic marker</span></label>
              <select id="fbs"><option value="0">0: No (<= 120 mg/dl)</option><option value="1">1: Yes (> 120 mg/dl)</option></select>
            </div>
            <div class="input-group">
              <label>Resting ECG Result</label>
              <select id="restecg">
                <option value="0">0: Normal</option>
                <option value="1" selected>1: ST-T Wave Abnormality</option>
                <option value="2">2: LV Hypertrophy</option>
              </select>
            </div>
            <div class="input-group">
              <label>Max Heart Rate Achieved <span class="unit">bpm (60-230)</span></label>
              <input type="number" id="thalach" value="150" min="60" max="230" required>
            </div>
            <div class="input-group">
              <label>Exercise-Induced Angina</label>
              <select id="exang"><option value="0">0: No</option><option value="1">1: Yes</option></select>
            </div>
            <div class="input-group">
              <label>ST Depression (oldpeak) <span class="unit">mm (0.0-10.0)</span></label>
              <input type="number" step="0.1" id="oldpeak" value="1.0" min="0.0" max="10.0" required>
            </div>
            <div class="input-group">
              <label>Slope of Peak Exercise ST</label>
              <select id="slope">
                <option value="0">0: Upsloping</option>
                <option value="1" selected>1: Flat</option>
                <option value="2">2: Downsloping (High Risk)</option>
              </select>
            </div>
            <div class="input-group">
              <label>Fluoroscopy Major Vessels (ca)</label>
              <select id="ca">
                <option value="0">0: None colored</option>
                <option value="1">1: 1 vessel</option>
                <option value="2">2: 2 vessels</option>
                <option value="3">3: 3 vessels</option>
              </select>
            </div>
            <div class="input-group">
              <label>Thalassemia Perfusion Defect</label>
              <select id="thal">
                <option value="1">1: Normal Perfusion</option>
                <option value="2" selected>2: Fixed Defect</option>
                <option value="3">3: Reversible Defect (Ischemia)</option>
              </select>
            </div>
          </div>
          <button type="submit" class="btn-primary" id="btnAnalyze">Run AI Risk & Explanation Analysis</button>
        </form>
      </div>

      <!-- Results & Explainable AI -->
      <div class="card">
        <div class="card-title">Diagnostic & Explainable Output</div>
        <div class="result-box" id="resultContainer">
          <div class="risk-circle low" id="riskCircle">
            <div class="score" id="riskScore">--%</div>
            <div class="label">Calibrated Risk</div>
          </div>
          <div class="result-badge low" id="resultBadge">READY FOR EVALUATION</div>
          <div class="result-title" id="resultTitle">Awaiting Assessment</div>
          <p class="result-desc" id="resultDesc">Submit clinical profile to calculate calibrated cardiac risk and SHAP attribution values.</p>

          <!-- SHAP Feature Contributions -->
          <div class="shap-section" id="shapSection" style="display: none;">
            <div class="shap-title">Explainable AI: Key Contributing Clinical Factors (SHAP)</div>
            <div id="shapBars"></div>
            <div class="narrative-box" id="narrativeBox"></div>
          </div>

          <!-- Clinical Flags -->
          <div class="alerts-list" id="alertsList"></div>

          <!-- PDF Download -->
          <button type="button" class="btn-download" id="btnPdf" onclick="downloadPdf()" style="display: none;">
            Download Clinical Assessment Report (PDF)
          </button>
        </div>
      </div>
    </div>

    <div class="disclaimer-banner">
      <strong>RESEARCH & EDUCATIONAL DISCLAIMER:</strong> This software is an academic machine learning prototype. It is not an FDA/CE cleared medical diagnostic device and must not be used for definitive medical diagnoses or clinical triage. Always consult a licensed cardiologist for medical care.
    </div>

    <footer>
      Rudra Tyagi &bull; Heart Disease Prediction &bull; Scikit-Learn 1.8 &bull; SHAP 0.54 &bull; FastAPI
    </footer>
  </div>

  <script>
    const samples = {
      high: { age: 62, sex: 1, cp: 0, trestbps: 160, chol: 290, fbs: 1, restecg: 2, thalach: 115, exang: 1, oldpeak: 2.8, slope: 1, ca: 2, thal: 3 },
      border: { age: 56, sex: 0, cp: 1, trestbps: 135, chol: 235, fbs: 0, restecg: 1, thalach: 142, exang: 0, oldpeak: 1.2, slope: 1, ca: 1, thal: 2 },
      low: { age: 41, sex: 0, cp: 2, trestbps: 118, chol: 195, fbs: 0, restecg: 0, thalach: 172, exang: 0, oldpeak: 0.2, slope: 2, ca: 0, thal: 2 }
    };

    let lastPayload = null;

    function loadSample(key) {
      const s = samples[key];
      for (const [k, v] of Object.entries(s)) {
        const el = document.getElementById(k);
        if (el) el.value = v;
      }
      handleAnalyze(new Event('submit'));
    }

    function getPayload() {
      return {
        age: parseInt(document.getElementById('age').value),
        sex: parseInt(document.getElementById('sex').value),
        cp: parseInt(document.getElementById('cp').value),
        trestbps: parseInt(document.getElementById('trestbps').value),
        chol: parseInt(document.getElementById('chol').value),
        fbs: parseInt(document.getElementById('fbs').value),
        restecg: parseInt(document.getElementById('restecg').value),
        thalach: parseInt(document.getElementById('thalach').value),
        exang: parseInt(document.getElementById('exang').value),
        oldpeak: parseFloat(document.getElementById('oldpeak').value),
        slope: parseInt(document.getElementById('slope').value),
        ca: parseInt(document.getElementById('ca').value),
        thal: parseInt(document.getElementById('thal').value)
      };
    }

    async function handleAnalyze(e) {
      if (e) e.preventDefault();
      const payload = getPayload();
      lastPayload = payload;

      const btn = document.getElementById('btnAnalyze');
      btn.innerText = 'Calculating Calibrated ML & SHAP Values...';
      btn.disabled = true;

      try {
        const res = await fetch('/api/explain', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.detail || 'Prediction failed');
        }

        const data = await res.json();
        renderResult(data);
      } catch (err) {
        alert('Assessment Error: ' + err.message);
      } finally {
        btn.innerText = 'Run AI Risk & Explanation Analysis';
        btn.disabled = false;
      }
    }

    function renderResult(data) {
      const isPos = data.prediction === 1;
      const circle = document.getElementById('riskCircle');
      const badge = document.getElementById('resultBadge');
      const score = document.getElementById('riskScore');
      const title = document.getElementById('resultTitle');
      const desc = document.getElementById('resultDesc');
      const shapSec = document.getElementById('shapSection');
      const shapBars = document.getElementById('shapBars');
      const narrativeBox = document.getElementById('narrativeBox');
      const alerts = document.getElementById('alertsList');
      const btnPdf = document.getElementById('btnPdf');

      score.innerText = data.confidence_percentage + '%';
      title.innerText = data.label;
      desc.innerText = `Calibrated disease probability: ${data.confidence_percentage}% (Ensemble raw: ${(data.raw_probability * 100).toFixed(1)}%). Assigned tier: ${data.risk_badge}.`;

      if (isPos) {
        circle.className = 'risk-circle high';
        badge.className = 'result-badge high';
      } else {
        circle.className = 'risk-circle low';
        badge.className = 'result-badge low';
      }
      badge.innerText = data.risk_badge;

      // Render SHAP features
      if (data.explanation && data.explanation.feature_contributions) {
        shapSec.style.display = 'block';
        const topFeats = data.explanation.feature_contributions.slice(0, 5);
        const maxVal = Math.max(...topFeats.map(f => f.abs_magnitude), 0.05);

        shapBars.innerHTML = topFeats.map(f => {
          const widthPct = Math.min((f.abs_magnitude / maxVal) * 100, 100);
          const isInc = f.direction === 'increases_risk';
          const sign = isInc ? '+' : '-';
          return `
            <div class="shap-bar-row">
              <div class="shap-label" title="${f.label}">${f.label}</div>
              <div class="shap-bar-container">
                <div class="shap-bar ${isInc ? 'inc' : 'dec'}" style="width: ${widthPct}%;"></div>
              </div>
              <div class="shap-val" style="color: ${isInc ? '#f87171' : '#34d399'};">${sign}${f.abs_magnitude.toFixed(3)}</div>
            </div>
          `;
        }).join('');

        narrativeBox.innerText = data.explanation.clinical_narrative;
      }

      // Clinical alerts
      if (data.risk_factors && data.risk_factors.length > 0) {
        alerts.innerHTML = '<h4 style="margin: 12px 0 6px; font-size: 0.85rem; color: #fca5a5; text-align: left;">Detected Clinical Risk Alerts:</h4>' +
          data.risk_factors.map(a => `
            <div class="alert-card">
              <strong>${a.feature}: ${a.value} (${a.status})</strong>
              <span>${a.clinical_note}</span>
            </div>
          `).join('');
      } else {
        alerts.innerHTML = '<div style="color:var(--text-muted); font-size:0.8rem; margin-top:10px;">No abnormal threshold alerts identified.</div>';
      }

      btnPdf.style.display = 'inline-flex';
    }

    async function downloadPdf() {
      if (!lastPayload) return;
      const btn = document.getElementById('btnPdf');
      btn.innerText = 'Generating Clinical Assessment PDF...';
      btn.disabled = true;

      try {
        const res = await fetch('/api/report/pdf', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(lastPayload)
        });

        if (!res.ok) throw new Error('PDF generation failed');

        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Cardiac_Risk_Report_${lastPayload.age}yo.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
      } catch (err) {
        alert('PDF Download Error: ' + err.message);
      } finally {
        btn.innerText = 'Download Clinical Assessment Report (PDF)';
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>"""


if __name__ == "__main__":
    import socket
    port = PORT
    for arg in sys.argv:
        if arg.startswith("--port="):
            port = int(arg.split("=")[1])

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0 and port == 8000:
            port = 8050

    print("=" * 70)
    print("  Starting CardioPredict Healthcare Analytics Web Service")
    print(f"  URL: http://{HOST}:{port}")
    print(f"  API Documentation: http://{HOST}:{port}/docs")
    print("=" * 70)
    uvicorn.run("app:app", host=HOST, port=port, reload=False)
