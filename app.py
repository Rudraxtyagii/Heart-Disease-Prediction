#!/usr/bin/env python3
"""
FastAPI Medical Dashboard & REST API for Heart Disease Prediction.
Usage:
    python app.py
    # Then open http://127.0.0.1:8000 in your browser
"""

import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_patient_risk
from src.utils import FEATURE_METADATA


class PatientInput(BaseModel):
    age: int = Field(52, ge=18, le=120, description="Age in years")
    sex: int = Field(1, ge=0, le=1, description="Sex (1: Male, 0: Female)")
    cp: int = Field(0, ge=0, le=3, description="Chest Pain Type (0-3)")
    trestbps: int = Field(125, ge=80, le=250, description="Resting Blood Pressure (mm Hg)")
    chol: int = Field(212, ge=100, le=600, description="Serum Cholesterol (mg/dl)")
    fbs: int = Field(0, ge=0, le=1, description="Fasting Blood Sugar > 120 mg/dl (1: True, 0: False)")
    restecg: int = Field(1, ge=0, le=2, description="Resting ECG Results (0-2)")
    thalach: int = Field(168, ge=60, le=230, description="Max Heart Rate Achieved (bpm)")
    exang: int = Field(0, ge=0, le=1, description="Exercise Induced Angina (1: Yes, 0: No)")
    oldpeak: float = Field(1.0, ge=0.0, le=10.0, description="ST Depression")
    slope: int = Field(2, ge=0, le=2, description="Slope of Peak Exercise ST Segment (0-2)")
    ca: int = Field(2, ge=0, le=3, description="Number of Major Vessels (0-3)")
    thal: int = Field(3, ge=1, le=3, description="Thalassemia (1: Normal, 2: Fixed Defect, 3: Reversible Defect)")


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Cardiovascular disease risk assessment engine powered by Machine Learning.",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Heart Disease Prediction System"}


@app.get("/api/features")
def get_features_metadata():
    return FEATURE_METADATA


@app.post("/api/predict")
def predict_endpoint(patient: PatientInput):
    try:
        result = predict_patient_risk(patient.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CardioPredict | ML Heart Disease Risk Assessment</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(22, 30, 49, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --danger: #ef4444;
      --warning: #f59e0b;
      --success: #10b981;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
    body {
      background: radial-gradient(circle at 10% 20%, #172554 0%, #090d16 80%);
      color: var(--text);
      min-height: 100vh;
      padding: 30px 20px;
    }
    .container { max-width: 1100px; margin: 0 auto; }
    header {
      text-align: center;
      margin-bottom: 35px;
    }
    .badge-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      background: rgba(59, 130, 246, 0.15);
      border: 1px solid rgba(59, 130, 246, 0.3);
      color: #60a5fa;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 600;
      margin-bottom: 12px;
      letter-spacing: 0.03em;
    }
    h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 8px; }
    .subtitle { color: var(--text-muted); font-size: 1.05rem; }
    .main-grid {
      display: grid;
      grid-template-columns: 1.3fr 0.9fr;
      gap: 25px;
    }
    @media (max-width: 860px) { .main-grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 26px;
      backdrop-filter: blur(16px);
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    .card-title {
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .quick-actions { display: flex; gap: 8px; }
    .btn-sm {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-sm:hover { color: #fff; background: rgba(255, 255, 255, 0.12); }
    .form-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 16px;
    }
    @media (max-width: 550px) { .form-grid { grid-template-columns: 1fr; } }
    .input-group label {
      display: block;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .input-group input, .input-group select {
      width: 100%;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      padding: 10px 14px;
      color: #fff;
      font-size: 0.95rem;
      transition: border 0.2s;
    }
    .input-group input:focus, .input-group select:focus {
      outline: none;
      border-color: var(--primary);
    }
    .btn-primary {
      width: 100%;
      margin-top: 22px;
      background: linear-gradient(135deg, #2563eb, #3b82f6);
      border: none;
      color: #fff;
      font-size: 1.05rem;
      font-weight: 700;
      padding: 14px;
      border-radius: 12px;
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.15s;
    }
    .btn-primary:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 25px rgba(37, 99, 235, 0.4);
    }
    .result-box {
      text-align: center;
      padding: 20px 10px;
    }
    .risk-circle {
      width: 140px;
      height: 140px;
      border-radius: 50%;
      margin: 0 auto 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      border: 8px solid rgba(255, 255, 255, 0.08);
      position: relative;
    }
    .risk-circle.high { border-color: var(--danger); background: rgba(239, 68, 68, 0.1); color: var(--danger); }
    .risk-circle.low { border-color: var(--success); background: rgba(16, 185, 129, 0.1); color: var(--success); }
    .risk-circle .score { font-size: 2.2rem; font-weight: 800; line-height: 1; }
    .risk-circle .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; margin-top: 4px; }
    .result-badge {
      display: inline-block;
      padding: 6px 16px;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 700;
      margin-bottom: 12px;
    }
    .result-badge.high { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }
    .result-badge.low { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }
    .result-title { font-size: 1.5rem; font-weight: 800; margin-bottom: 8px; }
    .alerts-list { margin-top: 20px; text-align: left; }
    .alert-card {
      background: rgba(239, 68, 68, 0.08);
      border-left: 4px solid var(--danger);
      padding: 10px 14px;
      border-radius: 6px;
      margin-bottom: 10px;
      font-size: 0.85rem;
    }
    .alert-card strong { color: #fca5a5; display: block; margin-bottom: 2px; }
    .alert-card span { color: var(--text-muted); font-size: 0.8rem; }
    footer { text-align: center; margin-top: 40px; color: var(--text-muted); font-size: 0.85rem; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge-pill">Cardiovascular Intelligence &bull; Random Forest AI</div>
      <h1>Heart Disease Risk Predictor</h1>
      <p class="subtitle">Clinical decision support system utilizing supervised machine learning</p>
    </header>

    <div class="main-grid">
      <!-- Input Form -->
      <div class="card">
        <div class="card-title">
          <span>Patient Clinical Indicators</span>
          <div class="quick-actions">
            <button class="btn-sm" onclick="loadProfile('sample1')">Sample A</button>
            <button class="btn-sm" onclick="loadProfile('sample2')">Sample B</button>
          </div>
        </div>

        <form id="predictForm" onsubmit="handlePredict(event)">
          <div class="form-grid">
            <div class="input-group">
              <label>Age (years)</label>
              <input type="number" id="age" value="52" min="18" max="110" required>
            </div>
            <div class="input-group">
              <label>Sex</label>
              <select id="sex"><option value="1">Male</option><option value="0">Female</option></select>
            </div>
            <div class="input-group">
              <label>Chest Pain Type (cp)</label>
              <select id="cp">
                <option value="0">0: Typical Angina</option>
                <option value="1">1: Atypical Angina</option>
                <option value="2">2: Non-Anginal Pain</option>
                <option value="3">3: Asymptomatic</option>
              </select>
            </div>
            <div class="input-group">
              <label>Resting Blood Pressure (mm Hg)</label>
              <input type="number" id="trestbps" value="125" min="80" max="240" required>
            </div>
            <div class="input-group">
              <label>Serum Cholesterol (mg/dl)</label>
              <input type="number" id="chol" value="212" min="100" max="600" required>
            </div>
            <div class="input-group">
              <label>Fasting Blood Sugar > 120 mg/dl</label>
              <select id="fbs"><option value="0">No (<= 120)</option><option value="1">Yes (> 120)</option></select>
            </div>
            <div class="input-group">
              <label>Resting ECG (restecg)</label>
              <select id="restecg">
                <option value="0">0: Normal</option>
                <option value="1" selected>1: ST-T Abnormality</option>
                <option value="2">2: LV Hypertrophy</option>
              </select>
            </div>
            <div class="input-group">
              <label>Max Heart Rate Achieved (bpm)</label>
              <input type="number" id="thalach" value="168" min="60" max="230" required>
            </div>
            <div class="input-group">
              <label>Exercise Induced Angina</label>
              <select id="exang"><option value="0">No</option><option value="1">Yes</option></select>
            </div>
            <div class="input-group">
              <label>ST Depression (oldpeak)</label>
              <input type="number" step="0.1" id="oldpeak" value="1.0" min="0" max="8" required>
            </div>
            <div class="input-group">
              <label>Slope of Peak Exercise ST</label>
              <select id="slope">
                <option value="0">0: Upsloping</option>
                <option value="1">1: Flat</option>
                <option value="2" selected>2: Downsloping</option>
              </select>
            </div>
            <div class="input-group">
              <label>Major Fluoroscopy Vessels (ca)</label>
              <select id="ca">
                <option value="0">0 vessels</option>
                <option value="1">1 vessel</option>
                <option value="2" selected>2 vessels</option>
                <option value="3">3 vessels</option>
              </select>
            </div>
            <div class="input-group" style="grid-column: span 2;">
              <label>Thalassemia Status</label>
              <select id="thal">
                <option value="1">1: Normal</option>
                <option value="2">2: Fixed Defect</option>
                <option value="3" selected>3: Reversible Defect</option>
              </select>
            </div>
          </div>
          <button type="submit" class="btn-primary" id="btnSubmit">Run AI Cardiac Analysis</button>
        </form>
      </div>

      <!-- Result Card -->
      <div class="card">
        <div class="card-title">Prediction & Risk Assessment</div>
        <div class="result-box" id="resultContainer">
          <div class="risk-circle low" id="riskCircle">
            <div class="score" id="riskScore">--%</div>
            <div class="label">Risk Index</div>
          </div>
          <div class="result-badge low" id="resultBadge">READY FOR EVALUATION</div>
          <div class="result-title" id="resultTitle">Enter patient data</div>
          <p style="color: var(--text-muted); font-size: 0.9rem;" id="resultDesc">Click 'Run AI Cardiac Analysis' to generate evaluation.</p>

          <div class="alerts-list" id="alertsList"></div>
        </div>
      </div>
    </div>

    <footer>
      Developed by Rudra Tyagi &bull; Heart Disease Prediction using Supervised ML &bull; B.Tech CSIT
    </footer>
  </div>

  <script>
    const profiles = {
      sample1: { age: 58, sex: 1, cp: 0, trestbps: 140, chol: 260, fbs: 0, restecg: 1, thalach: 125, exang: 1, oldpeak: 2.4, slope: 1, ca: 2, thal: 3 },
      sample2: { age: 44, sex: 0, cp: 2, trestbps: 118, chol: 195, fbs: 0, restecg: 0, thalach: 175, exang: 0, oldpeak: 0.2, slope: 2, ca: 0, thal: 2 }
    };

    function loadProfile(key) {
      const p = profiles[key];
      for (const [k, v] of Object.entries(p)) {
        const el = document.getElementById(k);
        if (el) el.value = v;
      }
      handlePredict(new Event('submit'));
    }

    async function handlePredict(e) {
      if (e) e.preventDefault();
      const payload = {
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

      const btn = document.getElementById('btnSubmit');
      btn.innerText = 'Evaluating ML Pipeline...';
      btn.disabled = true;

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        renderResult(data);
      } catch (err) {
        alert('Prediction request failed: ' + err.message);
      } finally {
        btn.innerText = 'Run AI Cardiac Analysis';
        btn.disabled = false;
      }
    }

    function renderResult(data) {
      const isPositive = data.prediction === 1;
      const circle = document.getElementById('riskCircle');
      const badge = document.getElementById('resultBadge');
      const score = document.getElementById('riskScore');
      const title = document.getElementById('resultTitle');
      const desc = document.getElementById('resultDesc');
      const alerts = document.getElementById('alertsList');

      score.innerText = data.confidence_percentage + '%';
      title.innerText = data.label;
      desc.innerText = `Classified as ${data.risk_badge} with ${data.confidence_percentage}% confidence.`;

      if (isPositive) {
        circle.className = 'risk-circle high';
        badge.className = 'result-badge high';
      } else {
        circle.className = 'risk-circle low';
        badge.className = 'result-badge low';
      }
      badge.innerText = data.risk_badge;

      if (data.risk_factors && data.risk_factors.length > 0) {
        alerts.innerHTML = '<h4 style="margin-bottom:8px; font-size:0.9rem; color:#fca5a5;">Detected Clinical Alerts:</h4>' +
          data.risk_factors.map(a => `
            <div class="alert-card">
              <strong>${a.feature}: ${a.value} (${a.status})</strong>
              <span>${a.clinical_note}</span>
            </div>
          `).join('');
      } else {
        alerts.innerHTML = '<div style="color:var(--text-muted); font-size:0.85rem; margin-top:15px;">No abnormal cardiovascular indicators identified.</div>';
      }
    }
  </script>
</body>
</html>"""


if __name__ == "__main__":
    import socket
    port = 8000
    for arg in sys.argv:
        if arg.startswith("--port="):
            port = int(arg.split("=")[1])
    
    # Check if port is in use; auto-fallback to 8050
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            port = 8050

    print("=" * 65)
    print("  Starting Heart Disease Prediction Web Service")
    print(f"  URL: http://127.0.0.1:{port}")
    print(f"  API Docs: http://127.0.0.1:{port}/docs")
    print("=" * 65)
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
