# CardioPredict: Explainable Machine Learning Decision-Support System for Heart Disease

[![CI Pipeline](https://github.com/Rudraxtyagii/Heart-Disease-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Rudraxtyagii/Heart-Disease-Prediction/actions)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP%20v0.54-green.svg)](https://shap.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **CardioPredict** is a production-grade clinical decision-support prototype utilizing supervised machine learning and game-theoretic **SHAP (SHapley Additive exPlanations)** to predict coronary heart disease risk from non-invasive physiological indicators.

---

## Table of Contents

- [Overview & Clinical Motivation](#overview--clinical-motivation)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dataset & Deduplication Audit](#dataset--deduplication-audit)
- [Clinical Feature Dictionary](#clinical-feature-dictionary)
- [Model Evaluation & Benchmarking](#model-evaluation--benchmarking)
- [Probability Calibration](#probability-calibration)
- [Explainable AI (SHAP) Methodology](#explainable-ai-shap-methodology)
- [Project Structure](#project-structure)
- [Installation & Quickstart](#installation--quickstart)
- [Usage Guide](#usage-guide)
  - [1. Web Dashboard](#1-web-dashboard)
  - [2. Interactive CLI](#2-interactive-cli)
  - [3. CLI Demonstration Mode](#3-cli-demonstration-mode)
  - [4. Model Retraining](#4-model-retraining)
  - [5. Automated Test Suite](#5-automated-test-suite)
- [REST API Reference & Examples](#rest-api-reference--examples)
- [PDF Assessment Report Generation](#pdf-assessment-report-generation)
- [Limitations & Ethical Medical Disclaimer](#limitations--ethical-medical-disclaimer)
- [Author & License](#author--license)

---

## Overview & Clinical Motivation

Cardiovascular diseases (CVDs) remain the leading cause of global mortality. While gold-standard diagnosis requires invasive coronary angiography, early non-invasive triaging based on routine intake indicators (blood pressure, exercise ECG response, lipid panels, fluoroscopy) can alert clinicians to occult myocardial ischemia.

CardioPredict translates statistical machine learning predictions into transparent, actionable clinical insights. Rather than functioning as a black-box model, it combines:
1. **Calibrated Random Forest Classification** ($80.33\%$ accuracy, $0.8842$ ROC-AUC, Brier score $0.1325$).
2. **Local and Global SHAP Attribution** to explain which factors drove a patient's risk up or down.
3. **American Heart Association (AHA) Clinical Heuristics** for hypertension, cholesterol, and ischemic ST-depression flags.
4. **Downloadable Clinical Assessment PDF Reports** generated dynamically via ReportLab.

---

## Key Features

- **Leak-Free Production Pipeline:** Preprocessing (`StandardScaler`) is fitted strictly within training folds using Scikit-Learn `Pipeline`.
- **Target Polarity Standardization:** Rigorously verifies that `1` denotes positive cardiac risk and `0` denotes healthy baseline, matching cardiovascular physiological markers.
- **Probability Calibration:** Integrates `CalibratedClassifierCV` (sigmoid scaling) to align predicted probabilities with empirical disease prevalence.
- **Explainable AI (SHAP):** Computes exact Shapley values for individual predictions, decomposing the prediction into risk-elevating and protective factors with plain-English narratives.
- **Clinical Heuristic Checks:** Evaluates American Heart Association (AHA) thresholds for Stage 2 hypertension ($\ge 140\text{ mm Hg}$), high cholesterol ($\ge 240\text{ mg/dl}$), and severe ST depression ($\ge 2.0\text{ mm}$).
- **Printable PDF Reports:** One-click export of structured evaluation summaries.
- **Comprehensive Test Suite:** 18 automated unit and integration tests covering data integrity, pipeline execution, SHAP explanations, API routes, and input validation.

---

## System Architecture

```
Patient Clinical Inputs (13 Features)
                │
                ▼
   [ Feature Range Validation ]  ──► Rejects invalid/out-of-bounds inputs
                │
                ▼
   [ Scikit-Learn Pipeline ]
     ├── StandardScaler           ──► Z-score normalization
     └── RandomForestClassifier   ──► 150 Estimators, Max Depth 6
                │
                ▼
   [ CalibratedClassifierCV ]    ──► Calibrated risk probability [0.0, 1.0]
                │
                ├───────────────────────────────────┐
                ▼                                   ▼
    [ SHAP TreeExplainer ]              [ AHA Clinical Heuristics ]
    ├── Base Rate Attribution           ├── Hypertension Stage 2
    ├── Positive Risk Drivers           ├── Hypercholesterolemia
    ├── Protective Factors              └── Ischemic ST Depression
    └── Plain-English Synthesis
                │                                   │
                └─────────────────┬─────────────────┘
                                  ▼
                [ Multi-Modal Presentation ]
                ├── Glassmorphism Web Dashboard (FastAPI)
                ├── OpenAPI REST Endpoints (/api/predict, /api/explain)
                ├── Interactive Terminal Wizard (predict.py)
                └── Downloadable PDF Clinical Report (ReportLab)
```

---

## Dataset & Deduplication Audit

The system is trained on the benchmark **Cleveland Heart Disease Dataset** ([`minor/data/heart.csv`](minor/data/heart.csv)).

### Deduplication Audit
- **Raw File Records:** 1,025 rows.
- **Duplicate Records Pruned:** 723 rows ($70.5\%$). The 1,025-row Kaggle distribution is an oversampled replication of the original 303 Cleveland patient cohort.
- **Retained Unique Patients:** 302 patients ($138$ disease-positive, $164$ disease-negative; prevalence $45.7\%$).
- **Stratified Partition:** $80\%$ training ($241$ samples, prevalence $45.6\%$), $20\%$ held-out test ($61$ samples, prevalence $45.9\%$).

---

## Clinical Feature Dictionary

The 13 clinical indicators evaluated by the model:

| Attribute | Name | Type | Medical Interpretation & Bounds | Normal / Reference Range |
| :--- | :--- | :---: | :--- | :--- |
| `age` | Age | Int | Patient biological age in years ($18 - 120$) | $18 - 120\text{ years}$ |
| `sex` | Biological Sex | Cat | Biological sex at birth (`1`: Male, `0`: Female) | Binary |
| `cp` | Chest Pain Type | Cat | `0`: Typical Angina, `1`: Atypical Angina<br>`2`: Non-Anginal, `3`: Asymptomatic | `0` strongly indicates ischemia |
| `trestbps` | Resting BP | Int | Resting blood pressure on hospital intake ($80 - 250\text{ mm Hg}$) | $< 120\text{ mm Hg}$ (Normal) |
| `chol` | Cholesterol | Int | Serum cholesterol ($100 - 600\text{ mg/dl}$) | $< 200\text{ mg/dl}$ (Desirable) |
| `fbs` | Fasting Sugar | Cat | Fasting blood sugar $> 120\text{ mg/dl}$ (`1`: True, `0`: False) | `0`: Normal fasting glucose |
| `restecg` | Resting ECG | Cat | `0`: Normal, `1`: ST-T wave abnormality<br>`2`: Left ventricular hypertrophy | `0`: Normal 12-lead ECG |
| `thalach` | Max Heart Rate | Int | Peak heart rate recorded during stress test ($60 - 230\text{ bpm}$) | Age predicted: $220 - \text{age}$ |
| `exang` | Exercise Angina | Cat | Anginal symptoms during exercise (`1`: Yes, `0`: No) | `0`: Absence of exertional pain |
| `oldpeak` | ST Depression | Float | ST depression induced by exercise vs rest ($0.0 - 10.0\text{ mm}$) | $< 1.0\text{ mm}$ (Normal) |
| `slope` | ST Slope | Cat | `0`: Upsloping, `1`: Flat, `2`: Downsloping | `0`: Benign; `2`: Marked ischemia |
| `ca` | Major Vessels | Int | Number of major coronary vessels colored by fluoroscopy ($0 - 3$) | `0`: Clear epicardial vessels |
| `thal` | Thalassemia | Cat | Perfusion defect: `1`: Normal, `2`: Fixed, `3`: Reversible | `1`: Normal myocardial perfusion |
| **`target`** | **Diagnosis** | Binary | **`1`: Heart Disease Present**, **`0`: Healthy Baseline** | Ground truth angiographic status |

---

## Model Evaluation & Benchmarking

All 5 candidate models were evaluated on the held-out test split ($61$ unique patient records) using Stratified 5-Fold Cross-Validation:

| Model | Test Accuracy | Sensitivity (Recall) | Specificity | F1 Score | ROC-AUC | 5-Fold CV Mean | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** *(Production)* | **80.33%** | **78.57%** | **81.82%** | **0.7857** | **0.8842** | **81.73% ± 0.025** | **0.1325** |
| **SVM (RBF Kernel)** | 81.97% | 78.57% | 84.85% | 0.8000 | 0.8842 | 80.90% ± 0.021 | 0.1368 |
| **K-Nearest Neighbors (KNN)** | 81.97% | 78.57% | 84.85% | 0.8000 | 0.8912 | 82.97% ± 0.041 | 0.1412 |
| **Decision Tree** | 80.33% | 78.57% | 81.82% | 0.7857 | 0.8225 | 77.13% ± 0.058 | 0.1803 |
| **Logistic Regression** | 78.69% | 75.00% | 81.82% | 0.7636 | 0.8907 | 83.38% ± 0.030 | 0.1385 |

> **Audit Note on Historical CV Scores:** In earlier unstratified splits, Random Forest and SVM exhibited identical arithmetic mean scores ($85.04\%$) due to discrete sample count overlaps across 48-patient folds ($\sum \frac{n_i}{N}$). Their standard deviations differed ($0.038$ vs. $0.045$), proving this was a mathematical coincidence of discrete integer fractions rather than a copy-paste anomaly.

Benchmark visual plots are automatically generated in [`minor/result/`](minor/result/):
- [`model_comparison.png`](minor/result/model_comparison.png) — Side-by-side metric comparison bar chart.
- [`roc_curves.png`](minor/result/roc_curves.png) — Multi-model Receiver Operating Characteristic curves.
- [`pr_curves.png`](minor/result/pr_curves.png) — Precision-Recall curves.
- [`calibration_curve.png`](minor/result/calibration_curve.png) — Reliability diagrams.

---

## Probability Calibration

A raw classification score is not an absolute clinical risk. Standard tree ensembles tend to produce probabilities pushed away from 0 and 1 or bunched in specific regions.

CardioPredict applies **Sigmoid Probability Calibration** (`CalibratedClassifierCV` with 5-fold internal splits):
- **Uncalibrated RF Brier Score:** $0.1404$
- **Calibrated RF Brier Score:** $0.1325$ (Lower Brier score indicates superior calibration alignment)

---

## Explainable AI (SHAP) Methodology

CardioPredict integrates **TreeSHAP** to decompose model predictions according to Cooperative Game Theory:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

### Global Feature Importance
The overall strongest drivers of cardiac risk across the Cleveland cohort:
1. **`ca` (Major Fluoroscopy Vessels):** Direct anatomical visualization of arterial occlusion.
2. **`cp` (Chest Pain Type):** Anginal pain characteristics.
3. **`oldpeak` (ST Depression):** Ischemic response under stress.
4. **`thal` (Thalassemia / Perfusion Defect):** Reversible myocardial ischemia.
5. **`thalach` (Maximum Heart Rate):** Impaired cardiac reserve.

A global Beeswarm plot is generated at [`minor/result/shap_summary.png`](minor/result/shap_summary.png).

---

## Project Structure

```
Heart-Disease-Prediction/
├── app.py                     # FastAPI REST API & Glassmorphism Web Dashboard
├── predict.py                 # CLI Tool (interactive wizard, demo, flag mode)
├── train.py                   # Automated retraining, benchmarking & XAI pipeline
├── requirements.txt           # Pinned production dependency configuration
├── run_app.bat                # Windows 1-click web launcher
├── run_demo.bat               # Windows 1-click CLI demo runner
├── README.md                  # Comprehensive portfolio documentation
│
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI automated testing pipeline
│
├── src/                       # Production Python Modules
│   ├── __init__.py
│   ├── config.py              # Centralized paths, metadata & clinical constraints
│   ├── dataset.py             # Data loading, deduplication & stratified splitting
│   ├── model.py               # Candidate model training, calibration & benchmarking
│   ├── explain.py             # SHAP TreeExplainer & clinical narrative synthesis
│   ├── predict.py             # Inference service, input validation & AHA heuristics
│   ├── report.py              # ReportLab multi-page clinical PDF report generator
│   └── utils.py               # Clinical heuristics & terminal styling helpers
│
├── tests/                     # Automated Test Suite (18 tests passing)
│   ├── __init__.py
│   ├── test_dataset.py        # Tests loading, deduplication, target verification
│   ├── test_model.py          # Tests pipeline serialization, probability bounds
│   ├── test_explain.py        # Tests SHAP attribution & narrative generation
│   ├── test_api.py            # Tests FastAPI routes, health, predict, 422 errors
│   └── test_report.py         # Tests PDF report binary generation
│
└── minor/
    ├── data/heart.csv         # Cleveland dataset
    ├── model/                 # Serialized production & calibrated pipelines (.pkl)
    ├── notebook/              # Exploratory data analysis notebooks
    ├── reports/               # Output directory for exported assessment reports
    └── result/                # Benchmark JSON/CSV records & visualization PNGs
```

---

## Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone Repository
```bash
git clone https://github.com/Rudraxtyagii/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Web Dashboard
Start the local healthcare analytics interface:
```bash
python app.py
```
Open **[http://127.0.0.1:8050](http://127.0.0.1:8050)** in your browser.  
Interactive OpenAPI documentation is live at **[http://127.0.0.1:8050/docs](http://127.0.0.1:8050/docs)**.

### 2. Interactive CLI Wizard
Guided questionnaire with automatic validation:
```bash
python predict.py --interactive
```

### 3. CLI Demonstration Mode
Evaluate sample contrasting clinical profiles (High-Risk, Borderline, Healthy):
```bash
python predict.py --demo
```

### 4. Model Retraining
To re-evaluate models, re-fit probability calibration, and update benchmark plots:
```bash
python train.py
```

### 5. Automated Test Suite
Run the 18 automated unit and integration tests:
```bash
python -m unittest discover tests
```

---

## REST API Reference & Examples

### Health Check
```http
GET /health
```
```json
{
  "status": "healthy",
  "service": "Heart Disease Prediction System",
  "model_version": "2.1.0",
  "pipeline_loaded": true,
  "calibration_active": true
}
```

### Explainable Risk Prediction
```http
POST /api/explain
Content-Type: application/json
```
```json
{
  "age": 62, "sex": 1, "cp": 0, "trestbps": 160, "chol": 290, "fbs": 1,
  "restecg": 2, "thalach": 115, "exang": 1, "oldpeak": 2.8, "slope": 1,
  "ca": 2, "thal": 3
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Elevated Risk (Heart Disease Indicated)",
  "calibrated_probability": 0.9600,
  "confidence_percentage": 96.0,
  "risk_level": "High Risk",
  "risk_badge": "HIGH CARDIAC RISK",
  "explanation": {
    "base_value": 0.4597,
    "top_risk_drivers": [
      {
        "feature": "ca",
        "label": "Major Vessels Colored (Fluoroscopy)",
        "patient_value": 2,
        "shap_value": 0.0868,
        "impact": "Elevates Risk"
      },
      {
        "feature": "oldpeak",
        "label": "ST Depression (oldpeak)",
        "patient_value": 2.8,
        "shap_value": 0.0840,
        "impact": "Elevates Risk"
      }
    ],
    "clinical_narrative": "The primary factors elevating cardiac risk above cohort baseline are: Major Vessels Colored (Fluoroscopy) (2 vessels), ST Depression (2.8 mm)..."
  }
}
```

---

## PDF Assessment Report Generation

To download a formal clinical summary PDF for any patient via API:
```bash
curl -X POST "http://127.0.0.1:8050/api/report/pdf" \
     -H "Content-Type: application/json" \
     -d '{"age":62,"sex":1,"cp":0,"trestbps":160,"chol":290,"fbs":1,"restecg":2,"thalach":115,"exang":1,"oldpeak":2.8,"slope":1,"ca":2,"thal":3}' \
     --output cardiac_report.pdf
```
The report includes patient indicators, calibrated risk score, AHA warning alerts, SHAP attributions, timestamp, and mandatory medical disclaimers.

---

## Limitations & Ethical Medical Disclaimer

> [!WARNING]
> **ACADEMIC DECISION-SUPPORT PROTOTYPE ONLY**
> 
> 1. **Not a Diagnostic Device:** This application is strictly an educational machine learning demonstration prototype. It has **NOT** been evaluated, certified, or cleared by the US FDA, European CE-IVD, or any healthcare regulatory body.
> 2. **Population Bias:** The model was trained exclusively on the Cleveland cohort ($N = 302$). Model generalizability to diverse demographic groups, pediatric cohorts, or modern ambulatory populations may be limited.
> 3. **Observational Association:** SHAP values represent statistical attribution within the trained algorithm, **not** definitive biological proof of individual medical etiology.
> 4. **Mandatory Clinical Oversight:** All diagnostic and treatment decisions must be made by a licensed healthcare professional based on comprehensive clinical history, physical examination, and diagnostic testing.

---

## Author & License

- **Author:** Rudra Tyagi ([@Rudraxtyagii](https://github.com/Rudraxtyagii))
- **Degree:** B.Tech CSIT
- **License:** Distributed under the [MIT License](LICENSE).
