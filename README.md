# Heart Disease Prediction System using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](#)

> **An end-to-end clinical decision support system utilizing supervised machine learning to predict coronary heart disease risk from non-invasive cardiovascular indicators.**

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset & Clinical Indicators](#dataset--clinical-indicators)
- [Machine Learning Architecture](#machine-learning-architecture)
- [Model Performance Benchmark](#model-performance-benchmark)
- [Project Structure](#project-structure)
- [Installation & Quickstart](#installation--quickstart)
- [Usage Guide](#usage-guide)
  - [1. Interactive CLI Wizard](#1-interactive-cli-wizard)
  - [2. Command-Line Inference](#2-command-line-inference)
  - [3. Web Dashboard (FastAPI)](#3-web-dashboard-fastapi)
  - [4. Model Retraining Pipeline](#4-model-retraining-pipeline)
  - [5. Automated Unit Tests](#5-automated-unit-tests)
- [Clinical Risk Heuristics](#clinical-risk-heuristics)
- [Contributing](#contributing)
- [Author & License](#author--license)

---

## Overview

Cardiovascular diseases (CVDs) remain the leading cause of mortality worldwide. Early detection of coronary artery disease enables timely intervention, potentially saving lives. This project implements a production-grade machine learning system trained on the Cleveland Heart Disease dataset.

The system combines statistical feature scaling (`StandardScaler`) with ensemble classification (`RandomForestClassifier`), validated against 5 machine learning algorithms using stratified 5-fold cross-validation. It provides a modular CLI, automated test suite, and interactive web dashboard with plain-language clinical risk breakdowns.

---

## Key Features

- **Multi-Algorithm Benchmarking:** Compares Logistic Regression, Decision Trees, Random Forest, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM).
- **Leak-Free Production Pipeline:** Preprocessing and feature scaling are encapsulated inside Scikit-Learn `Pipeline` objects to prevent data leakage during train/test splits and inference.
- **Explainable Clinical Alerts:** Translates numerical indicators into actionable medical alerts (e.g., stage 2 hypertension flags, hypercholesterolemia warnings, ischemic ST depression).
- **Interactive CLI & Web UI:** Includes both a guided terminal wizard (`predict.py`) and a modern glassmorphism web dashboard (`app.py`).
- **Comprehensive Unit Tests:** Full test suite verifying dataset integrity, pipeline execution, edge-case input validation, and model persistence.

---

## Dataset & Clinical Indicators

The dataset consists of 14 key clinical attributes (13 features + 1 binary target):

| Feature | Name | Description | Reference / Plausible Range |
| :--- | :--- | :--- | :--- |
| `age` | Age | Patient age in years | 18 - 120 |
| `sex` | Sex | Biological sex | `1` = Male, `0` = Female |
| `cp` | Chest Pain Type | Clinical classification of chest discomfort | `0`: Typical Angina, `1`: Atypical Angina, `2`: Non-Anginal, `3`: Asymptomatic |
| `trestbps` | Resting Blood Pressure | Resting blood pressure (hospital admission) | 80 - 250 mm Hg (Normal: < 120) |
| `chol` | Serum Cholesterol | Total serum cholesterol | 100 - 600 mg/dl (Desirable: < 200) |
| `fbs` | Fasting Blood Sugar | Fasting blood glucose level | `1` = > 120 mg/dl, `0` = <= 120 mg/dl |
| `restecg` | Resting ECG | Resting electrocardiographic findings | `0`: Normal, `1`: ST-T wave abnormality, `2`: LV hypertrophy |
| `thalach` | Max Heart Rate | Peak heart rate during exercise test | 60 - 230 bpm |
| `exang` | Exercise Angina | Exercise-induced angina | `1` = Yes, `0` = No |
| `oldpeak` | ST Depression | ST depression induced by exercise vs rest | 0.0 - 10.0 (Normal: < 1.0) |
| `slope` | ST Slope | Slope of the peak exercise ST segment | `0`: Upsloping, `1`: Flat, `2`: Downsloping |
| `ca` | Major Vessels | Major coronary vessels colored by fluoroscopy | 0 to 3 vessels |
| `thal` | Thalassemia | Thalassemia blood defect diagnostic status | `1`: Normal, `2`: Fixed Defect, `3`: Reversible Defect |
| **`target`** | **Diagnosis** | Cardiac outcome | `0`: Healthy / No Disease, `1`: Heart Disease Present |

---

## Machine Learning Architecture

```
Raw Patient Data / Cleveland Dataset
             │
             ▼
   [ Data Preprocessing ] ──► Drop Duplicates & Validate Ranges
             │
             ▼
   [ Stratified Splitting ] ─► 80% Train / 20% Test (Stratified K-Fold)
             │
             ▼
   [ Feature Normalization ] ─► StandardScaler (Z-Score Normalization)
             │
             ▼
   [ Model Selection ] ────► Multi-Model Evaluation (LR, DT, RF, KNN, SVM)
             │
             ▼
   [ Production Pipeline ] ──► StandardScaler + RandomForestClassifier
             │
             ▼
   [ Decision & Insights ] ──► Probability Score + Clinical Warning Breakdown
```

---

## Model Performance Benchmark

Models evaluated on the Cleveland test set (80/20 train-test split with Stratified 5-Fold Cross Validation):

| Algorithm | Test Accuracy | Precision | Recall | F1 Score | 5-Fold CV Mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (Selected) | **83.61%** | **0.7879** | **0.8966** | **0.8387** | **85.04%** |
| **Logistic Regression** | 77.05% | 0.7027 | 0.8966 | 0.7879 | 83.79% |
| **SVM** | 78.69% | 0.7500 | 0.8276 | 0.7869 | 85.04% |
| **K-Nearest Neighbors** | 73.77% | 0.6970 | 0.7931 | 0.7419 | 82.98% |
| **Decision Tree** | 73.77% | 0.7407 | 0.6897 | 0.7143 | 77.59% |

### Key Predictive Features (Feature Importance)
1. **`cp` (Chest Pain Type)** — Strongest indicator of myocardial perfusion.
2. **`thalach` (Maximum Heart Rate)** — Exercise capacity and cardiac output under stress.
3. **`ca` (Major Fluoroscopy Vessels)** — Direct visualization of anatomical lumen narrowing.
4. **`oldpeak` (ST Depression)** — Quantitative indicator of subendocardial ischemia.
5. **`thal` (Thalassemia / Perfusion Defect)** — Reversible defect status.

---

## Project Structure

```
Heart-Disease-Prediction/
├── app.py                     # FastAPI web application with interactive medical UI
├── predict.py                 # CLI inference tool (interactive, demo, and flag modes)
├── train.py                   # Automated model training and benchmarking pipeline
├── requirements.txt           # Python dependency specifications
├── .gitignore                 # Standard Python gitignore rules
├── README.md                  # Comprehensive project documentation
│
├── src/                       # Modular production source code
│   ├── __init__.py            # Package initializer
│   ├── model.py               # Pipeline definition, benchmarking, model persistence
│   ├── predict.py             # Inference handler, input validator, probability scoring
│   └── utils.py               # Feature metadata, path finders, clinical risk heuristics
│
├── tests/                     # Automated testing suite
│   ├── __init__.py
│   └── test_model.py          # Unit tests for dataset, pipeline, inference & edge-cases
│
└── minor/                     # Original project artifacts & reports
    ├── data/
    │   └── heart.csv          # Cleveland dataset
    ├── model/
    │   └── heart_disease_pipeline.pkl  # Trained production pipeline
    ├── notebook/
    │   ├── Heart_Disease_Prediction_minorProject.ipynb
    │   └── Heart_Disease_Project_Report_Final.pdf
    └── result/
        ├── classification_report.txt
        └── model_comparison.csv
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

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Interactive CLI Wizard
Launch the guided questionnaire in your terminal:
```bash
python predict.py --interactive
```

### 2. Command-Line Inference
Pass patient indicators directly using flags:
```bash
python predict.py \
  --age 58 --sex 1 --cp 0 --trestbps 140 --chol 260 --fbs 0 \
  --restecg 1 --thalach 125 --exang 1 --oldpeak 2.4 --slope 1 --ca 2 --thal 3
```

Run demo mode to test sample clinical profiles:
```bash
python predict.py --demo
```

### 3. Web Dashboard (FastAPI)
Start the modern local web interface:
```bash
python app.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your web browser.  
Interactive OpenAPI documentation is available at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

### 4. Model Retraining Pipeline
To retrain candidate algorithms, recalculate benchmarks, and re-serialize the pipeline:
```bash
python train.py
```

### 5. Automated Unit Tests
Run the test suite using Python's built-in `unittest` runner:
```bash
python -m unittest discover tests
```

---

## Clinical Risk Heuristics

In addition to machine learning classification, the inference engine evaluates American Heart Association (AHA) clinical threshold guidelines:

- **Hypertension Stage 2:** Flagged when `trestbps >= 140 mm Hg`.
- **Hypercholesterolemia:** Flagged when `chol >= 240 mg/dl`.
- **Myocardial Ischemia Alert:** Flagged when `exang == 1` (exercise-induced angina present).
- **Severe ST Depression:** Flagged when `oldpeak >= 2.0` (electrocardiographic ischemia).
- **Vascular Involvement:** Flagged when fluoroscopy reveals `>= 1` colored vessel (`ca > 0`).

---

## Contributing

Contributions, feedback, and pull requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/cardiac-enhancement`)
3. Commit your changes (`git commit -m 'Add cardiac risk feature'`)
4. Push to the branch (`git push origin feature/cardiac-enhancement`)
5. Open a Pull Request

---

## Author & License

- **Author:** Rudra Tyagi ([@Rudraxtyagii](https://github.com/Rudraxtyagii))
- **Degree:** B.Tech CSIT
- **License:** Distributed under the [MIT License](LICENSE).
