@echo off
title Heart Disease Prediction Web Dashboard
echo ===============================================================
echo Starting Heart Disease Prediction Web Service...
echo ===============================================================
cd /d "%~dp0"
python app.py
pause
