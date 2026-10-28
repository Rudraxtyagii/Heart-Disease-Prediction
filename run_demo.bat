@echo off
title Heart Disease Prediction CLI Demo
echo ===============================================================
echo Running Heart Disease Risk Analysis Demo...
echo ===============================================================
cd /d "%~dp0"
python predict.py --demo
echo.
pause
