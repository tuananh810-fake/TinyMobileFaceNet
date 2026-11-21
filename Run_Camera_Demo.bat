@echo off
title LSI CONTEST 2026 - REAL-TIME FACE RECOGNITION DEMO
color 0B

echo ===============================================================================
echo   REAL-TIME FACE RECOGNITION DEMO (TinyMobileFaceNet-32)
echo   Hardware Model: TinyMobileFaceNet-32 (32x32 input)
echo   Optimal Decision Threshold: 0.30
echo   Press [Q] on the camera window to EXIT.
echo ===============================================================================
echo.

set "PYTHON_EXE=C:\Users\admin\anaconda3\envs\lsi_face\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python.exe"
)

cd /d "%~dp0"

"%PYTHON_EXE%" camera_demo.py --threshold 0.30

pause
