@echo off
title TINY-MOBILEFACENET GPU TRAINING CONTROL CENTER
color 0A

cd /d "%~dp0"

:: Auto-detect Python Environment
set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if "%PYTHON_EXE%"=="" if exist "C:\Users\admin\anaconda3\envs\lsi_face\python.exe" set "PYTHON_EXE=C:\Users\admin\anaconda3\envs\lsi_face\python.exe"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python.exe"

:: Auto-detect LFW Dataset
set "DATA_OPT="
if exist "lfw_dataset" (
    set "DATA_OPT=--data-root lfw_dataset"
) else if exist "..\Python\lfw_dataset" (
    set "DATA_OPT=--data-root ..\Python\lfw_dataset"
) else if exist "dataset\lfw" (
    set "DATA_OPT=--data-root dataset\lfw"
)

:: Auto-detect Teacher Model
set "TEACHER_OPT="
if exist "weights\mobilefacenet_scripted.pt" (
    set "TEACHER_OPT=--teacher weights\mobilefacenet_scripted.pt"
) else if exist "..\Python\mobilefacenet_scripted.pt" (
    set "TEACHER_OPT=--teacher ..\Python\mobilefacenet_scripted.pt"
)

:MENU
cls
echo ===============================================================================
echo   TINY-MOBILEFACENET GPU TRAINING AND BENCHMARK CONTROL CENTER
echo   Author: Nguyen Sy Tuan Anh
echo ===============================================================================
echo   Python executable : %PYTHON_EXE%
if not "%DATA_OPT%"=="" echo   Dataset location  : %DATA_OPT%
echo.
echo   CHOOSE AN ACTION:
echo.
echo   [1] Standard Train (100 Epochs, Batch 128)
echo   [2] Custom Epochs (Set number of epochs, default 150)
echo   [3] Time Limit Train (60, 120, 180 minutes...)
echo   [4] Infinite Loop Train (50-epoch Cosine cycles, auto-eval every 50 ep, Ctrl+C to stop)
echo   [5] Quick LFW Benchmark (10-Fold Evaluation on 6,000 pairs)
echo   [6] Run Real-Time Webcam Demo (camera_demo.py)
echo   [7] View Training History and Reports
echo   [8] Exit
echo.
echo ===============================================================================
set /p CHOICE="Enter your choice [1-8]: "

if "%CHOICE%"=="1" goto QUICK_TRAIN
if "%CHOICE%"=="2" goto CUSTOM_EPOCHS
if "%CHOICE%"=="3" goto TIME_LIMIT
if "%CHOICE%"=="4" goto INFINITE_TRAIN
if "%CHOICE%"=="5" goto EVAL_ONLY
if "%CHOICE%"=="6" goto RUN_CAMERA
if "%CHOICE%"=="7" goto VIEW_HISTORY
if "%CHOICE%"=="8" goto EXIT

echo Invalid choice. Please enter a number from 1 to 8.
ping -n 3 127.0.0.1 >nul
goto MENU

:QUICK_TRAIN
echo.
echo [+] Starting Standard Training (100 Epochs, Batch Size 128)...
echo.
"%PYTHON_EXE%" train_lfw_distill.py --epochs 100 --batch-size 128 %DATA_OPT% %TEACHER_OPT%
goto FINISH

:CUSTOM_EPOCHS
echo.
set /p NUM_EPOCHS="Enter number of epochs (e.g. 100, 150, 300): "
if "%NUM_EPOCHS%"=="" set NUM_EPOCHS=150
echo [+] Starting Training for %NUM_EPOCHS% Epochs (Batch Size 128)...
echo.
"%PYTHON_EXE%" train_lfw_distill.py --epochs %NUM_EPOCHS% --batch-size 128 %DATA_OPT% %TEACHER_OPT%
goto FINISH

:TIME_LIMIT
echo.
set /p NUM_MINUTES="Enter training duration in minutes (e.g. 60, 120, 180): "
if "%NUM_MINUTES%"=="" set NUM_MINUTES=60
echo [+] Starting Time-Limited Training (%NUM_MINUTES% minutes, Batch Size 128)...
echo.
"%PYTHON_EXE%" train_lfw_distill.py --time-limit %NUM_MINUTES% --batch-size 128 %DATA_OPT% %TEACHER_OPT%
goto FINISH

:INFINITE_TRAIN
echo.
echo ===============================================================================
echo   [INFINITE TRAINING MODE ACTIVATED - 50-EPOCH COSINE CYCLES]
echo   The model will continuously train epoch after epoch (auto-eval every 50 ep).
echo   Press CTRL+C anytime in this window to stop.
echo   The system will automatically catch Ctrl+C, save the best weights,
echo   and generate the complete LFW benchmark report!
echo ===============================================================================
echo.
"%PYTHON_EXE%" train_lfw_distill.py --infinite --batch-size 128 --eval-every 50 %DATA_OPT% %TEACHER_OPT%
goto FINISH

:EVAL_ONLY
echo.
echo [+] Running 10-Fold LFW Evaluation Benchmark on current weights...
echo.
"%PYTHON_EXE%" train_lfw_distill.py --eval-only %DATA_OPT%
goto FINISH

:RUN_CAMERA
echo.
echo [+] Starting Webcam Demo...
echo.
"%PYTHON_EXE%" camera_demo.py
goto FINISH

:VIEW_HISTORY
echo.
if exist "training_history.csv" (
    echo [+] Opening latest training_history.csv...
    start "" "training_history.csv"
) else if exist "docs\training_history.csv" (
    echo [+] Opening docs\training_history.csv...
    start "" "docs\training_history.csv"
) else (
    echo [!] training_history.csv not found. Run a training session first!
)
if exist "reports" (
    echo [+] Opening reports folder (all timestamped charts)...
    start "" "reports"
)
if exist "lfw_benchmark_report.png" (
    echo [+] Opening latest lfw_benchmark_report.png...
    start "" "lfw_benchmark_report.png"
) else if exist "docs\lfw_benchmark_report.png" (
    start "" "docs\lfw_benchmark_report.png"
)
echo.
pause
goto MENU

:FINISH
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===============================================================================
    echo   [SUCCESS] Operation completed!
    echo   - Checkpoint weights  : weights\tiny_mobilefacenet_lfw.pt
    echo   - Latest report image : lfw_benchmark_report.png
    echo   - Historical charts   : reports\lfw_report_YYYYMMDD_HHMMSS.png
    echo   - Full history log    : training_history.csv
    echo ===============================================================================
    if exist "lfw_benchmark_report.png" (
        start "" "lfw_benchmark_report.png"
    ) else if exist "docs\lfw_benchmark_report.png" (
        start "" "docs\lfw_benchmark_report.png"
    )
) else (
    echo.
    echo [!] Process exited with code %ERRORLEVEL%.
)
echo.
pause
goto MENU

:EXIT
echo Exiting...
exit /b
