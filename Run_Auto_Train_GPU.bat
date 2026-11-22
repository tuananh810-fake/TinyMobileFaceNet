@echo off
chcp 65001 >nul
title TINY-MOBILEFACENET GPU TRAINING CONTROL CENTER
color 0A

cd /d "%~dp0"

:: Tự động nhận diện môi trường Python
set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if "%PYTHON_EXE%"=="" if exist "C:\Users\admin\anaconda3\envs\lsi_face\python.exe" set "PYTHON_EXE=C:\Users\admin\anaconda3\envs\lsi_face\python.exe"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python.exe"

:: Tự động nhận diện đường dẫn Dataset LFW
set "DATA_OPT="
if exist "lfw_dataset" (
    set "DATA_OPT=--data-root lfw_dataset"
) else if exist "..\Python\lfw_dataset" (
    set "DATA_OPT=--data-root ..\Python\lfw_dataset"
) else if exist "dataset\lfw" (
    set "DATA_OPT=--data-root dataset\lfw"
)

:: Tự động nhận diện đường dẫn Teacher Model
set "TEACHER_OPT="
if exist "weights\mobilefacenet_scripted.pt" (
    set "TEACHER_OPT=--teacher weights\mobilefacenet_scripted.pt"
) else if exist "..\Python\mobilefacenet_scripted.pt" (
    set "TEACHER_OPT=--teacher ..\Python\mobilefacenet_scripted.pt"
)

:MENU
cls
echo ===============================================================================
echo   TINY-MOBILEFACENET GPU TRAINING & BENCHMARK CONTROL CENTER
echo   Tác giả: Nguyễn Sỹ Tuấn Anh
echo ===============================================================================
echo   Python executable : %PYTHON_EXE%
if not "%DATA_OPT%"=="" echo   Dataset location  : %DATA_OPT%
echo.
echo   CHỌN CHỨC NĂNG:
echo.
echo   [1] Huấn luyện nhanh (Quick Train - 30 Epochs)
echo   [2] Tùy chỉnh số Epochs (Custom Epochs)
echo   [3] Huấn luyện theo giới hạn thời gian (Time Limit: 30, 60, 120 phút...)
echo   [4] Huấn luyện liên tục không giới hạn (Dừng bất kỳ lúc nào bằng Ctrl+C)
echo   [5] Đánh giá Benchmark LFW trên trọng số hiện tại (6.000 cặp kiểm thử)
echo   [6] Chạy Demo nhận diện khuôn mặt qua Webcam (camera_demo.py)
echo   [7] Xem lịch sử huấn luyện (training_history.csv) và đồ thị báo cáo
echo   [8] Thoát
echo.
echo ===============================================================================
set /p CHOICE="Nhập lựa chọn của bạn [1-8]: "

if "%CHOICE%"=="1" goto QUICK_TRAIN
if "%CHOICE%"=="2" goto CUSTOM_EPOCHS
if "%CHOICE%"=="3" goto TIME_LIMIT
if "%CHOICE%"=="4" goto INFINITE_TRAIN
if "%CHOICE%"=="5" goto EVAL_ONLY
if "%CHOICE%"=="6" goto RUN_CAMERA
if "%CHOICE%"=="7" goto VIEW_HISTORY
if "%CHOICE%"=="8" goto EXIT

echo Lựa chọn không hợp lệ. Vui lòng nhập từ 1 đến 8.
timeout /t 2 >nul
goto MENU

:QUICK_TRAIN
echo.
echo [+] Bắt đầu huấn luyện nhanh (30 Epochs)...
echo.
"%PYTHON_EXE%" train_lfw_distill.py --epochs 30 --batch-size 64 %DATA_OPT% %TEACHER_OPT%
goto FINISH

:CUSTOM_EPOCHS
echo.
set /p NUM_EPOCHS="Nhập số lượng Epochs (ví dụ: 50, 100, 200): "
if "%NUM_EPOCHS%"=="" set NUM_EPOCHS=50
echo [+] Bắt đầu huấn luyện %NUM_EPOCHS% Epochs...
echo.
"%PYTHON_EXE%" train_lfw_distill.py --epochs %NUM_EPOCHS% --batch-size 64 %DATA_OPT% %TEACHER_OPT%
goto FINISH

:TIME_LIMIT
echo.
set /p NUM_MINUTES="Nhập thời gian huấn luyện tính bằng phút (ví dụ: 30, 60, 120): "
if "%NUM_MINUTES%"=="" set NUM_MINUTES=30
echo [+] Bắt đầu huấn luyện có giới hạn thời gian (%NUM_MINUTES% phút)...
echo.
"%PYTHON_EXE%" train_lfw_distill.py --time-limit %NUM_MINUTES% --batch-size 64 %DATA_OPT% %TEACHER_OPT%
goto FINISH

:INFINITE_TRAIN
echo.
echo ===============================================================================
echo   [CHẾ ĐỘ HUẤN LUYỆN LIÊN TỤC KHÔNG GIỚI HẠN]
echo   Mô hình sẽ học liên tục epoch này qua epoch khác.
echo   Bất kỳ khi nào muốn dừng: Hãy bấm phím CTRL+C trong cửa sổ này.
echo   Hệ thống sẽ tự động bắt tín hiệu Ctrl+C, lưu lại trọng số tốt nhất,
echo   và xuất báo cáo kiểm thử LFW đầy đủ!
echo ===============================================================================
echo.
"%PYTHON_EXE%" train_lfw_distill.py --infinite --batch-size 64 %DATA_OPT% %TEACHER_OPT%
goto FINISH

:EVAL_ONLY
echo.
echo [+] Chạy đánh giá kiểm thử LFW 10-Fold trên file trọng số hiện tại...
echo.
"%PYTHON_EXE%" train_lfw_distill.py --eval-only %DATA_OPT%
goto FINISH

:RUN_CAMERA
echo.
echo [+] Đang khởi động ứng dụng nhận diện Webcam...
echo.
"%PYTHON_EXE%" camera_demo.py
goto FINISH

:VIEW_HISTORY
echo.
if exist "docs\training_history.csv" (
    echo [+] Mở docs\training_history.csv...
    start "" "docs\training_history.csv"
) else if exist "training_history.csv" (
    echo [+] Mở training_history.csv...
    start "" "training_history.csv"
) else (
    echo [!] Chưa tìm thấy training_history.csv. Hãy chạy một phiên huấn luyện trước!
)
if exist "reports" (
    echo [+] Mở thư mục reports...
    start "" "reports"
)
if exist "docs\lfw_benchmark_report.png" (
    start "" "docs\lfw_benchmark_report.png"
) else if exist "lfw_benchmark_report.png" (
    start "" "lfw_benchmark_report.png"
)
echo.
pause
goto MENU

:FINISH
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===============================================================================
    echo   [THÀNH CÔNG] Hoàn tất tiến trình!
    echo   - File trọng số đã lưu: weights\tiny_mobilefacenet_lfw.pt
    echo   - File biểu đồ báo cáo: lfw_benchmark_report.png (hoặc docs\)
    echo   - Lịch sử huấn luyện  : training_history.csv
    echo ===============================================================================
    if exist "lfw_benchmark_report.png" start "" lfw_benchmark_report.png
) else (
    echo.
    echo [!] Tiến trình kết thúc với mã lỗi %ERRORLEVEL%.
)
echo.
pause
goto MENU

:EXIT
echo Đang thoát...
exit /b
