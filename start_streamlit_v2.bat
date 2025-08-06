@echo off
chcp 65001 >nul 2>&1
title Corn Defect Detection App Deployment
echo Starting Corn Defect Detection Application...

:: 手动调用 conda 激活器（防止 PATH 缺失问题）
call "D:\Program Files\miniconda\Scripts\activate.bat"

:: 激活环境并运行 Streamlit
call conda activate corn_defect_detection
python -m streamlit run main.py --server.port 8501

if errorlevel 1 (
    echo Error: Failed to start the application. Check environment name or file path.
) else (
    echo Application started successfully!
    echo Local URL: http://localhost:8501
    echo Network URL: http://10.194.180.180:8501
)

:end
pause
