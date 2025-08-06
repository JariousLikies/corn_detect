@echo off
chcp 65001 >nul 2>&1
title Corn Defect Detection App Deployment
echo Starting Corn Defect Detection Application...

:: Check if conda is available
where conda >nul 2>&1
if errorlevel 1 (
    echo Error: Conda not found. Please make sure Conda is installed and added to PATH.
    goto :end
)

:: Run Streamlit app using conda environment
conda run -n corn_defect_detection python -m streamlit run main.py --server.port 8501

if errorlevel 1 (
    echo Error: Failed to start the application. Check environment name or file path.
) else (
    echo Application started successfully!
    echo Local URL: http://localhost:8503
    echo Network URL: http://10.194.180.180:8503
)
:end
pause