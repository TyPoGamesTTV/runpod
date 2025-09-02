@echo off
title Media Classifier GUI - 768x768 Model
echo ========================================
echo Starting Media Classifier GUI
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import torch" 2>nul
if errorlevel 1 (
    echo Installing PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
)

python -c "import cv2" 2>nul
if errorlevel 1 (
    echo Installing OpenCV...
    pip install opencv-python
)

python -c "import numpy" 2>nul
if errorlevel 1 (
    echo Installing NumPy...
    pip install numpy
)

REM Launch the GUI
echo.
echo Launching classifier...
echo ========================================
python media_classifier_gui.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Failed to start classifier
    echo Check the error message above
    echo ========================================
    pause
)