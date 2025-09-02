@echo off
REM Quick launcher without dependency checks (faster startup)
title Media Classifier - Quick Start
python media_classifier_gui.py
if errorlevel 1 pause