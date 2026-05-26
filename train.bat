@echo off
chcp 65001 > nul
echo.
echo ============================================================
echo   STARTING FASHION MNIST MODEL TRAINING
echo ============================================================
echo.
.\venv\Scripts\python.exe -X utf8 backend/train_model.py
echo.
echo ============================================================
echo   TRAINING COMPLETED!
echo   Run evaluate_model.py or evaluate_uploads.py to test.
echo ============================================================
pause
