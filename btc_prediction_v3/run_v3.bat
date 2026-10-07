@echo off
cd /d "%~dp0"
python btc_model_v3.py --years 3 --horizon 12
pause
