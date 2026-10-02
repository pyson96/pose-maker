@echo off
rem Daily scheduled run: analyse the cameras in cameras.txt until 19:00, then close the files cleanly.
rem Output is appended to logs\analysis.log.
cd /d %~dp0
if not exist logs mkdir logs
echo ===== %date% %time% start >> logs\analysis.log
venv\Scripts\python.exe make_analysis.py --until 19:00 >> logs\analysis.log 2>&1
echo ===== %date% %time% exit %errorlevel% >> logs\analysis.log
