@echo off
rem Daily scheduled run: analyse the cameras in cameras.txt until 19:00 (or the HH:MM given as %1),
rem then close the files cleanly. Output goes to logs\analysis_YYMMDD.log (one file per day).
cd /d %~dp0
set UNTIL=%1
if "%UNTIL%"=="" set UNTIL=19:00
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyMMdd"') do set DAY=%%d
if not exist logs mkdir logs
set LOG=logs\analysis_%DAY%.log
echo ===== %date% %time% start (until %UNTIL%) >> %LOG%
venv\Scripts\python.exe make_analysis.py --until %UNTIL% >> %LOG% 2>&1
echo ===== %date% %time% exit %errorlevel% >> %LOG%
