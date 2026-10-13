@echo off
rem Refresh: the engine's data, who plays, the projections and game lines; the Results snapshot; then
rem publish to the data release the Streamlit Cloud app reads. No odds credits are spent here.
cd /d "%~dp0"
(set "PYTHONUTF8=1")
(set "PYTHONIOENCODING=utf-8")
(set "POLARS_MAX_THREADS=8")
set PY="%USERPROFILE%\nfl-projection-system\.venv\Scripts\python.exe"
if not exist data mkdir data
echo ==== %date% %time% >> "data\refresh.log"
%PY% live.py >> "data\refresh.log" 2>&1
%PY% tracking.py snapshot >> "data\refresh.log" 2>&1
%PY% publish.py >> "data\refresh.log" 2>&1
