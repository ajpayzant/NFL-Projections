@echo off
rem NFL player game projections: http://localhost:8690
cd /d "%~dp0"
(set "PYTHONUTF8=1")
(set "POLARS_MAX_THREADS=4")
"%USERPROFILE%\nfl-projection-system\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8690
