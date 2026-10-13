@echo off
rem Weekly refresh: the engine's data, who plays, the projections, game lines, then today's Results snapshot.
cd /d "%~dp0"
(set "PYTHONUTF8=1")
(set "PYTHONIOENCODING=utf-8")
(set "POLARS_MAX_THREADS=8")
echo ==== %date% %time% >> "data\refresh.log"
"%USERPROFILE%\nfl-projection-system\.venv\Scripts\python.exe" live.py >> "data\refresh.log" 2>&1
