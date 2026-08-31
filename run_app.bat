@echo off
REM Start the NFL season projections app on its own port and leave it running.
REM
REM Double-click this, or point a Desktop shortcut at it. The point of having a
REM launcher at all is ownership of the process: a server started from inside an
REM agent/terminal session is a CHILD of that session, and dies with it -- which
REM shows up in the browser as "connection error" mid-edit. Started from here it
REM belongs to the desktop and stays up until it is closed.
REM
REM Port is pinned in .streamlit\config.toml (8611). See %USERPROFILE%\PORTS.md.

setlocal
cd /d "%~dp0"
REM quoted on purpose: `set PYTHONUTF8=1 ` with a trailing space is a *fatal* CPython preconfig error
set "PYTHONUTF8=1"

if not exist "build\review" mkdir "build\review"

start "" http://localhost:8611

".venv\Scripts\python.exe" -m streamlit run app\Home.py >> "build\review\serve.log" 2>&1
endlocal
