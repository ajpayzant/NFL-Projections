@echo off
REM Start the NFL season projections app on its own port, or open the one already running.
REM
REM Double-click this, or click the Desktop / Start Menu shortcut that
REM scripts\install_shortcuts.ps1 makes. The point of having a launcher at all is
REM ownership of the process: a server started from inside an agent/terminal session
REM is a CHILD of that session and dies with it -- which shows up in the browser as
REM "connection error" mid-edit. Started from here it belongs to the desktop and
REM stays up until it is closed.
REM
REM Clicking it twice is safe. A second Streamlit on a taken port either fails or
REM wanders onto another one, and a second copy of this app would be a second set of
REM caches for the same work, so if 8611 is already serving this only opens the tab.
REM
REM Port is pinned in .streamlit\config.toml (8611). See %USERPROFILE%\PORTS.md.

setlocal
cd /d "%~dp0"
REM quoted on purpose: `set PYTHONUTF8=1 ` with a trailing space is a *fatal* CPython preconfig error
set "PYTHONUTF8=1"

if not exist "build\review" mkdir "build\review"

REM Already up? Then this click is "show me the app", not "start another one".
netstat -ano | findstr /r /c:"TCP.*:8611 .*LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo Already serving on http://localhost:8611 -- opening it.
  start "" http://localhost:8611
  goto :done
)

start "" http://localhost:8611
".venv\Scripts\python.exe" -m streamlit run app\Home.py >> "build\review\serve.log" 2>&1

:done
endlocal
