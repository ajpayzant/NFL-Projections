@echo off
REM Update the data behind the projection. This is a wrapper -- the work is in update_data.py.
REM
REM     scripts\weekly_update.bat            light: rosters, depth charts, schedules, injuries
REM     scripts\weekly_update.bat --full     light, after rebuilding the played-game lake
REM
REM Exists because a Windows scheduled task wants one file to point at and no environment set up for
REM it, and because PYTHONUTF8 has to be on before the interpreter starts. scripts\install_schedule.ps1
REM registers the tasks that call this; build\review\update.log is where every run is written.

setlocal
cd /d "%~dp0.."
REM quoted on purpose: `set PYTHONUTF8=1 ` with a trailing space is a *fatal* CPython preconfig error
set "PYTHONUTF8=1"

if not exist ".venv\Scripts\python.exe" (
  echo Cannot find .venv\Scripts\python.exe -- create the venv first.
  exit /b 2
)

".venv\Scripts\python.exe" scripts\update_data.py %*
exit /b %ERRORLEVEL%
