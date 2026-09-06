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
REM polars is a native extension, and a fault inside it kills the process with no Python traceback --
REM which is what a "connection error" mid-edit actually is. faulthandler prints the Python stack of
REM every thread into build\review\serve.log on the way down, so the next one names the line.
set "PYTHONFAULTHANDLER=1"
REM One polars worker, for two reasons that happen to agree.
REM
REM Correctness first: every crash that took this server down on 2 and 3 September was the same
REM instruction in polars' own native runtime -- `_polars_runtime.pyd+0x83a65c7`, a read of address
REM 0x18, i.e. a NULL dereference -- on a polars *worker* thread two frames from the thread entry
REM point, with no CPython frame anywhere on the stack. Ten dumps under
REM %LOCALAPPDATA%\CrashDumps, ten identical faults. That is polars' work-stealing pool racing with
REM itself, not anything this app can express in Python, and one worker cannot steal from itself.
REM
REM Speed second: this engine is thousands of tiny operations on ~1,000-row frames, so coordinating
REM 14 workers costs far more than it buys. One worker runs the engine in 1.2s against 3.0s warm and
REM the whole page-by-page smoke in 150s against 178s. Results are unaffected -- verified identical
REM at 1 and at 14 -- so this is not a speed/accuracy trade.
set "POLARS_MAX_THREADS=1"

if not exist "build\review" mkdir "build\review"

REM Already up? Then this click is "show me the app", not "start another one".
netstat -ano | findstr /r /c:"TCP.*:8611 .*LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo Already serving on http://localhost:8611 -- opening it.
  start "" http://localhost:8611
  goto :done
)

start "" http://localhost:8611

REM Supervised, because a native fault is not something Python can catch. Every edit is already
REM autosaved through ui.set_live(), so a crash costs no work -- it costs the *session*, and that is
REM what "Is Streamlit still running?" is. Restarting here turns that into a few seconds of the
REM browser reconnecting on its own. Closing this window still stops the server for good: the loop
REM dies with the console, so this does not fight a deliberate shutdown.
set /a tries=0
:serve
".venv\Scripts\python.exe" -m streamlit run app\Home.py >> "build\review\serve.log" 2>&1
set "code=%errorlevel%"
if "%code%"=="0" goto :done
REM Ctrl+C / Ctrl+Break is a deliberate stop, not a fault.
if "%code%"=="-1073741510" goto :done
set /a tries+=1
REM A server that cannot survive its own startup would otherwise spin here forever.
if %tries% GEQ 20 (
  echo [%date% %time%] gave up after %tries% restarts, last exit %code% >> "build\review\serve.log"
  echo The app crashed %tries% times. See build\review\serve.log -- press a key to close.
  pause >nul
  goto :done
)
echo [%date% %time%] exit %code% -- restarting (%tries%/20) >> "build\review\serve.log"
timeout /t 2 /nobreak >nul
goto :serve

:done
endlocal
