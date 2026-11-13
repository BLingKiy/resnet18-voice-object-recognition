@echo off
rem ==============================================================
rem  RN18 - launch the GUI (Windows)
rem  Activates .venv, then runs app.py.
rem  Run setup_env.bat once before the first launch.
rem  Chinese guide: docs\STARTUP.md  (section 2)
rem  Keep this file pure ASCII with CRLF line endings.
rem ==============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto :no_venv

echo Activating .venv and starting the RN18 GUI ...
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :activate_failed

python app.py
if errorlevel 1 goto :app_failed

endlocal
exit /b 0

:no_venv
echo [ERROR] .venv not found. Run setup_env.bat first.
exit /b 1

:activate_failed
echo [ERROR] Failed to activate .venv. Delete .venv and run setup_env.bat again.
exit /b 1

:app_failed
echo [ERROR] app.py exited with an error. Read the output above.
exit /b 1
