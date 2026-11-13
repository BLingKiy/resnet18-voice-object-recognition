@echo off
rem ==============================================================
rem  RN18 - environment setup (Windows)
rem  1) check python  2) create .venv  3) upgrade pip
rem  4) install PyTorch CPU build  5) install requirements.txt
rem  6) verify imports
rem  Chinese guide: docs\STARTUP.md  (section 1)
rem  Keep this file pure ASCII with CRLF line endings.
rem ==============================================================
setlocal
cd /d "%~dp0"

echo [1/6] Checking Python ...
python --version
if errorlevel 1 goto :no_python

echo [2/6] Creating virtual environment .venv ...
if exist ".venv\Scripts\python.exe" (
    echo        .venv already exists, skip.
) else (
    python -m venv .venv
    if errorlevel 1 goto :venv_failed
)

echo [3/6] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :pip_failed

echo [4/6] Installing PyTorch CPU build ...
echo        torch==2.2.2+cpu / torchvision==0.17.2+cpu
echo        CPU wheels only, from the official PyTorch CPU index.
".venv\Scripts\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch==2.2.2+cpu torchvision==0.17.2+cpu
if errorlevel 1 goto :torch_failed

echo [5/6] Installing requirements.txt ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :req_failed

echo [6/6] Verifying imports ...
".venv\Scripts\python.exe" -c "import torch, torchvision, gradio, numpy; print('import ok')"
if errorlevel 1 goto :import_failed

echo.
echo Done. Run run.bat to launch the GUI.
endlocal
exit /b 0

:no_python
echo [ERROR] Python not found. Install Python 3.10+ and add it to PATH.
exit /b 1

:venv_failed
echo [ERROR] Failed to create .venv. Check that Python is 3.10 or newer.
exit /b 1

:pip_failed
echo [ERROR] Failed to upgrade pip. Check the network and the proxy settings.
exit /b 1

:torch_failed
echo [ERROR] Failed to install PyTorch CPU build. Check the network and retry.
exit /b 1

:req_failed
echo [ERROR] Failed to install requirements.txt. Check the network and retry.
exit /b 1

:import_failed
echo [ERROR] Import check failed. Re-run setup_env.bat and read the pip output above.
exit /b 1
