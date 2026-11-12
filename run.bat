@echo off
chcp 936 >nul
setlocal
if not exist ".venv\Scripts\python.exe" (
  echo [失败] 未找到 .venv，请先运行 setup_env.bat 生成虚拟环境。
  exit /b 1
)
echo 正在激活 .venv 并启动 RN18 图形界面 ...
call ".venv\Scriptsctivate.bat"
if errorlevel 1 (
  echo [失败] 激活 .venv 失败。
  exit /b 1
)
python app.py
if errorlevel 1 (
  echo [失败] app.py 启动失败，请查看上方报错。
  exit /b 1
)
endlocal
exit /b 0
