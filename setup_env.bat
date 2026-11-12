@echo off
chcp 936 >nul
setlocal
echo ============================================================
echo   RN18 交付环境一键安装（在本机现场生成 .venv）
echo ============================================================

echo [1/6] 检查 Python（需 3.10 及以上，且已加入 PATH）...
python --version
if errorlevel 1 (
  echo [失败] 未找到可用的 python。请安装 Python 3.10+ 并勾选 Add Python to PATH 后重试。
  exit /b 1
)

echo [2/6] 创建虚拟环境 .venv ...
if exist ".venv\Scripts\python.exe" (
  echo       .venv 已存在，跳过创建。
) else (
  python -m venv .venv
  if errorlevel 1 (
    echo [失败] 创建 .venv 失败，请确认 Python 版本 ^>= 3.10 且当前目录可写。
    exit /b 1
  )
)

echo [3/6] 升级 pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo [失败] 升级 pip 失败，请检查网络后重试。
  exit /b 1
)

echo [4/6] 安装 PyTorch CPU 版 ...
echo   提示：torch==2.2.2+cpu / torchvision==0.17.2+cpu 只在官方 CPU 源发布，
echo         必须带 --index-url https://download.pytorch.org/whl/cpu 才会装到 CPU 版；
echo         否则装到的可能是 CUDA 版或版本不匹配。
".venv\Scripts\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch==2.2.2+cpu torchvision==0.17.2+cpu
if errorlevel 1 (
  echo [失败] PyTorch CPU 版安装失败，请确认网络与 --index-url 指向的 CPU 源可达。
  exit /b 1
)

echo [5/6] 安装其余锁定依赖（requirements.txt）...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [失败] 依赖安装失败，请查看上方 pip 报错。
  exit /b 1
)

echo [6/6] 验证安装 ...
".venv\Scripts\python.exe" -c "import torch,torchvision,gradio,numpy; print('import ok')"
if errorlevel 1 (
  echo [失败] 依赖导入验证未通过，请勿继续，检查上方报错。
  exit /b 1
)

echo.
echo 环境就绪，运行 run.bat 启动
endlocal
exit /b 0
