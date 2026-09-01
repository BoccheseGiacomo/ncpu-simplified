@echo off
setlocal
set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\slackenv\python.exe"
set "CHECKPOINT=%~dp0checkpoints\best.pt"

if not exist "%PYTHON_EXE%" (
  echo Python was not found at:
  echo %PYTHON_EXE%
  exit /b 1
)

if not exist "%CHECKPOINT%" (
  echo No checkpoint was found at:
  echo %CHECKPOINT%
  echo Train the model with run\run.ipynb first.
  exit /b 1
)

cd /d "%~dp0"
"%PYTHON_EXE%" -c "import ncpu_simplified" >nul 2>&1
if errorlevel 1 (
  echo Install this repository in slackenv first:
  echo "%PYTHON_EXE%" -m pip install -e "%~dp0"
  exit /b 1
)

"%PYTHON_EXE%" -m ncpu_simplified.visualize --checkpoint "%CHECKPOINT%" --open
endlocal
