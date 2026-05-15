@echo off
REM RecycleAI-Station — launcher Windows
REM Usa o runtime interno em runtime_inferencia\venv (runtime operacional padrao do projeto).
REM Executar a partir da raiz do projeto: run.bat

set "ROOT=%~dp0"
set "PYTHON=%ROOT%runtime_inferencia\venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [ERRO] Runtime interno nao encontrado: %PYTHON%
    echo Execute: python -m venv runtime_inferencia\venv  e  pip install -r requirements_windows.txt
    pause
    exit /b 1
)

"%PYTHON%" "%ROOT%app\main.py" %*
