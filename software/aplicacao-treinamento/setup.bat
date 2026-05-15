@echo off
:: RecycleAI — Setup e Lançamento (Windows)
:: Cria ambiente isolado em runtime\, instala dependências e inicia o treinamento.
::
:: Uso:
::   setup.bat                  — interativo: pergunta antes de instalar
::   setup.bat --install        — não interativo: instala sem perguntar
::   setup.bat --check          — apenas verifica dependências, não instala
::   setup.bat --dry-run        — valida ambiente e dataset sem treinar
::   setup.bat --epochs 100     — passa args para train.py
::
:: Requisitos mínimos:
::   Python 3.10–3.12 no PATH
::   GPU NVIDIA com driver CUDA (recomendado)

setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "RUNTIME=%ROOT%runtime"
set "PYTHON_VENV=%RUNTIME%\Scripts\python.exe"
set "PIP_VENV=%RUNTIME%\Scripts\pip.exe"

echo ============================================================
echo   RecycleAI — Setup de Ambiente (Windows)
echo ============================================================

:: ── Verificar Python no PATH ──────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [ERRO] Python nao encontrado no PATH.
    echo   Instale Python 3.10 ou superior em: https://python.org/downloads/
    echo   Marque "Add Python to PATH" durante a instalacao.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
echo   Python encontrado: %PY_VER%

:: ── Criar ambiente virtual isolado ───────────────────────────────────────────
if not exist "%PYTHON_VENV%" (
    echo.
    echo   Criando ambiente virtual em runtime\...
    python -m venv "%RUNTIME%"
    if errorlevel 1 (
        echo   [ERRO] Falha ao criar ambiente virtual.
        echo   Verifique se o modulo venv esta disponivel: python -m venv --help
        pause
        exit /b 1
    )
    echo   Ambiente criado com sucesso.
) else (
    echo   Ambiente virtual existente: runtime\
)

:: ── Garantir pip atualizado ───────────────────────────────────────────────────
echo.
echo   Atualizando pip...
"%PYTHON_VENV%" -m pip install --upgrade pip --quiet

:: ── Instalar / verificar dependências via bootstrap.py ────────────────────────
echo.

:: Verificar se primeiro argumento é --check
if "%~1"=="--check" (
    "%PYTHON_VENV%" "%ROOT%core\bootstrap.py" --check
    exit /b %errorlevel%
)

if "%~1"=="--install" (
    "%PYTHON_VENV%" "%ROOT%core\bootstrap.py" --install
) else (
    "%PYTHON_VENV%" "%ROOT%core\bootstrap.py"
)

if errorlevel 1 (
    echo.
    echo   [ERRO] Instalacao de dependencias falhou.
    echo   Verifique a saida acima e tente novamente.
    pause
    exit /b 1
)

:: ── Remover --install / --check dos argumentos antes de passar ao train.py ───
:: (train.py não reconhece esses flags — os demais são repassados)
set "TRAIN_ARGS="
set "SKIP_NEXT=0"
:parse_args
if "%~1"=="" goto launch
if "%~1"=="--install" (
    shift
    goto parse_args
)
if "%~1"=="--check" (
    shift
    goto parse_args
)
set "TRAIN_ARGS=%TRAIN_ARGS% %1"
shift
goto parse_args

:launch
:: ── Lançar train.py ───────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Iniciando train.py%TRAIN_ARGS%
echo ============================================================
echo.
"%PYTHON_VENV%" "%ROOT%train.py"%TRAIN_ARGS%
set "EXIT_CODE=%errorlevel%"

echo.
if %EXIT_CODE%==0 (
    echo   Treinamento concluido. Pacote salvo em exports\
) else (
    echo   Treinamento encerrado com erros (codigo %EXIT_CODE%).
)

pause
exit /b %EXIT_CODE%
