@echo off
:: build.bat — Gera bundle PyInstaller e instalador Inno Setup
:: Executar de dentro de software/aplicacao-inferencia/
:: Uso: install\build.bat
::
:: O que faz:
::   1. Roda PyInstaller (gera dist\RecycleAI-Station\)
::   2. Roda Inno Setup  (gera dist\RecycleAI-Station-Setup-1.0.0.exe)

setlocal enabledelayedexpansion

set BASE=%~dp0..
set PYTHON=%BASE%\runtime_inferencia\venv\Scripts\python.exe
set ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe

echo ===================================================
echo RecycleAI-Station — Build
echo ===================================================
echo.

:: Etapa 1 — PyInstaller
echo [1/2] Gerando bundle PyInstaller...
if not exist "%PYTHON%" (
    echo ERRO: python.exe nao encontrado em: %PYTHON%
    exit /b 1
)
cd /d "%BASE%"
"%PYTHON%" -m PyInstaller install\recycleai.spec --noconfirm
if !errorlevel! neq 0 (
    echo ERRO: PyInstaller falhou.
    exit /b 1
)
echo     Bundle gerado: dist\RecycleAI-Station\
echo.

:: Etapa 2 — Inno Setup
echo [2/2] Gerando instalador Inno Setup...
if not exist "%ISCC%" (
    echo AVISO: ISCC.exe nao encontrado em: %ISCC%
    echo        Instale Inno Setup 6 para gerar o instalador.
    echo        Download: https://jrsoftware.org/isdl.php
    goto :done
)
"%ISCC%" "install\recycleai_setup.iss"
if !errorlevel! neq 0 (
    echo ERRO: Inno Setup falhou.
    exit /b 1
)
echo     Instalador gerado: dist\RecycleAI-Station-Setup-1.0.0.exe
echo.

:done
echo ===================================================
echo BUILD CONCLUIDA COM SUCESSO
echo   Executavel : %BASE%\dist\RecycleAI-Station\RecycleAI-Station.exe
echo   Instalador : %BASE%\dist\RecycleAI-Station-Setup-1.0.0.exe
echo ===================================================
endlocal
