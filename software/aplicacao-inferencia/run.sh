#!/usr/bin/env bash
# RecycleAI-Station — launcher Linux/macOS
# Usa o runtime interno em runtime_inferencia/venv (runtime operacional padrao do projeto).
# Executar a partir da raiz do projeto: bash run.sh

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/runtime_inferencia/venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "[ERRO] Runtime interno nao encontrado: $PYTHON"
    echo "Execute: python -m venv runtime_inferencia/venv  e  pip install -r requirements_linux.txt"
    exit 1
fi

exec "$PYTHON" "$ROOT/app/main.py" "$@"
