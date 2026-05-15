#!/usr/bin/env bash
# RecycleAI — Setup e Lançamento (Linux / macOS)
# Cria ambiente isolado em runtime/, instala dependências e inicia o treinamento.
#
# Uso:
#   bash setup.sh                  — interativo: pergunta antes de instalar
#   bash setup.sh --install        — não interativo: instala sem perguntar
#   bash setup.sh --check          — apenas verifica dependências, não instala
#   bash setup.sh --dry-run        — valida ambiente e dataset sem treinar
#   bash setup.sh --epochs 100     — passa args para train.py
#
# Requisitos mínimos:
#   Python 3.10–3.12 disponível como python3
#   GPU NVIDIA com driver CUDA (recomendado)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$SCRIPT_DIR/runtime"
PYTHON_VENV="$RUNTIME_DIR/bin/python"

echo "============================================================"
echo "  RecycleAI — Setup de Ambiente (Linux/macOS)"
echo "============================================================"

# ── Localizar Python ──────────────────────────────────────────────────────────
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')" 2>/dev/null || true)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "${major:-0}" -ge 3 ] && [ "${minor:-0}" -ge 10 ]; then
            PYTHON_BIN="$candidate"
            echo "  Python encontrado: $candidate ($ver)"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo ""
    echo "  [ERRO] Python 3.10+ não encontrado."
    echo "  Instale com:"
    echo "    Ubuntu/Debian : sudo apt install python3.11 python3.11-venv"
    echo "    Fedora/RHEL   : sudo dnf install python3.11"
    echo "    macOS (brew)  : brew install python@3.11"
    exit 1
fi

# ── Criar ambiente virtual isolado ───────────────────────────────────────────
if [ ! -f "$PYTHON_VENV" ]; then
    echo ""
    echo "  Criando ambiente virtual em runtime/..."
    "$PYTHON_BIN" -m venv "$RUNTIME_DIR"
    echo "  Ambiente criado com sucesso."
else
    echo "  Ambiente virtual existente: runtime/"
fi

# ── Garantir pip atualizado ───────────────────────────────────────────────────
echo ""
echo "  Atualizando pip..."
"$PYTHON_VENV" -m pip install --upgrade pip --quiet

# ── Instalar / verificar dependências via bootstrap.py ────────────────────────
echo ""

# Filtrar --install / --check antes de repassar ao train.py
BOOTSTRAP_ARG=""
TRAIN_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --install) BOOTSTRAP_ARG="--install" ;;
        --check)   BOOTSTRAP_ARG="--check"   ;;
        *)         TRAIN_ARGS+=("$arg")       ;;
    esac
done

if [ "$BOOTSTRAP_ARG" = "--check" ]; then
    "$PYTHON_VENV" "$SCRIPT_DIR/core/bootstrap.py" --check
    exit $?
elif [ "$BOOTSTRAP_ARG" = "--install" ]; then
    "$PYTHON_VENV" "$SCRIPT_DIR/core/bootstrap.py" --install
else
    "$PYTHON_VENV" "$SCRIPT_DIR/core/bootstrap.py"
fi

# ── Lançar train.py ───────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Iniciando train.py ${TRAIN_ARGS[*]+"${TRAIN_ARGS[*]}"}"
echo "============================================================"
echo ""

"$PYTHON_VENV" "$SCRIPT_DIR/train.py" "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Treinamento concluído. Pacote salvo em exports/"
else
    echo "  Treinamento encerrado com erros (código $EXIT_CODE)."
fi

exit $EXIT_CODE
