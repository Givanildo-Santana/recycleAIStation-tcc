"""
Bootstrap de dependências — aplicacao-treinamento.

Responsabilidades:
  1. Verificar quais dependências críticas estão instaladas
  2. Detectar a versão CUDA do sistema via nvidia-smi
  3. Determinar a URL correta do wheel PyTorch (cu121, cu118, cpu, etc.)
  4. Instalar torch + demais dependências no ambiente Python atual

Chamado por:
  - setup.bat / setup.sh  (--install, modo não interativo)
  - train.py              (check_and_warn, modo passivo)
  - setup_colab.py        (--install, modo Colab)

Uso direto:
  python core/bootstrap.py           # checa e pergunta se instala
  python core/bootstrap.py --install # instala sem perguntar
  python core/bootstrap.py --check   # apenas reporta o que falta
"""
from __future__ import annotations

import subprocess
import sys
import re
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Dependências mínimas obrigatórias para o treinador funcionar
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED: list[tuple[str, str]] = [
    # (import_name, pip_package_name)
    ("torch",        "torch"),
    ("torchvision",  "torchvision"),
    ("ultralytics",  "ultralytics>=8.0.0"),
    ("cv2",          "opencv-python>=4.8.0"),
    ("yaml",         "PyYAML>=6.0"),
    ("psutil",       "psutil>=5.9.0"),
    ("tqdm",         "tqdm>=4.65.0"),
    ("pandas",       "pandas>=2.0.0"),
    ("PIL",          "Pillow>=9.0.0"),
    ("matplotlib",   "matplotlib>=3.7.0"),
    ("seaborn",      "seaborn>=0.12.0"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Detecção de CUDA
# ─────────────────────────────────────────────────────────────────────────────

def detect_cuda_version() -> Optional[str]:
    """
    Detecta a versão CUDA do driver via nvidia-smi.
    Retorna string como '12.1', '11.8' ou None se não houver GPU NVIDIA.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            match = re.search(r'CUDA Version:\s*(\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def cuda_to_torch_index(cuda_ver: Optional[str]) -> Optional[str]:
    """
    Mapeia versão CUDA para a URL do índice PyPI do PyTorch.
    Retorna None se PyTorch CPU deve ser instalado (sem CUDA).
    """
    if cuda_ver is None:
        return None

    try:
        major, minor = (int(x) for x in cuda_ver.split(".")[:2])
        cuda_int = major * 10 + minor   # ex.: 12.1 → 121
    except ValueError:
        return None

    # Mapeamento: CUDA → wheel tag suportado pelo pytorch.org
    if cuda_int >= 124:
        return "https://download.pytorch.org/whl/cu124"
    if cuda_int >= 121:
        return "https://download.pytorch.org/whl/cu121"
    if cuda_int >= 118:
        return "https://download.pytorch.org/whl/cu118"
    if cuda_int >= 117:
        return "https://download.pytorch.org/whl/cu117"
    # CUDA muito antiga — tenta CPU (treino será bloqueado de qualquer forma)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Verificação de dependências
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies() -> tuple[bool, list[tuple[str, str]]]:
    """
    Verifica quais dependências estão ausentes.

    Returns:
        (all_ok, missing_list)
        missing_list: lista de (import_name, pip_package_name) ausentes.
    """
    missing: list[tuple[str, str]] = []
    for import_name, pip_name in _REQUIRED:
        try:
            __import__(import_name)
        except ImportError:
            missing.append((import_name, pip_name))
    return len(missing) == 0, missing


def check_and_warn() -> bool:
    """
    Verificação passiva: reporta dependências ausentes mas não instala.
    Usado pelo train.py na inicialização para orientar o usuário.

    Returns: True se tudo ok, False se há dependências ausentes.
    """
    ok, missing = check_dependencies()
    if ok:
        return True

    pip_names = [p for _, p in missing]
    print("\n  [AVISO] Dependências ausentes:", ", ".join(p for p, _ in missing))
    print("  Execute o script de setup para seu sistema:")
    print("    Windows : setup.bat")
    print("    Linux   : bash setup.sh")
    print("    Colab   : python setup_colab.py")
    print(f"  Ou instale manualmente: pip install {' '.join(pip_names)}")
    print()
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Instalação
# ─────────────────────────────────────────────────────────────────────────────

def install_dependencies(interactive: bool = True) -> bool:
    """
    Instala dependências ausentes no ambiente Python atual.

    Args:
        interactive: se True, pergunta antes de instalar.

    Returns: True se bem-sucedido, False se falhou ou cancelado.
    """
    ok, missing = check_dependencies()
    if ok:
        print("  Todas as dependências já estão instaladas.")
        return True

    pip_names = [p for _, p in missing]
    print(f"  Dependências ausentes: {', '.join(m for m, _ in missing)}")

    if interactive:
        try:
            resp = input("  Instalar agora? [S/n]: ").strip().lower()
            if resp == "n":
                print("  Instalação cancelada.")
                return False
        except (EOFError, KeyboardInterrupt):
            # Em ambientes não interativos (CI, pipes) prossegue automaticamente
            pass

    pip = [sys.executable, "-m", "pip", "install", "--upgrade"]

    # ── PyTorch: instala separadamente com CUDA correta ───────────────────────
    torch_missing = any(m in ("torch", "torchvision") for m, _ in missing)
    if torch_missing:
        cuda_ver = detect_cuda_version()
        index_url = cuda_to_torch_index(cuda_ver)

        if cuda_ver and index_url:
            print(f"\n  CUDA detectado: {cuda_ver} → instalando PyTorch CUDA ({index_url.split('/')[-1]})...")
            cmd = pip + ["torch", "torchvision", "--index-url", index_url]
        else:
            if cuda_ver:
                print(f"\n  CUDA {cuda_ver} detectado mas versão sem suporte → instalando PyTorch CPU.")
            else:
                print("\n  GPU CUDA não detectada → instalando PyTorch CPU.")
                print("  AVISO: treino não será possível sem GPU CUDA.")
            cmd = pip + ["torch", "torchvision"]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("  [ERRO] Falha ao instalar PyTorch. Verifique a saída acima.")
            return False

        # Remove torch/torchvision da lista de pendentes
        pip_names = [p for p in pip_names if p not in ("torch", "torchvision")]

    # ── Demais dependências ───────────────────────────────────────────────────
    if pip_names:
        req_file = Path(__file__).resolve().parent.parent / "requirements.txt"
        print("\n  Instalando demais dependências...")
        result = subprocess.run(pip + ["-r", str(req_file)])
        if result.returncode != 0:
            print("  [ERRO] Falha ao instalar dependências. Verifique a saída acima.")
            return False

    # ── Verificação final ─────────────────────────────────────────────────────
    ok, still_missing = check_dependencies()
    if ok:
        print("\n  Todas as dependências instaladas com sucesso!")
        return True
    else:
        print(f"\n  [AVISO] Ainda ausentes após instalação: {[m for m, _ in still_missing]}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI direto
# ─────────────────────────────────────────────────────────────────────────────

def _main():
    import argparse
    p = argparse.ArgumentParser(description="RecycleAI — Bootstrap de dependências")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--install", action="store_true",
                     help="Instala dependências sem perguntar (não interativo)")
    grp.add_argument("--check",   action="store_true",
                     help="Apenas verifica e reporta, sem instalar")
    args = p.parse_args()

    if args.check:
        ok, missing = check_dependencies()
        if ok:
            print("Todas as dependências estão instaladas.")
            sys.exit(0)
        else:
            for imp, pip_name in missing:
                print(f"  AUSENTE: {imp}  (pip: {pip_name})")
            sys.exit(1)

    elif args.install:
        success = install_dependencies(interactive=False)
        sys.exit(0 if success else 1)

    else:
        # Modo padrão: interativo
        success = install_dependencies(interactive=True)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    _main()
