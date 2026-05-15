#!/usr/bin/env python3
"""
RecycleAI — Setup para Google Colab / RunPod / pods em geral.

NÃO cria venv — instala direto no Python do ambiente (que já é isolado
pelo próprio Colab/RunPod). Detecta CUDA e instala o PyTorch correto.

Uso no Colab (numa célula de código):
    !git clone <repo>
    %cd aplicacao-treinamento
    !python setup_colab.py
    !python train.py --epochs 100

Uso em RunPod / pod JupyterLab:
    python setup_colab.py
    python train.py --epochs 100

Flags:
    --install   Instala sem perguntar (padrão em ambientes detectados como Colab/pod)
    --check     Apenas verifica e reporta, sem instalar
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Adiciona o diretório raiz ao path para importar core.bootstrap
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _detect_cloud_env() -> str:
    """Detecta se estamos em Colab, RunPod, pod genérico ou local."""
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    if "RUNPOD_POD_ID" in os.environ:
        return "runpod"
    # Pod genérico: tipicamente não tem display e roda como root
    if os.environ.get("JUPYTER_SERVER_ROOT") or os.environ.get("JUPYTERHUB_USER"):
        return "pod"
    return "local"


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="RecycleAI — Setup para Colab/RunPod/pods"
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--install", action="store_true",
                     help="Instala dependências sem perguntar")
    grp.add_argument("--check",   action="store_true",
                     help="Apenas verifica e reporta, sem instalar")
    args = p.parse_args()

    env_name = _detect_cloud_env()

    print("=" * 62)
    print("  RecycleAI — Setup de Dependências")
    print(f"  Ambiente detectado: {env_name}")
    print("=" * 62)

    # Colab e pods são não-interativos por padrão
    force_non_interactive = env_name in ("colab", "runpod", "pod") or args.install

    if args.check:
        from core.bootstrap import check_dependencies
        ok, missing = check_dependencies()
        if ok:
            print("\n  Todas as dependências estão instaladas.")
            sys.exit(0)
        else:
            print("\n  Dependências ausentes:")
            for imp, pip_name in missing:
                print(f"    AUSENTE: {imp}  (pip: {pip_name})")
            sys.exit(1)

    # Aviso sobre GPU no Colab
    if env_name == "colab":
        print("\n  Dica: certifique-se de que o acelerador de hardware está")
        print("  configurado como GPU em: Ambiente de execução → Alterar tipo.")
        print()

    from core.bootstrap import install_dependencies
    success = install_dependencies(interactive=not force_non_interactive)

    if not success:
        print("\n  [ERRO] Falha na instalação de dependências.")
        sys.exit(1)

    print("\n  Setup concluído. Execute o treinamento com:")
    print("    python train.py")
    print("    python train.py --epochs 100 --batch 16")
    print("    python train.py --dry-run  # apenas valida ambiente e dataset")
    sys.exit(0)


if __name__ == "__main__":
    main()
