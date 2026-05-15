#!/usr/bin/env python3
"""
RecycleAI — Software de Treinamento YOLOv5s
Entry point único.

Compatível com: Windows, Linux, Google Colab, RunPod/pods

Uso:
  python train.py                            # menu interativo (sem args)
  python train.py --config minha_config.yaml # override por arquivo (modo direto)
  python train.py --epochs 100 --batch 16    # override por CLI (modo direto)
  python train.py --dry-run                  # valida ambiente sem treinar
  python train.py --no-export               # treina sem exportar pacote

Modo menu (sem argumentos):
  Exibe menu interativo com opções de novo treinamento, refinamento e utilidades.

Modo direto (com argumentos CLI):
  Executa treinamento completo sem interação — adequado para scripts automáticos.
  Fluxo direto:
    1. Detectar ambiente (OS, GPU, VRAM, RAM)
    2. Validar pré-requisitos de hardware
    3. Validar dataset (datasets/dataset_ativo)
    4. Montar configuração sugerida + aplicar overrides CLI
    5. Executar treinamento YOLOv5s
    6. Exportar pacote completo do modelo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Verificação passiva de dependências — orienta o usuário sem interromper
from core.bootstrap import check_and_warn
check_and_warn()


# ─────────────────────────────────────────────────────────────────────────────
# Parsing de argumentos
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train.py",
        description="RecycleAI — Treinamento YOLOv5s",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", type=Path, default=None,
        metavar="FILE",
        help="Arquivo YAML de configuração (override parcial ou total)",
    )
    p.add_argument(
        "--dataset", type=Path,
        default=None,
        metavar="DIR",
        help="Caminho do dataset no formato YOLO (default: datasets/dataset_ativo)",
    )
    p.add_argument(
        "--epochs",   type=int,   default=None, help="Número de épocas (override)"
    )
    p.add_argument(
        "--batch",    type=int,   default=None, help="Tamanho do batch (override)"
    )
    p.add_argument(
        "--img-size", type=int,   default=640,  help="Tamanho da imagem (default 640)"
    )
    p.add_argument(
        "--name",     type=str,   default=None,
        help="Nome do experimento (pasta em runs/)",
    )
    p.add_argument(
        "--dry-run",  action="store_true",
        help="Valida ambiente e dataset sem executar treino",
    )
    p.add_argument(
        "--no-export", action="store_true",
        help="Pula exportação do pacote do modelo ao final",
    )
    return p.parse_args()


def _is_menu_mode(args: argparse.Namespace) -> bool:
    """
    Retorna True se nenhum argumento CLI relevante foi fornecido.
    Nesse caso o menu interativo é exibido.
    """
    return not any([
        args.config,
        args.dataset,
        args.epochs,
        args.batch,
        args.name,
        args.dry_run,
        args.no_export,
    ])


def _header():
    print("=" * 62)
    print("  RecycleAI — Software de Treinamento YOLOv5s  v1.0.0")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Modo direto (com argumentos CLI)
# ─────────────────────────────────────────────────────────────────────────────

def _run_direct(args: argparse.Namespace) -> int:
    """
    Executa o fluxo completo de novo treinamento sem interação.
    Usado quando o usuário passa argumentos CLI explícitos.
    """
    from core.environment import detect_environment
    from core.validator import validate_dataset, validate_training_prerequisites
    from core.config import build_config, load_config_override
    from core.trainer import run_training
    from core.exporter import export_model_package

    dataset_path = (args.dataset or ROOT / "datasets" / "dataset_ativo").resolve()

    # ── 1. Detectar ambiente ──────────────────────────────────────────────────
    print("\n[1/6] Detectando ambiente...")
    env = detect_environment()
    env.imprimir_resumo()

    # ── 2. Validar pré-requisitos ─────────────────────────────────────────────
    print("\n[2/6] Validando pré-requisitos de hardware...")
    ok, msg = validate_training_prerequisites(env)
    if not ok:
        print(f"\n  [BLOQUEADO] {msg}")
        print("  Verifique o driver NVIDIA e a instalação do PyTorch com suporte a CUDA.")
        return 1
    print(f"  OK — {msg}")

    # ── 3. Validar dataset ────────────────────────────────────────────────────
    print("\n[3/6] Validando dataset...")
    ok, lines = validate_dataset(dataset_path)
    for line in lines:
        print(f"  {line}")
    if not ok:
        print("\n  [BLOQUEADO] Corrija os problemas no dataset e tente novamente.")
        return 1

    if args.dry_run:
        print("\n  [DRY-RUN] Ambiente e dataset válidos. Nenhum treino executado.")
        return 0

    # ── 4. Montar configuração ────────────────────────────────────────────────
    print("\n[4/6] Montando configuração...")
    cfg = build_config(env, dataset_path, args)
    if args.config:
        cfg = load_config_override(cfg, args.config)
    cfg.imprimir_resumo()

    # ── 5. Executar treinamento ───────────────────────────────────────────────
    print("\n[5/6] Iniciando treinamento YOLOv5s...")
    result = run_training(cfg)
    if not result.success:
        print(f"\n  [ERRO] Treinamento falhou: {result.error}")
        return 1
    print(f"  Pesos: {result.best_weights}")
    if result.map50 is not None:
        print(f"  mAP50: {result.map50:.4f}  |  mAP50-95: {result.map50_95:.4f}")

    # ── 6. Exportar pacote ────────────────────────────────────────────────────
    if args.no_export:
        print("\n[6/6] Exportação pulada (--no-export).")
    else:
        print("\n[6/6] Exportando pacote completo do modelo...")
        pkg_path = export_model_package(cfg, result, env)
        print(f"  Pacote salvo em: {pkg_path}")

    print("\n" + "=" * 62)
    print("  Treinamento concluído com sucesso!")
    print("=" * 62)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    _header()
    args = _parse_args()

    if _is_menu_mode(args):
        from core.menu import run_menu
        return run_menu()

    return _run_direct(args)


if __name__ == "__main__":
    sys.exit(main())
