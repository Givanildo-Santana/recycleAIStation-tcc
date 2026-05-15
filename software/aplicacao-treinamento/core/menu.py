"""
Menu interativo e orquestração dos fluxos de treinamento e refinamento.

Entry point chamado por train.py quando invocado sem argumentos CLI.

Menu principal:
  1) Novo treinamento        — detecta ambiente, valida, treina, exporta
  2) Refinar modelo existente — carrega pacote base, valida, refina, exporta
  3) Verificações manuais    — utilitário opcional (ambiente, dataset, pacote)
  4) Sair

Ambos os fluxos principais executam automaticamente:
  - Detecção de ambiente (OS, GPU, VRAM, RAM)
  - Validação de pré-requisitos de hardware
  - Validação de dataset
  - Geração de configuração sugerida com ajuste opcional
  - Treinamento / refinamento
  - Exportação do pacote completo
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Entry point público
# ─────────────────────────────────────────────────────────────────────────────

def run_menu() -> int:
    """Exibe o menu principal e roteia para o fluxo escolhido."""
    while True:
        _print_header()
        print("  1) Novo treinamento")
        print("  2) Refinar modelo existente")
        print("  3) Verificações manuais")
        print("  4) Sair")
        print()

        choice = _ask("Escolha [1-4]", valid={"1", "2", "3", "4"})

        if choice == "1":
            return _flow_new_training()
        elif choice == "2":
            return _flow_refine()
        elif choice == "3":
            _flow_manual_checks()
        else:
            print("\n  Até logo.\n")
            return 0


# ─────────────────────────────────────────────────────────────────────────────
# Fluxo 1 — Novo treinamento
# ─────────────────────────────────────────────────────────────────────────────

def _flow_new_training() -> int:
    """
    Fluxo completo de novo treinamento com validações automáticas.

    1. Detectar ambiente
    2. Validar pré-requisitos de hardware
    3. Validar dataset (datasets/dataset_ativo/)
    4. Gerar configuração sugerida + ajuste opcional
    5. Executar treinamento
    6. Exportar pacote completo
    """
    from core.environment import detect_environment
    from core.validator import validate_training_prerequisites, validate_dataset
    from core.config import build_config
    from core.trainer import run_training
    from core.exporter import export_model_package
    import argparse

    _section("Novo Treinamento — Validação de Ambiente")

    # 1. Ambiente
    print("  Detectando ambiente...")
    env = detect_environment()
    env.imprimir_resumo()

    # 2. Hardware
    print("\n  Validando pré-requisitos de hardware...")
    ok, msg = validate_training_prerequisites(env)
    if not ok:
        _block(msg)
        return 1
    print(f"  OK — {msg}")

    # 3. Dataset
    dataset_path = _ROOT / "datasets" / "dataset_ativo"
    print(f"\n  Validando dataset: {dataset_path}")
    ok, lines = validate_dataset(dataset_path)
    for line in lines:
        print(f"  {line}")
    if not ok:
        _block("Corrija os problemas no dataset antes de treinar.")
        return 1

    # 4. Configuração sugerida + ajuste
    _section("Novo Treinamento — Configuração")
    args = _build_args_namespace()
    cfg  = build_config(env, dataset_path, args)
    cfg.imprimir_resumo()

    cfg = _apply_config_overrides(cfg, mode="training")

    # Confirmação final
    if not _confirm("\nIniciar treinamento com esta configuração?"):
        print("  Cancelado.")
        return 0

    # 5. Treinamento
    _section("Novo Treinamento — Treinamento")
    result = run_training(cfg)
    if not result.success:
        _block(f"Treinamento falhou: {result.error}")
        return 1
    print(f"\n  Pesos salvos: {result.best_weights}")
    if result.map50 is not None:
        print(f"  mAP50: {result.map50:.4f}  |  mAP50-95: {result.map50_95:.4f}")

    # 6. Exportação
    _section("Novo Treinamento — Exportação do Pacote")
    pkg_path = export_model_package(cfg, result, env)
    print(f"\n  Pacote exportado: {pkg_path}")
    _done()
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Fluxo 2 — Refinar modelo existente
# ─────────────────────────────────────────────────────────────────────────────

def _flow_refine() -> int:
    """
    Fluxo completo de refinamento (fine-tuning) com validações automáticas.

    1. Detectar ambiente
    2. Validar pré-requisitos de hardware
    3. Selecionar pacote base em modelos-base/
    4. Validar pacote base
    5. Validar dataset de refinamento (datasets/dataset_refino/)
    6. Verificar compatibilidade pacote base ↔ dataset
    7. Gerar configuração sugerida + ajuste opcional
    8. Executar refinamento
    9. Exportar novo pacote completo
    """
    from core.environment import detect_environment
    from core.validator import validate_training_prerequisites, validate_dataset
    from core.config import build_refine_config
    from core.trainer import run_training
    from core.exporter import export_model_package
    from core.refine_validator import (
        list_base_packages,
        validate_base_package,
        validate_refinement_compatibility,
    )
    import argparse

    _section("Refinamento — Validação de Ambiente")

    # 1. Ambiente
    print("  Detectando ambiente...")
    env = detect_environment()
    env.imprimir_resumo()

    # 2. Hardware
    print("\n  Validando pré-requisitos de hardware...")
    ok, msg = validate_training_prerequisites(env)
    if not ok:
        _block(msg)
        return 1
    print(f"  OK — {msg}")

    # 3. Selecionar pacote base
    _section("Refinamento — Seleção do Pacote Base")
    modelos_base_dir = _ROOT / "modelos-base"
    packages = list_base_packages(modelos_base_dir)

    if not packages:
        _block(
            f"Nenhum pacote base encontrado em: {modelos_base_dir}\n"
            "  Copie um pacote completo (pasta com manifest.json) para modelos-base/."
        )
        return 1

    print(f"  Pacotes disponíveis em {modelos_base_dir}:\n")
    for i, pkg in enumerate(packages, 1):
        _show_package_summary(pkg, i)

    idx = _ask_int("Escolha o pacote base", min_val=1, max_val=len(packages))
    base_package_dir = packages[idx - 1]
    print(f"\n  Selecionado: {base_package_dir.name}")

    # 4. Validar pacote base
    print("\n  Validando pacote base...")
    ok, issues, base_manifest = validate_base_package(base_package_dir)
    if not ok:
        for issue in issues:
            print(f"  ERRO: {issue}")
        _block("Pacote base inválido. Corrija os problemas acima.")
        return 1
    print(f"  OK — {base_manifest.name} ({base_manifest.classes.nc} classes: "
          f"{base_manifest.classes.names})")

    # 5. Validar dataset de refinamento
    refine_dataset_path = _ROOT / "datasets" / "dataset_refino"
    print(f"\n  Validando dataset de refinamento: {refine_dataset_path}")
    ok, lines = validate_dataset(refine_dataset_path)
    for line in lines:
        print(f"  {line}")
    if not ok:
        _block(
            "Dataset de refinamento inválido. "
            "Corrija a estrutura em datasets/dataset_refino/ antes de continuar."
        )
        return 1

    # 6. Compatibilidade pacote base ↔ dataset
    print("\n  Verificando compatibilidade...")
    ok, issues = validate_refinement_compatibility(base_manifest, refine_dataset_path)
    if not ok:
        for issue in issues:
            print(f"  ERRO: {issue}")
        _block("Incompatibilidade entre pacote base e dataset de refinamento.")
        return 1
    print("  OK — classes compatíveis.")

    # 7. Configuração sugerida + ajuste
    _section("Refinamento — Configuração")
    args = _build_args_namespace()
    cfg  = build_refine_config(env, base_manifest, base_package_dir,
                               refine_dataset_path, args)
    cfg.imprimir_resumo()

    cfg = _apply_config_overrides(cfg, mode="refinement")

    # Confirmação final
    if not _confirm("\nIniciar refinamento com esta configuração?"):
        print("  Cancelado.")
        return 0

    # 8. Refinamento (usa o mesmo trainer — diferença está no pretrained_weights)
    _section("Refinamento — Fine-tuning")
    result = run_training(cfg)
    if not result.success:
        _block(f"Refinamento falhou: {result.error}")
        return 1
    print(f"\n  Pesos salvos: {result.best_weights}")
    if result.map50 is not None:
        print(f"  mAP50: {result.map50:.4f}  |  mAP50-95: {result.map50_95:.4f}")

    # 9. Exportação
    _section("Refinamento — Exportação do Pacote")
    pkg_path = export_model_package(cfg, result, env)
    print(f"\n  Novo pacote refinado exportado: {pkg_path}")
    _done()
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Fluxo 3 — Verificações manuais (utilitário opcional)
# ─────────────────────────────────────────────────────────────────────────────

def _flow_manual_checks():
    """Menu de verificações manuais — utilitário opcional, não é o fluxo principal."""
    from core.environment import detect_environment
    from core.validator import validate_training_prerequisites, validate_dataset
    from core.refine_validator import list_base_packages, validate_base_package
    from core.model_package import check_inference_compatibility, ModelPackageManifest

    while True:
        print("\n  ── Verificações Manuais ────────────────────────")
        print("  a) Verificar ambiente e hardware")
        print("  b) Verificar dataset (dataset_ativo)")
        print("  c) Verificar dataset (dataset_refino)")
        print("  d) Verificar pacote base (modelos-base/)")
        print("  v) Voltar ao menu principal")
        print()

        choice = _ask("Escolha", valid={"a", "b", "c", "d", "v"})

        if choice == "v":
            break

        elif choice == "a":
            env = detect_environment()
            env.imprimir_resumo()
            ok, msg = validate_training_prerequisites(env)
            status = "APTO" if ok else "BLOQUEADO"
            print(f"\n  Status para treino: {status} — {msg}")

        elif choice == "b":
            dataset_path = _ROOT / "datasets" / "dataset_ativo"
            ok, lines = validate_dataset(dataset_path)
            for line in lines:
                print(f"  {line}")
            print(f"\n  Resultado: {'OK' if ok else 'PROBLEMAS ENCONTRADOS'}")

        elif choice == "c":
            dataset_path = _ROOT / "datasets" / "dataset_refino"
            ok, lines = validate_dataset(dataset_path)
            for line in lines:
                print(f"  {line}")
            print(f"\n  Resultado: {'OK' if ok else 'PROBLEMAS ENCONTRADOS'}")

        elif choice == "d":
            modelos_base_dir = _ROOT / "modelos-base"
            packages = list_base_packages(modelos_base_dir)
            if not packages:
                print(f"  Nenhum pacote encontrado em: {modelos_base_dir}")
            else:
                for i, pkg in enumerate(packages, 1):
                    print(f"\n  [{i}] {pkg.name}")
                    ok, issues, manifest = validate_base_package(pkg)
                    if ok:
                        print(f"       OK — {manifest.name} | "
                              f"{manifest.classes.nc} classes | "
                              f"modo: {manifest.training_mode}")
                    else:
                        for issue in issues:
                            print(f"       ERRO: {issue}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de UI
# ─────────────────────────────────────────────────────────────────────────────

def _print_header():
    print()
    print("=" * 62)
    print("  RecycleAI — Software de Treinamento YOLOv5s  v1.0.0")
    print("=" * 62)
    print()


def _section(title: str):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


def _block(msg: str):
    print(f"\n  [BLOQUEADO] {msg}")


def _done():
    print("\n" + "=" * 62)
    print("  Concluído com sucesso!")
    print("=" * 62)


def _ask(prompt: str, valid: set[str]) -> str:
    while True:
        try:
            val = input(f"  {prompt}: ").strip().lower()
            if val in valid:
                return val
            print(f"  Opção inválida. Escolha entre: {', '.join(sorted(valid))}")
        except (EOFError, KeyboardInterrupt):
            print("\n  Operação cancelada.")
            sys.exit(0)


def _ask_int(prompt: str, min_val: int, max_val: int) -> int:
    while True:
        try:
            val = input(f"  {prompt} [{min_val}-{max_val}]: ").strip()
            n = int(val)
            if min_val <= n <= max_val:
                return n
            print(f"  Valor fora do intervalo [{min_val}-{max_val}].")
        except ValueError:
            print("  Número inteiro inválido.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Operação cancelada.")
            sys.exit(0)


def _confirm(prompt: str) -> bool:
    try:
        resp = input(f"{prompt} [S/n]: ").strip().lower()
        return resp not in ("n", "nao", "não", "no")
    except (EOFError, KeyboardInterrupt):
        return False


def _build_args_namespace():
    """Retorna namespace de args com valores padrão para uso no menu (sem CLI)."""
    import argparse
    ns = argparse.Namespace()
    ns.config    = None
    ns.epochs    = None
    ns.batch     = None
    ns.img_size  = 640
    ns.name      = None
    ns.dry_run   = False
    ns.no_export = False
    return ns


def _apply_config_overrides(cfg, mode: str):
    """
    Permite ao usuário ajustar parâmetros antes de iniciar o treino/refino.
    Retorna a configuração possivelmente modificada.
    """
    mode_label = "refinamento" if mode == "refinement" else "treinamento"
    print(f"\n  Deseja ajustar parâmetros do {mode_label}? [s/N]: ", end="")
    try:
        resp = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return cfg

    if resp not in ("s", "sim", "y", "yes"):
        return cfg

    print()
    cfg = _override_field(cfg, "epochs",     "Épocas",      int)
    cfg = _override_field(cfg, "batch_size", "Batch size",  int)
    cfg = _override_field(cfg, "lr0",        "LR inicial",  float)
    cfg = _override_field(cfg, "name",       "Nome do experimento", str)

    print("\n  Configuração atualizada:")
    cfg.imprimir_resumo()
    return cfg


def _override_field(cfg, attr: str, label: str, typ):
    current = getattr(cfg, attr)
    try:
        raw = input(f"  {label} [{current}]: ").strip()
        if raw:
            setattr(cfg, attr, typ(raw))
    except (ValueError, EOFError, KeyboardInterrupt):
        pass
    return cfg


def _show_package_summary(pkg_dir: Path, index: int):
    manifest_path = pkg_dir / "manifest.json"
    try:
        from core.model_package import ModelPackageManifest
        m = ModelPackageManifest.load(manifest_path)
        mode_tag = " [refinamento]" if m.training_mode == "refinement" else ""
        classes_str = ", ".join(m.classes.names) if m.classes else "?"
        print(f"  [{index}] {pkg_dir.name}{mode_tag}")
        print(f"       Criado em   : {m.created_at}")
        print(f"       Classes     : {classes_str}")
        if m.training and m.training.map50:
            print(f"       mAP50       : {m.training.map50:.4f}")
        print()
    except Exception:
        print(f"  [{index}] {pkg_dir.name}  (manifest inválido ou ilegível)")
        print()
