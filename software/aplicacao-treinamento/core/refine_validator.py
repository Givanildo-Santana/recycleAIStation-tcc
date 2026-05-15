"""
Validação de pacote base e compatibilidade para refinamento (fine-tuning).

Regras de bloqueio do refinamento:
  1. Pacote base deve existir e conter manifest.json válido
  2. schema_version == "recycleai-pkg-v1" e pipeline_version >= "1.0"
  3. model.framework == "yolov5s"
  4. weights/best.pt deve existir (necessário para continuar o treino)
  5. Dataset de refinamento deve ser estrutura YOLO válida
  6. Classes e ordem de classes devem ser idênticas entre pacote base e dataset
  7. GPU dedicada utilizável (verificado pela cadeia de validators existente)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from core.model_package import (
    ModelPackageManifest,
    check_inference_compatibility,
    REQUIRED_FRAMEWORK,
)


# ─────────────────────────────────────────────────────────────────────────────
# Validação do pacote base
# ─────────────────────────────────────────────────────────────────────────────

def validate_base_package(
    package_dir: Path,
) -> tuple[bool, list[str], Optional[ModelPackageManifest]]:
    """
    Valida um pacote de modelo base para uso em refinamento.

    Returns:
        (ok, issues, manifest_ou_None)
        ok=False bloqueia o refinamento; manifest é None se ok=False.
    """
    issues: list[str] = []

    # ── Existência do diretório ───────────────────────────────────────────────
    if not package_dir.exists():
        return False, [f"Pacote base não encontrado: {package_dir}"], None
    if not package_dir.is_dir():
        return False, [f"Caminho não é um diretório: {package_dir}"], None

    # ── manifest.json ─────────────────────────────────────────────────────────
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        return False, [f"manifest.json ausente em: {package_dir}"], None

    try:
        manifest = ModelPackageManifest.load(manifest_path)
    except Exception as exc:
        return False, [f"Erro ao carregar manifest.json: {exc}"], None

    # ── Compatibilidade de schema/pipeline (reutiliza contrato existente) ─────
    compat_ok, compat_issues = check_inference_compatibility(manifest, package_dir)
    if not compat_ok:
        return False, compat_issues, None

    # ── best.pt obrigatório para refinamento ──────────────────────────────────
    # (best_ts.pt = TorchScript, não permite continuar treino)
    best_pt = package_dir / "weights" / "best.pt"
    if not best_pt.exists():
        issues.append(
            "weights/best.pt não encontrado no pacote base. "
            "O refinamento requer os pesos PyTorch originais (.pt), "
            "não apenas o TorchScript de deploy (.torchscript)."
        )
        return False, issues, None

    return True, [], manifest


# ─────────────────────────────────────────────────────────────────────────────
# Validação de compatibilidade: pacote base ↔ dataset de refinamento
# ─────────────────────────────────────────────────────────────────────────────

def validate_refinement_compatibility(
    base_manifest: ModelPackageManifest,
    refine_dataset_path: Path,
) -> tuple[bool, list[str]]:
    """
    Verifica compatibilidade entre o pacote base e o dataset de refinamento.

    Requisitos:
      - Arquitetura yolov5s no pacote base
      - Classes e ordem idênticas entre pacote base e data.yaml do refinamento

    Returns:
        (ok, issues)
    """
    issues: list[str] = []
    ok = True

    # ── Arquitetura ───────────────────────────────────────────────────────────
    if base_manifest.model and base_manifest.model.framework != REQUIRED_FRAMEWORK:
        issues.append(
            f"Arquitetura incompatível: pacote base usa "
            f"'{base_manifest.model.framework}', esperado '{REQUIRED_FRAMEWORK}'."
        )
        ok = False

    # ── Ler classes do dataset de refinamento ────────────────────────────────
    data_yaml = refine_dataset_path / "data.yaml"
    if not data_yaml.exists():
        issues.append(f"data.yaml não encontrado em: {refine_dataset_path}")
        return False, issues

    try:
        with open(data_yaml, encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        refine_nc    = int(meta.get("nc", 0))
        refine_names = list(meta.get("names", []))
    except Exception as exc:
        issues.append(f"Erro ao ler data.yaml do dataset de refinamento: {exc}")
        return False, issues

    # ── Comparar com classes do pacote base ───────────────────────────────────
    if not base_manifest.classes:
        issues.append("Pacote base não tem informações de classes no manifesto.")
        return False, issues

    base_nc    = base_manifest.classes.nc
    base_names = base_manifest.classes.names

    if refine_nc != base_nc:
        issues.append(
            f"Número de classes incompatível:\n"
            f"  Pacote base      : {base_nc} classes\n"
            f"  Dataset refino   : {refine_nc} classes\n"
            "  O refinamento exige o mesmo número de classes."
        )
        ok = False

    if refine_names != base_names:
        issues.append(
            f"Classes incompatíveis:\n"
            f"  Pacote base      : {base_names}\n"
            f"  Dataset refino   : {refine_names}\n"
            "  As classes devem ser idênticas e na mesma ordem."
        )
        ok = False

    return ok, issues


# ─────────────────────────────────────────────────────────────────────────────
# Utilitário: lista pacotes disponíveis em modelos-base/
# ─────────────────────────────────────────────────────────────────────────────

def list_base_packages(modelos_base_dir: Path) -> list[Path]:
    """
    Retorna lista de subdiretórios de modelos-base/ que contêm manifest.json.
    Ordenados por data de modificação (mais recente primeiro).
    """
    if not modelos_base_dir.exists():
        return []
    packages = [
        d for d in sorted(modelos_base_dir.iterdir())
        if d.is_dir() and (d / "manifest.json").exists()
    ]
    return sorted(packages, key=lambda p: p.stat().st_mtime, reverse=True)
