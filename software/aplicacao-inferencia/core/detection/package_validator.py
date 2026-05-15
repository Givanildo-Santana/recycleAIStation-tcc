"""
Validação de pacotes completos de modelo RecycleAI.

Um pacote válido é uma pasta <nome>_package/ contendo:
  manifest.json           — metadados e contrato do pacote
  weights/best_ts.pt      — arquivo TorchScript para deploy (caminho em manifest.model.deploy_file)
  weights/best.pt         — pesos originais PyTorch (opcional, para refinamento futuro)
  config/data.yaml        — definição de classes (opcional)
  results/                — métricas de treinamento (opcional)

Esta aplicação é SOMENTE de operação. Aceita apenas pacotes prontos
produzidos pela aplicação de treinamento RecycleAI (pipeline v1.0+).

Validação em dois níveis:
  Nível 1 (rápido, sem GPU): manifest existe e é válido, estrutura de arquivos correta
  Nível 2 (lento, requer torch): torch.jit.load no deploy_file

Contrato aceito:
  schema_version  = "recycleai-pkg-v1"
  architecture    = "yolov5s"          (ou framework para v1.0)
  deploy_format   = "torchscript"      (ou format para v1.0)
  pipeline_version >= "1.0"
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Contrato aceito ───────────────────────────────────────────────────────────

_ACCEPTED_SCHEMA     = "recycleai-pkg-v1"
_ACCEPTED_ARCH       = "yolov5s"
_ACCEPTED_DEPLOY_FMT = "torchscript"
_PIPELINE_MIN        = "1.0"
_MIN_SIZE_MB         = 1.0


# ── Resultado da validação ────────────────────────────────────────────────────

@dataclass
class PackageValidationResult:
    ok: bool
    detail: str       = ""
    pkg_name: str     = ""
    deploy_file: str  = ""          # relativo à pasta do pacote (ex: "weights/best_ts.pt")
    deploy_path: Optional[Path] = None   # absoluto
    nc: int           = 0
    class_names: list = field(default_factory=list)
    size_mb: float    = 0.0


# ── Validação principal ───────────────────────────────────────────────────────

def validate_package(pkg_dir: str | Path, deep: bool = True) -> PackageValidationResult:
    """
    Valida uma pasta de pacote RecycleAI.

    Args:
        pkg_dir: caminho para a pasta <nome>_package/
        deep:    se True, executa torch.jit.load no deploy_file (lento)

    Returns:
        PackageValidationResult com ok=True se o pacote é utilizável.
    """
    pkg_dir = Path(pkg_dir)

    # ── Existência e tipo ─────────────────────────────────────────────────────
    if not pkg_dir.exists():
        return PackageValidationResult(
            ok=False,
            detail=f"Pasta não encontrada: {pkg_dir}",
        )
    if not pkg_dir.is_dir():
        return PackageValidationResult(
            ok=False,
            detail=f"Caminho informado não é uma pasta: {pkg_dir}",
        )

    # ── 1. Manifest ───────────────────────────────────────────────────────────
    manifest_path = pkg_dir / "manifest.json"
    if not manifest_path.exists():
        return PackageValidationResult(
            ok=False,
            detail=(
                "manifest.json não encontrado. "
                "Esta não é uma pasta de pacote RecycleAI válida. "
                "Certifique-se de selecionar a pasta <nome>_package/ gerada pela "
                "aplicação de treinamento."
            ),
        )

    try:
        with open(manifest_path, encoding="utf-8") as f:
            mdata = json.load(f)
    except json.JSONDecodeError as exc:
        return PackageValidationResult(
            ok=False, detail=f"manifest.json corrompido (JSON inválido): {exc}"
        )
    except Exception as exc:
        return PackageValidationResult(
            ok=False, detail=f"Erro ao ler manifest.json: {exc}"
        )

    # ── 2. Schema version ─────────────────────────────────────────────────────
    schema = mdata.get("schema_version", "")
    if schema != _ACCEPTED_SCHEMA:
        return PackageValidationResult(
            ok=False,
            detail=(
                f"schema_version '{schema}' não aceito. "
                f"Esperado: '{_ACCEPTED_SCHEMA}'."
            ),
        )

    # ── 3. Pipeline version ───────────────────────────────────────────────────
    pipeline = mdata.get("pipeline_version", "0.0")
    if not _pipeline_ok(pipeline):
        return PackageValidationResult(
            ok=False,
            detail=(
                f"pipeline_version '{pipeline}' abaixo do mínimo aceito "
                f"'{_PIPELINE_MIN}'."
            ),
        )

    pkg_name = mdata.get("name", pkg_dir.name)

    # ── 4. Arquitetura e formato de deploy ────────────────────────────────────
    model_data = mdata.get("model", {})

    # Aceita campos v1.1 (architecture/deploy_format) e v1.0 (framework/format)
    arch = model_data.get("architecture") or model_data.get("framework", "")
    deploy_fmt = model_data.get("deploy_format") or model_data.get("format", "")

    if arch != _ACCEPTED_ARCH:
        return PackageValidationResult(
            ok=False,
            detail=(
                f"Arquitetura '{arch}' não suportada. "
                f"Esperado: '{_ACCEPTED_ARCH}'."
            ),
            pkg_name=pkg_name,
        )

    if deploy_fmt != _ACCEPTED_DEPLOY_FMT:
        return PackageValidationResult(
            ok=False,
            detail=(
                f"Formato de deploy '{deploy_fmt}' não suportado. "
                f"Esperado: '{_ACCEPTED_DEPLOY_FMT}'."
            ),
            pkg_name=pkg_name,
        )

    # ── 5. Classes ────────────────────────────────────────────────────────────
    classes_data = mdata.get("classes", {})
    try:
        nc = int(classes_data.get("nc", 0))
    except (TypeError, ValueError):
        nc = 0
    class_names = list(classes_data.get("names", []))

    if nc == 0 or not class_names:
        return PackageValidationResult(
            ok=False,
            detail="Pacote sem informação de classes válida (nc=0 ou names vazio).",
            pkg_name=pkg_name,
        )

    # ── 6. Arquivo de deploy ──────────────────────────────────────────────────
    deploy_file = model_data.get("deploy_file", "")
    if not deploy_file:
        return PackageValidationResult(
            ok=False,
            detail="manifest.json não especifica deploy_file.",
            pkg_name=pkg_name,
        )

    # Normaliza separadores (manifest pode usar '/' ou '\')
    deploy_file = deploy_file.replace("\\", "/")
    deploy_path = pkg_dir / deploy_file

    if not deploy_path.exists():
        return PackageValidationResult(
            ok=False,
            detail=(
                f"Arquivo de deploy não encontrado: {deploy_path}. "
                f"Verifique se o pacote está completo."
            ),
            pkg_name=pkg_name,
        )

    size_mb = deploy_path.stat().st_size / (1024 * 1024)
    if size_mb < _MIN_SIZE_MB:
        return PackageValidationResult(
            ok=False,
            detail=(
                f"Arquivo de deploy muito pequeno ({size_mb:.2f} MB) "
                f"— pode estar corrompido."
            ),
            pkg_name=pkg_name,
        )

    # ── 7. Validação profunda (torch.jit.load) ────────────────────────────────
    if deep:
        try:
            import torch
            mdl = torch.jit.load(str(deploy_path), map_location="cpu")
            mdl.eval()
            if not callable(mdl):
                return PackageValidationResult(
                    ok=False,
                    detail="torch.jit.load OK mas modelo não é callable.",
                    pkg_name=pkg_name,
                    deploy_file=deploy_file,
                    deploy_path=deploy_path,
                    nc=nc,
                    class_names=class_names,
                    size_mb=size_mb,
                )
        except Exception as exc:
            return PackageValidationResult(
                ok=False,
                detail=f"torch.jit.load falhou: {exc}",
                pkg_name=pkg_name,
                deploy_file=deploy_file,
                deploy_path=deploy_path,
                nc=nc,
                class_names=class_names,
                size_mb=size_mb,
            )

    depth_tag = "validado com torch.jit.load" if deep else "validação superficial"
    return PackageValidationResult(
        ok=True,
        detail=(
            f"Pacote OK: {pkg_name} — {nc} classes — "
            f"{size_mb:.1f} MB ({depth_tag})"
        ),
        pkg_name=pkg_name,
        deploy_file=deploy_file,
        deploy_path=deploy_path,
        nc=nc,
        class_names=class_names,
        size_mb=size_mb,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pipeline_ok(pipeline_version: str) -> bool:
    """
    Retorna True se pipeline_version >= _PIPELINE_MIN.
    Compara como tuplas de inteiros: "1.1" >= "1.0" → True.
    """
    try:
        current = tuple(int(x) for x in str(pipeline_version).split("."))
        minimum = tuple(int(x) for x in _PIPELINE_MIN.split("."))
        return current >= minimum
    except Exception:
        return False
