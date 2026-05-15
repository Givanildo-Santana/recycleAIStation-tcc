"""
Validação de modelos operacionais.

Esta aplicação é SOMENTE de operação — sem treinamento.
A validação verifica se um modelo artefato externo é compatível
com o pipeline de inferência existente.

Níveis de validação:
  Nível 1 (rápido, sempre): arquivo existe, extensão correta, tamanho mínimo
  Nível 2 (lento, opcional): tenta torch.jit.load e verifica se é chamável

Nível 2 não é executado por padrão para não atrasar o boot.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_MIN_SIZE_MB = 1.0
_SUPPORTED_FORMATS = {
    ".pt":  "torchscript",
    ".pth": "torchscript",
    ".onnx": "onnx",
}


@dataclass
class ValidationResult:
    ok: bool
    fmt: str = ""        # formato detectado: torchscript | onnx | desconhecido
    size_mb: float = 0.0
    detail: str = ""
    deep_ok: bool | None = None   # None = não testado


def validate(file_path: str | Path, deep: bool = False) -> ValidationResult:
    """
    Valida um modelo operacional.

    Args:
        file_path: caminho para o arquivo do modelo (absoluto ou relativo à raiz).
        deep:      se True, tenta carregar via torch.jit.load (lento).

    Returns:
        ValidationResult com ok=True se compatível.
    """
    path = _resolve(file_path)

    # ── Nível 1: existência, extensão, tamanho ──────────────────────────
    if not path.exists():
        return ValidationResult(ok=False, detail=f"Arquivo não encontrado: {path}")

    ext = path.suffix.lower()
    if ext not in _SUPPORTED_FORMATS:
        return ValidationResult(
            ok=False,
            detail=f"Extensão '{ext}' não suportada. Use: {list(_SUPPORTED_FORMATS)}"
        )

    fmt = _SUPPORTED_FORMATS[ext]
    size_mb = path.stat().st_size / (1024 * 1024)

    if size_mb < _MIN_SIZE_MB:
        return ValidationResult(
            ok=False, fmt=fmt, size_mb=size_mb,
            detail=f"Arquivo muito pequeno ({size_mb:.2f} MB) — pode estar corrompido"
        )

    if not deep:
        return ValidationResult(
            ok=True, fmt=fmt, size_mb=size_mb,
            detail=f"{path.name} ({size_mb:.1f} MB) — validação superficial OK",
            deep_ok=None,
        )

    # ── Nível 2: carregamento real ──────────────────────────────────────
    return _deep_validate(path, fmt, size_mb)


def _deep_validate(path: Path, fmt: str, size_mb: float) -> ValidationResult:
    if fmt != "torchscript":
        return ValidationResult(
            ok=True, fmt=fmt, size_mb=size_mb,
            detail=f"Validação profunda não implementada para formato '{fmt}'",
            deep_ok=None,
        )
    try:
        import torch
        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
        deep_ok = callable(model)
        return ValidationResult(
            ok=deep_ok, fmt=fmt, size_mb=size_mb,
            detail=f"torch.jit.load OK — modelo callable={deep_ok}",
            deep_ok=deep_ok,
        )
    except Exception as exc:
        return ValidationResult(
            ok=False, fmt=fmt, size_mb=size_mb,
            detail=f"torch.jit.load falhou: {exc}",
            deep_ok=False,
        )


def _resolve(file_path: str | Path) -> Path:
    from core.utils.paths import resolve_model_path
    return resolve_model_path(str(file_path))
