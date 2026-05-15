"""
Carrega o modelo TorchScript ativo registrado no model_registry.

Ponto unico de carregamento para o pipeline de inferencia.
Cache em memoria: recarrega apenas se o caminho do modelo ativo mudar.
"""
from __future__ import annotations

from pathlib import Path

_cached_model = None
_cached_path: Path | None = None


def load_active(device: str = "cpu"):
    """
    Retorna o modelo TorchScript ativo (com cache em memoria).
    Lanca RuntimeError se nao houver modelo ativo ou o arquivo estiver ausente.
    """
    global _cached_model, _cached_path

    import torch
    from core.detection.model_registry import get_active_path

    path = get_active_path()
    if path is None:
        raise RuntimeError("Nenhum modelo ativo registrado no banco.")

    if _cached_model is not None and _cached_path == path:
        return _cached_model

    if not path.exists():
        raise RuntimeError(f"Arquivo do modelo nao encontrado: {path}")

    model = torch.jit.load(str(path), map_location=device)
    model.eval()

    _cached_model = model
    _cached_path = path
    return model


def get_active_info() -> dict:
    """Retorna metadados do modelo ativo sem carrega-lo na memoria."""
    from core.detection.model_registry import get_active_path, get_active_classes
    from db.repositories import model_repo

    row = model_repo.get_active()
    if row is None:
        return {}
    return {
        "id": row["id"],
        "name": row["name"],
        "file_path": row["file_path"],
        "path": get_active_path(),
        "class_names": get_active_classes(),
        "nc": row["nc"],
    }


def invalidate_cache():
    """Forca recarga na proxima chamada (ex: apos troca de modelo ativo)."""
    global _cached_model, _cached_path
    _cached_model = None
    _cached_path = None
