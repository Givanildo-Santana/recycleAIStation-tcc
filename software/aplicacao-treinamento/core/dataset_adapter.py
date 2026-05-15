"""
Adaptador de dataset — suporte nativo a Roboflow e YOLO padrão.

Roboflow exporta com variações que o pipeline padrão não aceita:

  - Chave 'valid' em vez de 'val' no data.yaml
  - Diretório 'valid/' em vez de 'val/'
  - Chave 'path' definindo a raiz do dataset (caminhos de split são relativos a ela)
  - Caminhos relativos ao data.yaml: 'train/images', '../train/images', etc.
  - Arquivos auxiliares: README.dataset.txt, README.roboflow.txt

Este módulo normaliza todas essas variações para que o pipeline de treinamento
e o Ultralytics funcionem sem ajustes manuais no dataset.

─────────────────────────────────────────────────────────────────────────────
REGRAS DE RESOLUÇÃO DE CAMINHO
─────────────────────────────────────────────────────────────────────────────
1. Se data.yaml contém chave 'path': caminhos de split são relativos a ela
2. Caso contrário: caminhos são relativos ao diretório do data.yaml
3. Caminhos absolutos são usados diretamente
4. Diretório de labels é inferido substituindo 'images' por 'labels' no path
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml


_ROBOFLOW_ARTIFACTS = frozenset({"README.dataset.txt", "README.roboflow.txt"})
_VAL_KEY_ALIASES    = ("val", "valid")   # aceita ambos como split de validação
_IMAGE_EXTS         = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})


# ─────────────────────────────────────────────────────────────────────────────
# Tipos públicos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SplitInfo:
    """Informações resolvidas de um split do dataset."""
    name:       str    # nome canônico: "train", "val", "test"
    images_dir: Path   # caminho absoluto do diretório de imagens
    labels_dir: Path   # caminho absoluto do diretório de labels (pode não existir)


@dataclass
class DatasetInfo:
    """
    Dataset completamente resolvido, independente de variações de formato.
    Produzido por resolve_dataset().
    """
    root:             Path
    data_yaml:        Path
    nc:               int
    names:            list[str]
    train:            SplitInfo
    val:              SplitInfo
    test:             Optional[SplitInfo]
    roboflow_origin:  bool   # True se artefatos ou convenções Roboflow detectados


# ─────────────────────────────────────────────────────────────────────────────
# Resolução do dataset
# ─────────────────────────────────────────────────────────────────────────────

def resolve_dataset(dataset_path: Path) -> DatasetInfo:
    """
    Lê data.yaml e resolve os caminhos reais de todos os splits.

    Suporta:
      - Chaves 'val' ou 'valid' para o split de validação
      - Chave 'path' como raiz do dataset (padrão Roboflow e Ultralytics)
      - Caminhos relativos e absolutos
      - Detecção automática de origem Roboflow

    Raises:
        FileNotFoundError: data.yaml não encontrado
        ValueError:        data.yaml inválido ou splits obrigatórios ausentes
    """
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml não encontrado em: {dataset_path}")

    with open(yaml_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    nc    = int(meta.get("nc", 0))
    names = list(meta.get("names", []))

    # Função de resolução de caminhos, respeitando a chave 'path' se presente
    resolve = _make_resolver(meta, yaml_path.parent)

    # ── Split de treino ───────────────────────────────────────────────────────
    train_decl = meta.get("train")
    if not train_decl:
        raise ValueError("data.yaml não declara split 'train'")
    train_images = resolve(str(train_decl))
    train = SplitInfo("train", train_images, _infer_labels_dir(train_images))

    # ── Split de validação (val ou valid) ─────────────────────────────────────
    val_decl = None
    for key in _VAL_KEY_ALIASES:
        if key in meta:
            val_decl = meta[key]
            break
    if not val_decl:
        raise ValueError("data.yaml não declara split 'val' nem 'valid'")
    val_images = resolve(str(val_decl))
    val = SplitInfo("val", val_images, _infer_labels_dir(val_images))

    # ── Split de teste (opcional) ─────────────────────────────────────────────
    test: Optional[SplitInfo] = None
    test_decl = meta.get("test")
    if test_decl:
        test_images = resolve(str(test_decl))
        test = SplitInfo("test", test_images, _infer_labels_dir(test_images))

    # ── Detectar origem Roboflow ──────────────────────────────────────────────
    roboflow_origin = (
        any((dataset_path / a).exists() for a in _ROBOFLOW_ARTIFACTS)
        or "valid" in meta
    )

    return DatasetInfo(
        root=dataset_path,
        data_yaml=yaml_path,
        nc=nc,
        names=names,
        train=train,
        val=val,
        test=test,
        roboflow_origin=roboflow_origin,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Normalização do data.yaml para o Ultralytics
# ─────────────────────────────────────────────────────────────────────────────

def get_normalized_data_yaml(
    dataset_info: DatasetInfo,
    target_dir: Path,
) -> Path:
    """
    Retorna caminho de um data.yaml compatível com o Ultralytics.

    Se o dataset já está no formato padrão (chave 'val', sem 'valid'),
    retorna o data.yaml original sem criar arquivo adicional.

    Caso contrário, escreve um data.yaml normalizado em target_dir com:
      - Chave 'val' em vez de 'valid'
      - Caminhos absolutos para todos os splits (seguro independente de CWD)

    Args:
        dataset_info: resultado de resolve_dataset()
        target_dir:   diretório onde gravar o yaml normalizado

    Returns:
        Path do yaml a ser passado ao Ultralytics.
    """
    with open(dataset_info.data_yaml, encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    needs_norm = "valid" in meta and "val" not in meta

    if not needs_norm:
        return dataset_info.data_yaml

    # Cria yaml normalizado
    target_dir.mkdir(parents=True, exist_ok=True)
    norm_path = target_dir / "data_normalized.yaml"

    normalized: dict = {}
    # Preserva metadados (nc, names, etc.) exceto as chaves de split
    for k, v in meta.items():
        if k not in ("train", "val", "valid", "test", "path"):
            normalized[k] = v

    # Usa caminhos absolutos resolvidos — seguro independente do CWD do processo
    normalized["train"] = str(dataset_info.train.images_dir)
    normalized["val"]   = str(dataset_info.val.images_dir)
    if dataset_info.test:
        normalized["test"] = str(dataset_info.test.images_dir)

    with open(norm_path, "w", encoding="utf-8") as f:
        yaml.dump(normalized, f, allow_unicode=True, sort_keys=False)

    return norm_path


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários públicos
# ─────────────────────────────────────────────────────────────────────────────

def count_train_images(dataset_path: Path) -> int:
    """
    Conta imagens no split de treino usando resolução via data.yaml.
    Retorna 0 em caso de erro (sem bloquear a execução).
    """
    try:
        info = resolve_dataset(dataset_path)
        img_dir = info.train.images_dir
        if img_dir.exists():
            return sum(
                1 for f in img_dir.iterdir()
                if f.suffix.lower() in _IMAGE_EXTS
            )
    except Exception:
        pass
    # Fallback: path hardcoded para datasets sem data.yaml de treino declarado
    try:
        return sum(
            1 for f in (dataset_path / "train" / "images").iterdir()
            if f.suffix.lower() in _IMAGE_EXTS
        )
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _make_resolver(meta: dict, yaml_dir: Path) -> Callable[[str], Path]:
    """
    Cria função de resolução de caminhos respeitando a chave 'path' do yaml.

    Se 'path' está presente: caminhos de split são relativos a ela.
    Caso contrário: relativos ao diretório do data.yaml.
    """
    path_key = meta.get("path")
    if path_key:
        base = Path(path_key)
        if not base.is_absolute():
            base = (yaml_dir / base).resolve()
    else:
        base = yaml_dir

    def resolve(declared: str) -> Path:
        p = Path(declared)
        if p.is_absolute():
            return p
        return (base / p).resolve()

    return resolve


def _infer_labels_dir(images_dir: Path) -> Path:
    """
    Infere o diretório de labels a partir do diretório de imagens.

    Substitui a última ocorrência de 'images' no path por 'labels'.
    Se não houver componente 'images', usa um subdiretório 'labels'
    dentro do mesmo pai.

    Exemplos:
      .../train/images  → .../train/labels
      .../valid/images  → .../valid/labels
      .../train         → .../train/labels  (fallback)
    """
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    # Sem componente 'images': labels como subdiretório do mesmo nível
    return images_dir / "labels"
