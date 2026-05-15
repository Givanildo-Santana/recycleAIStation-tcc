"""
Configuração de treinamento.

Gera configuração sugerida automaticamente baseada em:
  - VRAM disponível (batch size, img_size)
  - Tamanho do dataset (épocas sugeridas)
  - Ambiente (Colab/pod: sem limite de tempo; local: modo conservador)

Permite override total ou parcial por:
  - Arquivo YAML (--config)
  - Argumentos CLI (--epochs, --batch, --img-size, --name)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from core.environment import EnvironmentReport


# ─────────────────────────────────────────────────────────────────────────────
# Contrato de configuração
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Identificação do experimento
    name:         str
    timestamp:    str

    # Caminhos
    dataset_path: Path
    runs_dir:     Path
    exports_dir:  Path

    # Modelo
    model_arch:   str = "yolov5s"        # fixo nesta versão

    # Hiperparâmetros
    epochs:       int   = 50
    batch_size:   int   = 16
    img_size:     int   = 640
    lr0:          float = 0.01           # learning rate inicial
    workers:      int   = 4
    device:       str   = "0"            # "0" = GPU 0, "cpu" = CPU
    patience:     int   = 50             # early stopping
    cache:        bool  = False          # cache imagens em RAM

    # Augmentação (valores padrão YOLOv5)
    augment:      bool  = True

    # Export
    export_torchscript: bool = True      # obrigatório para inferência
    export_onnx:        bool = False

    # ── Modo de treinamento ───────────────────────────────────────────────────
    training_mode: str = "training"       # "training" | "refinement"

    # ── Campos de refinamento (preenchidos apenas em modo "refinement") ───────
    pretrained_weights:       Optional[Path] = None  # best.pt do pacote base
    parent_package_name:      str = ""
    parent_package_created_at: str = ""
    parent_package_dataset:   str = ""

    def imprimir_resumo(self):
        mode_label = "Refinamento" if self.training_mode == "refinement" else "Treinamento"
        print(f"  Modo        : {mode_label}")
        print(f"  Experimento : {self.name}")
        if self.training_mode == "refinement" and self.parent_package_name:
            print(f"  Base        : {self.parent_package_name}")
        print(f"  Modelo      : {self.model_arch}")
        print(f"  Épocas      : {self.epochs}")
        print(f"  Batch       : {self.batch_size}")
        print(f"  LR inicial  : {self.lr0}")
        print(f"  Img-size    : {self.img_size}")
        print(f"  Device      : {self.device}")
        print(f"  Dataset     : {self.dataset_path}")
        print(f"  Saída       : {self.runs_dir / self.name}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dataset_path"] = str(self.dataset_path)
        d["runs_dir"]     = str(self.runs_dir)
        d["exports_dir"]  = str(self.exports_dir)
        return d


class ConfigError(ValueError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Lógica de sugestão automática
# ─────────────────────────────────────────────────────────────────────────────

def _suggest_batch(vram_gb: float, img_size: int) -> int:
    """Sugere batch size baseado em VRAM e img_size."""
    # Heurística conservadora para YOLOv5s (mínimo garantido: 4 GB VRAM)
    table = {
        (4, 640):  8,
        (6, 640): 16,
        (8, 640): 32,
        (12, 640): 32,
        (16, 640): 64,
        (24, 640): 64,
    }
    # Encontra entrada mais próxima
    vram_int = max(4, min(24, int(vram_gb)))
    for vram_key in sorted(table.keys(), key=lambda x: x[0], reverse=True):
        if vram_int >= vram_key[0]:
            return table[vram_key]
    return 4


def _suggest_epochs(dataset_path: Path) -> int:
    """
    Sugere número de épocas baseado no tamanho do dataset.
    Usa dataset_adapter para resolver o caminho correto do split de treino,
    compatível com Roboflow (valid/) e YOLO padrão (val/).
    """
    try:
        from core.dataset_adapter import count_train_images
        n = count_train_images(dataset_path)
    except Exception:
        n = 0
    if n == 0:    return 50
    if n < 100:   return 150
    if n < 500:   return 100
    if n < 2000:  return 80
    return 50


def _suggest_workers(runtime: str) -> int:
    if runtime in ("colab", "runpod", "pod"):
        return 8
    return 4


# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent


def build_config(
    env: EnvironmentReport,
    dataset_path: Path,
    args: argparse.Namespace,
) -> TrainingConfig:
    """
    Monta a configuração sugerida e aplica overrides dos argumentos CLI.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"recycleai_{timestamp}"

    # Detecta VRAM da GPU primária
    vram_gb = 0.0
    if env.primary_gpu:
        vram_gb = env.primary_gpu.vram_livre_gb

    img_size   = args.img_size or 640
    batch_size = args.batch    or _suggest_batch(vram_gb, img_size)
    epochs     = args.epochs   or _suggest_epochs(dataset_path)
    workers    = _suggest_workers(env.runtime)
    device     = "0" if env.training_viable else "cpu"

    return TrainingConfig(
        name         = name,
        timestamp    = timestamp,
        dataset_path = dataset_path,
        runs_dir     = _ROOT / "runs",
        exports_dir  = _ROOT / "exports",
        epochs       = epochs,
        batch_size   = batch_size,
        img_size     = img_size,
        workers      = workers,
        device       = device,
    )


def build_refine_config(
    env: EnvironmentReport,
    base_manifest: "ModelPackageManifest",  # noqa: F821 — evita import circular
    base_package_dir: Path,
    refine_dataset_path: Path,
    args: argparse.Namespace,
) -> TrainingConfig:
    """
    Monta configuração de refinamento a partir de um pacote base existente.

    Diferenças em relação ao treinamento normal:
      - pretrained_weights = best.pt do pacote base (continua treino a partir daí)
      - lr0 reduzido (0.001 padrão para fine-tuning)
      - epochs padrão menor (50 ou sugerido pelo dataset)
      - training_mode = "refinement"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = base_manifest.name or "modelo_base"
    name = args.name or f"{base_name}_refined_{timestamp}"

    vram_gb = env.primary_gpu.vram_livre_gb if env.primary_gpu else 0.0
    img_size   = args.img_size or (base_manifest.model.img_size if base_manifest.model else 640)
    batch_size = args.batch    or _suggest_batch(vram_gb, img_size)
    epochs     = args.epochs   or _suggest_refine_epochs(refine_dataset_path)
    workers    = _suggest_workers(env.runtime)
    device     = "0" if env.training_viable else "cpu"

    return TrainingConfig(
        name               = name,
        timestamp          = timestamp,
        dataset_path       = refine_dataset_path,
        runs_dir           = _ROOT / "runs",
        exports_dir        = _ROOT / "exports",
        epochs             = epochs,
        batch_size         = batch_size,
        img_size           = img_size,
        lr0                = 0.001,         # LR reduzido para fine-tuning
        workers            = workers,
        device             = device,
        training_mode      = "refinement",
        pretrained_weights = base_package_dir / "weights" / "best.pt",
        parent_package_name       = base_manifest.name,
        parent_package_created_at = base_manifest.created_at,
        parent_package_dataset    = base_manifest.dataset_origin,
    )


def _suggest_refine_epochs(dataset_path: Path) -> int:
    """
    Sugere número de épocas para refinamento (tipicamente menor que treino completo).
    Usa dataset_adapter para resolver o caminho correto do split de treino.
    """
    try:
        from core.dataset_adapter import count_train_images
        n = count_train_images(dataset_path)
    except Exception:
        n = 0
    if n == 0:   return 50
    if n < 100:  return 80
    if n < 500:  return 50
    return 30


def load_config_override(
    base: TrainingConfig,
    config_file: Path,
) -> TrainingConfig:
    """
    Aplica overrides de um arquivo YAML sobre a configuração base.
    Apenas os campos presentes no arquivo sobrescrevem o valor sugerido.
    """
    if not config_file.exists():
        raise ConfigError(f"Arquivo de configuração não encontrado: {config_file}")

    with open(config_file, encoding="utf-8") as f:
        overrides: dict = yaml.safe_load(f) or {}

    mapping = {
        "name":       ("name",       str),
        "epochs":     ("epochs",     int),
        "batch":      ("batch_size", int),
        "batch_size": ("batch_size", int),
        "img_size":   ("img_size",   int),
        "img-size":   ("img_size",   int),
        "workers":    ("workers",    int),
        "device":     ("device",     str),
        "patience":   ("patience",   int),
        "cache":      ("cache",      bool),
        "augment":    ("augment",    bool),
        "export_torchscript": ("export_torchscript", bool),
        "export_onnx":        ("export_onnx",        bool),
    }

    for yaml_key, (attr, typ) in mapping.items():
        if yaml_key in overrides:
            setattr(base, attr, typ(overrides[yaml_key]))

    return base
