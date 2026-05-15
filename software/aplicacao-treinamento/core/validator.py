"""
Validação de pré-requisitos e dataset.

─────────────────────────────────────────────────────────────────────────────
POLÍTICA DE HARDWARE (treinamento) — NVIDIA only
─────────────────────────────────────────────────────────────────────────────
Treinamento YOLOv5s é permitido APENAS quando os três critérios abaixo são
satisfeitos simultaneamente:

  Critério 1 — GPU NVIDIA detectada?
    Não → BLOQUEADO: qualquer outra marca (AMD, Intel, etc.) é recusada.

  Critério 2 — CUDA funcional?
    Não → BLOQUEADO: NVIDIA sem CUDA ativo não é aceita.

  Critério 3 — VRAM ≥ 4 GB?
    < 4 GB → BLOQUEADO: VRAM insuficiente para YOLOv5s.
    ≥ 4 GB → APROVADO.

Cenários explicitamente bloqueados:
  - AMD (qualquer backend, incluindo ROCm)
  - Intel (qualquer backend, incluindo XPU)
  - CPU-only
  - NVIDIA sem CUDA funcional
  - NVIDIA com VRAM < 4 GB
  - Qualquer ambiente que não comprove os três critérios acima

─────────────────────────────────────────────────────────────────────────────
ESTRUTURA DE DATASET SUPORTADA
─────────────────────────────────────────────────────────────────────────────
Suporta datasets YOLO padrão E datasets exportados pelo Roboflow:

  Formato YOLO padrão:
    data.yaml  (val: val/images)
    train/images/  train/labels/
    val/images/    val/labels/
    test/images/   test/labels/   ← opcional

  Formato Roboflow:
    data.yaml  (valid: valid/images  ← 'valid' aceito como equivalente de 'val')
    train/images/  train/labels/
    valid/images/  valid/labels/
    test/images/   test/labels/   ← opcional
    README.dataset.txt             ← ignorado

A validação lê os caminhos declarados em data.yaml e os resolve antes de
verificar a existência de imagens e labels. Não assume nomes fixos de pasta.
"""
from __future__ import annotations

from pathlib import Path

from core.environment import EnvironmentReport


# ─────────────────────────────────────────────────────────────────────────────
# Política de hardware — NVIDIA only
# ─────────────────────────────────────────────────────────────────────────────

_VRAM_MINIMA_GB = 4.0   # mínimo obrigatório para YOLOv5s


def validate_training_prerequisites(
    env: EnvironmentReport,
) -> tuple[bool, str]:
    """
    Verifica se o ambiente atende aos três critérios obrigatórios para
    executar treinamento YOLOv5s: GPU NVIDIA + CUDA funcional + VRAM ≥ 4 GB.

    Retorna:
        (ok, mensagem)  — ok=False bloqueia o início do treinamento.
    """
    # ── 1. Alguma GPU detectada? ──────────────────────────────────────────────
    if not env.gpus:
        return False, (
            "Nenhuma GPU detectada. "
            "Treinamento YOLOv5s requer GPU NVIDIA com CUDA e mínimo de 4 GB de VRAM. "
            "Considere Google Colab (T4, 15 GB) ou RunPod."
        )

    # ── 2. GPU NVIDIA presente? ───────────────────────────────────────────────
    nvidia_gpus = [g for g in env.gpus if g.vendor == "NVIDIA"]
    if not nvidia_gpus:
        detected = ", ".join(f"{g.name} ({g.vendor})" for g in env.gpus)
        return False, (
            f"GPU detectada: {detected}. "
            "Somente GPU NVIDIA é suportada. "
            "AMD e Intel não são suportadas nesta versão, independentemente do backend disponível. "
            "Considere Google Colab (T4, 15 GB) ou RunPod com instância NVIDIA."
        )

    # ── 3. CUDA funcional? ────────────────────────────────────────────────────
    cuda_gpus = [g for g in nvidia_gpus if g.backend == "cuda"]
    if not cuda_gpus:
        detected = ", ".join(g.name for g in nvidia_gpus)
        return False, (
            f"GPU NVIDIA detectada ({detected}), mas CUDA não está funcional. "
            "Verifique a instalação do driver NVIDIA e do PyTorch com suporte a CUDA. "
            "Execute: pip install torch --index-url https://download.pytorch.org/whl/cu118"
        )

    # ── 4. VRAM suficiente? ───────────────────────────────────────────────────
    primary = cuda_gpus[0]
    if primary.vram_total_gb < _VRAM_MINIMA_GB:
        return False, (
            f"VRAM insuficiente: {primary.vram_total_gb:.1f} GB "
            f"(mínimo obrigatório: {_VRAM_MINIMA_GB:.0f} GB). "
            f"GPU: {primary.name}. "
            "Considere Google Colab (T4, 15 GB) ou RunPod com GPU compatível."
        )

    vram_tag = f"{primary.vram_livre_gb:.1f} GB livre / {primary.vram_total_gb:.1f} GB total"
    return True, f"GPU: {primary.name} (NVIDIA, CUDA, {vram_tag})."


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})


def validate_dataset(dataset_path: Path) -> tuple[bool, list[str]]:
    """
    Valida a estrutura do dataset YOLO, com suporte nativo a Roboflow.

    Usa dataset_adapter.resolve_dataset() para ler os caminhos reais dos
    splits a partir do data.yaml, aceitando 'valid' como equivalente de
    'val' e respeitando a chave 'path' do Roboflow.

    Returns:
        (ok, lines): ok indica se o dataset é utilizável.
                     lines é uma lista de linhas de relatório para exibição.
    """
    from core.dataset_adapter import resolve_dataset, SplitInfo

    lines: list[str] = []
    ok = True

    if not dataset_path.exists():
        return False, [
            f"ERRO: pasta do dataset não encontrada: {dataset_path}",
            "       Crie a pasta e popule com imagens e labels no formato YOLO.",
        ]

    lines.append(f"Dataset : {dataset_path}")

    # ── Resolve dataset via adapter (suporta Roboflow e YOLO padrão) ─────────
    try:
        info = resolve_dataset(dataset_path)
    except FileNotFoundError:
        lines.append("  ERRO  : data.yaml ausente")
        return False, lines
    except (ValueError, Exception) as exc:
        lines.append(f"  ERRO  : data.yaml inválido — {exc}")
        return False, lines

    rb_tag = " [Roboflow]" if info.roboflow_origin else ""
    lines.append(f"  YAML  : OK — {info.nc} classes: {info.names}{rb_tag}")

    # ── Splits obrigatórios ───────────────────────────────────────────────────
    for split in (info.train, info.val):
        split_ok, split_lines = _check_split_resolved(split, required=True)
        lines.extend(split_lines)
        if not split_ok:
            ok = False

    # ── Split opcional (test) ─────────────────────────────────────────────────
    if info.test and info.test.images_dir.exists():
        _, split_lines = _check_split_resolved(info.test, required=False)
        lines.extend(split_lines)

    return ok, lines


def _check_split_resolved(
    split: "SplitInfo",  # noqa: F821
    required: bool,
) -> tuple[bool, list[str]]:
    """
    Valida um split com caminhos já resolvidos pelo dataset_adapter.
    Aceita qualquer nome de pasta (val/, valid/, etc.) desde que os
    caminhos declarados no data.yaml existam e contenham imagens/labels.
    """
    lines: list[str] = []
    img_dir = split.images_dir
    lbl_dir = split.labels_dir

    if not img_dir.exists():
        msg = f"{split.name}: diretório de imagens não encontrado ({img_dir})"
        lines.append(f"  ERRO  : {msg}" if required else f"  AVISO : {msg}")
        return not required, lines

    imgs = [f for f in img_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS]
    lbls = [f for f in lbl_dir.iterdir() if f.suffix == ".txt"] if lbl_dir.exists() else []

    if required and len(imgs) == 0:
        lines.append(f"  ERRO  : {split.name} — diretório de imagens está vazio ({img_dir})")
        return False, lines

    if required and not lbl_dir.exists():
        lines.append(f"  ERRO  : {split.name} — diretório de labels não encontrado ({lbl_dir})")
        return False, lines

    missing = _count_missing_labels(imgs, lbl_dir) if lbl_dir.exists() else len(imgs)
    warn    = f" ({missing} imagens sem label)" if missing else ""
    lines.append(
        f"  OK    : {split.name} — {len(imgs)} imagens, {len(lbls)} labels{warn}"
    )
    return True, lines


def _count_missing_labels(imgs: list[Path], lbl_dir: Path) -> int:
    label_stems = {f.stem for f in lbl_dir.iterdir() if f.suffix == ".txt"}
    return sum(1 for img in imgs if img.stem not in label_stems)
