"""
Exportação do pacote completo do modelo.

Gera um diretório estruturado contendo:
  <nome>_package/
    manifest.json
    weights/
      best.pt          ← pesos originais PyTorch (treino / reexport / refinamento)
      best_ts.pt       ← TorchScript (obrigatório para inferência)
    config/
      data.yaml
      hyp.yaml
      training_config.yaml
    results/
      results.csv e imagens de métricas (se geradas pelo treino)
    README.txt

O diretório é salvo em exports/<nome_experimento>_package/.
"""
from __future__ import annotations

import shutil
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.config import TrainingConfig
from core.trainer import TrainingResult
from core.environment import EnvironmentReport
from core.model_package import (
    ModelPackageManifest,
    ModelInfo,
    ClassesInfo,
    DatasetOriginInfo,
    PreprocessingInfo,
    PostprocessingInfo,
    TrainingMetrics,
    CompatibilityInfo,
    RefinementInfo,
    SCHEMA_VERSION,
    PIPELINE_VERSION,
    REQUIRED_ARCHITECTURE,
    REQUIRED_FRAMEWORK,
    REQUIRED_FORMAT,
    REQUIRED_DEPLOY_FORMAT,
    REQUIRED_TRAINING_FORMAT,
)


_RESULTS_FILES = [
    "results.csv",
    "confusion_matrix.png",
    "F1_curve.png",
    "PR_curve.png",
    "P_curve.png",
    "R_curve.png",
    "val_batch0_labels.jpg",
    "val_batch0_pred.jpg",
]

# Raiz do projeto (dois níveis acima deste arquivo: core/ → aplicacao-treinamento/)
_ROOT = Path(__file__).resolve().parent.parent


def export_model_package(
    cfg: TrainingConfig,
    result: TrainingResult,
    env: EnvironmentReport,
) -> Path:
    """
    Monta e salva o pacote completo do modelo.

    Retorna: caminho do diretório do pacote.
    Lança: RuntimeError se best.pt não existir.
    """
    if not result.best_weights or not result.best_weights.exists():
        raise RuntimeError(
            f"Pesos não encontrados: {result.best_weights}. "
            "Verifique se o treino foi concluído com sucesso."
        )

    pkg_name = f"{cfg.name}_package"
    pkg_dir  = cfg.exports_dir / pkg_name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # ── weights/ ─────────────────────────────────────────────────────────────
    weights_dir = pkg_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    best_pt = weights_dir / "best.pt"
    shutil.copy2(result.best_weights, best_pt)

    best_ts = weights_dir / "best_ts.pt"
    _export_torchscript(result.best_weights, best_ts, cfg.img_size)

    # ── config/ ──────────────────────────────────────────────────────────────
    config_dir = pkg_dir / "config"
    config_dir.mkdir(exist_ok=True)

    _copy_if_exists(cfg.dataset_path / "data.yaml", config_dir / "data.yaml")
    if result.results_dir:
        _copy_if_exists(result.results_dir / "args.yaml", config_dir / "hyp.yaml")
    _save_training_config(cfg, config_dir / "training_config.yaml")

    # ── results/ ─────────────────────────────────────────────────────────────
    results_dir = pkg_dir / "results"
    results_dir.mkdir(exist_ok=True)
    if result.results_dir:
        for fname in _RESULTS_FILES:
            _copy_if_exists(result.results_dir / fname, results_dir / fname)

    # ── manifest.json ─────────────────────────────────────────────────────────
    # Verificar arquitetura real do artefato antes de escrever o manifesto.
    # O manifesto só declara 'architecture=yolov5s' após confirmação do binário.
    verified_arch = _verify_artifact_arch(best_pt, cfg.model_arch)
    print(f"  Arquitetura verificada: {verified_arch} ✓")

    classes_info   = _load_classes(cfg.dataset_path / "data.yaml")
    dataset_origin = _make_dataset_origin(cfg.dataset_path)

    manifest = ModelPackageManifest(
        name             = cfg.name,
        schema_version   = SCHEMA_VERSION,
        pipeline_version = PIPELINE_VERSION,
        created_at       = datetime.now().isoformat(timespec="seconds"),

        model = ModelInfo(
            file            = "weights/best.pt",
            deploy_file     = "weights/best_ts.pt",
            architecture    = verified_arch,
            framework       = verified_arch,
            training_format = REQUIRED_TRAINING_FORMAT,
            deploy_format   = REQUIRED_DEPLOY_FORMAT,
            format          = REQUIRED_FORMAT,
            img_size        = cfg.img_size,
        ),

        classes = classes_info,

        preprocessing = PreprocessingInfo(
            img_size     = cfg.img_size,
            channels     = 3,
            input_layout = "BCHW",
            resize_mode  = "letterbox",
            normalize    = True,
            mean         = [0.0, 0.0, 0.0],
            std          = [1.0, 1.0, 1.0],
        ),

        postprocessing = PostprocessingInfo(
            nms            = True,
            conf_threshold = 0.25,
            iou_threshold  = 0.45,
            max_det        = 1000,
            bbox_format    = "xyxy",
        ),

        training = TrainingMetrics(
            epochs_trained = result.epochs_trained,
            best_epoch     = result.best_epoch,
            map50          = result.map50,
            map50_95       = result.map50_95,
            precision      = result.precision,
            recall         = result.recall,
        ),

        compatibility = CompatibilityInfo(
            cpu_inference   = True,
            gpu_inference   = True,
            deploy_priority = "gpu_first",
            min_vram_gb     = 0.0,
        ),

        dataset         = dataset_origin,
        trained_on_os   = f"{env.os_name} {env.os_version}",
        trained_on_gpu  = env.primary_gpu.name if env.primary_gpu else "N/A",
        dataset_origin  = dataset_origin.path,   # campo legado mantido por compat
        observations    = "",
        training_mode   = cfg.training_mode,
        refinement      = RefinementInfo(
            parent_package_name = cfg.parent_package_name,
            parent_created_at   = cfg.parent_package_created_at,
            parent_dataset      = cfg.parent_package_dataset,
            refinement_dataset  = dataset_origin.path,
        ) if cfg.training_mode == "refinement" else None,
    )
    manifest.save(pkg_dir / "manifest.json")

    # ── README.txt ────────────────────────────────────────────────────────────
    _write_readme(pkg_dir, cfg, classes_info)

    return pkg_dir


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _verify_artifact_arch(best_pt: Path, expected_arch: str) -> str:
    """
    Verifica que best.pt contém a arquitetura esperada antes de gerar o manifesto.

    Bloqueia exportação (RuntimeError) se:
      - A arquitetura não puder ser determinada
      - A arquitetura real for diferente de expected_arch

    Retorna o nome da arquitetura confirmada para uso no manifesto.
    O manifesto só escreve 'architecture=yolov5s' após esta verificação.
    """
    try:
        from ultralytics import YOLO
        m = YOLO(str(best_pt))

        # Mesma estratégia de detecção do trainer: yaml_file embutido
        actual: Optional[str] = None
        inner = getattr(m, 'model', None)
        if inner is not None:
            yaml_data = getattr(inner, 'yaml', None)
            if isinstance(yaml_data, dict):
                yaml_file = yaml_data.get('yaml_file', '') or ''
                if yaml_file:
                    actual = Path(yaml_file).stem

        if actual is None:
            raise RuntimeError(
                f"Exportação bloqueada: arquitetura do artefato não pôde ser verificada.\n"
                f"  Esperado : {expected_arch}\n"
                f"  Ação     : Re-execute o treino com peso base controlado em weights/base/."
            )
        if actual != expected_arch:
            raise RuntimeError(
                f"Exportação bloqueada: arquitetura do artefato diverge do contrato.\n"
                f"  Esperado  : {expected_arch}\n"
                f"  Encontrado: {actual}\n"
                f"  Ação      : Re-execute o treino após resolver a questão do peso base "
                f"(coloque '{expected_arch}.pt' verificado em weights/base/)."
            )
        return actual

    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Exportação bloqueada: falha ao verificar arquitetura do artefato ({exc})."
        )


def _export_torchscript(src: Path, dest: Path, img_size: int):
    """Exporta best.pt para TorchScript usando Ultralytics."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(src))
        model.export(format="torchscript", imgsz=img_size)
        # Ultralytics salva <nome_sem_ext>.torchscript ou <nome_sem_ext>_torchscript.pt
        candidates = [
            src.parent / f"{src.stem}.torchscript",
            src.parent / f"{src.stem}_torchscript.pt",
            src.with_suffix(".torchscript"),
        ]
        for c in candidates:
            if c.exists():
                shutil.move(str(c), str(dest))
                return
        # Fallback: copia best.pt como best_ts.pt com aviso
        shutil.copy2(src, dest)
        print("  AVISO: export TorchScript não encontrou arquivo gerado; "
              "copiou best.pt como fallback.")
    except Exception as exc:
        shutil.copy2(src, dest)
        print(f"  AVISO: falha ao exportar TorchScript ({exc}); "
              "copiou best.pt como fallback.")


def _copy_if_exists(src: Path, dest: Path):
    if src.exists():
        shutil.copy2(src, dest)


def _save_training_config(cfg: TrainingConfig, dest: Path):
    with open(dest, "w", encoding="utf-8") as f:
        yaml.dump(cfg.to_dict(), f, allow_unicode=True, sort_keys=False)


def _load_classes(data_yaml: Path) -> ClassesInfo:
    """
    Lê classes do data.yaml e constrói ClassesInfo com class_to_idx explícito.
    Usa dataset_adapter para suportar datasets Roboflow (valid: key, path: key).
    """
    try:
        # Preferência: resolver via adapter (correto para Roboflow)
        try:
            from core.dataset_adapter import resolve_dataset
            info = resolve_dataset(data_yaml.parent)
            nc    = info.nc
            names = info.names
        except Exception:
            # Fallback: leitura direta (dataset YOLO padrão)
            with open(data_yaml, encoding="utf-8") as f:
                meta = yaml.safe_load(f)
            nc    = int(meta.get("nc", 0))
            names = list(meta.get("names", []))

        return ClassesInfo(
            nc           = nc,
            names        = names,
            class_to_idx = {name: idx for idx, name in enumerate(names)},
            canonical_names = None,   # não há tradução automática
        )
    except Exception:
        return ClassesInfo(nc=0, names=[], class_to_idx={})


def _make_dataset_origin(dataset_path: Path) -> DatasetOriginInfo:
    """
    Cria DatasetOriginInfo com nome, fonte e caminho relativo ao projeto.

    source: "roboflow" se detectado artefatos/convenção Roboflow, "local" caso contrário.
    path:   caminho relativo à raiz do projeto quando possível; caminho original senão.
    """
    # Detecta origem via adapter
    source = "local"
    try:
        from core.dataset_adapter import resolve_dataset
        info = resolve_dataset(dataset_path)
        if info.roboflow_origin:
            source = "roboflow"
    except Exception:
        pass

    # Calcula caminho relativo ao projeto
    try:
        rel = dataset_path.resolve().relative_to(_ROOT)
        path_str = str(rel).replace("\\", "/")
    except ValueError:
        # Dataset fora da árvore do projeto (ex: Google Colab /content/...)
        path_str = str(dataset_path)

    return DatasetOriginInfo(
        name   = dataset_path.name,
        source = source,
        path   = path_str,
    )


def _write_readme(pkg_dir: Path, cfg: TrainingConfig, classes: ClassesInfo):
    lines = [
        "RecycleAI — Pacote de Modelo Treinado",
        "=" * 42,
        "",
        f"Experimento : {cfg.name}",
        f"Arquitetura : {cfg.model_arch} (TorchScript para deploy)",
        f"Classes     : {classes.nc} — {classes.names}",
        "",
        "Arquivos principais:",
        "  weights/best_ts.pt     — TorchScript para produção (use este na inferência)",
        "  weights/best.pt        — pesos PyTorch originais (backup / refinamento)",
        "  config/data.yaml       — definição de classes",
        "  manifest.json          — metadados completos (pipeline v1.1)",
        "",
        "Uso na aplicação de inferência RecycleAI:",
        "  1. Abra a tela de Administração e clique em 'Importar Modelo Externo'.",
        "  2. Selecione a pasta deste pacote (<nome>_package/).",
        "  3. O manifest.json valida a compatibilidade automaticamente.",
        "",
        "Compatibilidade:",
        "  - Inferência CPU: suportada (sem GPU)",
        "  - Inferência GPU: suportada (CUDA)",
        "  - PyTorch mínimo: 2.0.0",
        "  - Formato: TorchScript (torch.jit.load)",
        "  - Pré-processamento: letterbox + normalização [0,1]",
        "  - Pós-processamento: NMS, bbox formato xyxy",
    ]
    (pkg_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")
