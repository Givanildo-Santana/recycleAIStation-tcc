"""
Contrato do pacote de modelo RecycleAI — schema "recycleai-pkg-v1".

Estrutura do pacote exportado:

  <nome_experimento>_package/
    manifest.json               ← contrato principal (este módulo)
    weights/
      best.pt                   ← pesos PyTorch originais (backup / reexport)
      best_ts.pt                ← TorchScript para deploy (torch.jit.load)
    config/
      data.yaml                 ← nc + names em ordem exata
      hyp.yaml                  ← hiperparâmetros do treino (informativo)
      training_config.yaml      ← snapshot de TrainingConfig (informativo)
    results/
      results.csv               ← métricas por época
      confusion_matrix.png
      F1_curve.png  PR_curve.png  P_curve.png  R_curve.png
      val_batch0_labels.jpg  val_batch0_pred.jpg
    README.txt                  ← instruções de importação para o operador

─────────────────────────────────────────────────────────────────────────────
CONTRATO DE COMPATIBILIDADE TREINO ↔ INFERÊNCIA
─────────────────────────────────────────────────────────────────────────────
Um pacote é compatível com a aplicação de inferência RecycleAI se e somente se:
  1. schema_version == "recycleai-pkg-v1"
  2. pipeline_version >= PIPELINE_MIN_VERSION ("1.0")
  3. model.deploy_format == "torchscript"  (ou model.format para pacotes v1.0)
  4. model.architecture == "yolov5s"       (ou model.framework para pacotes v1.0)
  5. classes.nc >= 1  e  classes.names não vazio
  6. arquivo weights/best_ts.pt presente no pacote

Qualquer falha nessas verificações torna o pacote INCOMPATÍVEL e a importação
deve ser recusada pela aplicação de inferência.

─────────────────────────────────────────────────────────────────────────────
VERSIONAMENTO DO MANIFESTO
─────────────────────────────────────────────────────────────────────────────
  pipeline_version "1.0"  — formato original
  pipeline_version "1.1"  — modelo enriquecido: architecture/training_format/
                            deploy_format separados; class_to_idx; resize_mode;
                            channels; input_layout; nms; bbox_format;
                            DatasetOriginInfo estruturado
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Constantes do contrato
# ─────────────────────────────────────────────────────────────────────────────

# Versão mínima aceita pela aplicação de inferência RecycleAI
PIPELINE_MIN_VERSION = "1.0"

# Versão atual gerada por esta implementação
PIPELINE_VERSION = "1.1"

# Schema imutável do pacote — incrementar apenas em breaking changes
SCHEMA_VERSION = "recycleai-pkg-v1"

# Arquitetura obrigatória para treino e inferência
REQUIRED_ARCHITECTURE = "yolov5s"

# Mantido para compatibilidade com código anterior (mesmo valor)
REQUIRED_FRAMEWORK = REQUIRED_ARCHITECTURE

# Formato obrigatório do peso de deploy
REQUIRED_DEPLOY_FORMAT = "torchscript"

# Formato do peso de treino
REQUIRED_TRAINING_FORMAT = "pytorch"

# Mantido para compatibilidade com código anterior (mesmo valor que REQUIRED_DEPLOY_FORMAT)
REQUIRED_FORMAT = REQUIRED_DEPLOY_FORMAT


# ─────────────────────────────────────────────────────────────────────────────
# Subseções do manifesto
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelInfo:
    """
    Identificação, arquitetura e localização dos pesos dentro do pacote.

    Campos de compatibilidade (pipeline 1.0 e 1.1):
      architecture / framework   — arquitetura do modelo (obrigatório "yolov5s")
      deploy_format / format     — formato do peso de deploy (obrigatório "torchscript")

    Campos adicionados em pipeline 1.1:
      architecture    — nome explícito da arquitetura (separado do framework)
      training_format — formato do peso de treino ("pytorch")
      deploy_format   — formato do peso de deploy ("torchscript")
    """
    file:             str              # "weights/best.pt" — caminho relativo ao pacote
    deploy_file:      str              # "weights/best_ts.pt"
    architecture:     str = REQUIRED_ARCHITECTURE   # "yolov5s" — obrigatório
    framework:        str = REQUIRED_FRAMEWORK      # idem — mantido para compat v1.0
    training_format:  str = REQUIRED_TRAINING_FORMAT  # "pytorch" — formato de best.pt
    deploy_format:    str = REQUIRED_DEPLOY_FORMAT  # "torchscript" — formato de best_ts.pt
    format:           str = REQUIRED_FORMAT         # idem — mantido para compat v1.0
    img_size:         int = 640


@dataclass
class ClassesInfo:
    """
    Classes detectáveis, sua ordem exata e mapeamento índice → nome.

    Campos adicionados em pipeline 1.1:
      class_to_idx    — mapeamento nome → índice (derivado de names, mas declarado
                        explicitamente para evitar ambiguidade na inferência)
      canonical_names — nomes canônicos/padronizados opcionais; None quando as classes
                        do dataset já são os nomes definitivos. Não é tradução automática.
    """
    nc:              int            # número de classes (≥ 1)
    names:           list           # nomes em ordem exata: índice 0 = classe 0
    class_to_idx:    dict = field(default_factory=dict)   # {"metal": 0, "papel": 1, ...}
    canonical_names: Optional[list] = None                # nomes canônicos se diferirem


@dataclass
class PreprocessingInfo:
    """
    Pré-processamento aplicado antes da inferência.
    Deve ser idêntico ao pipeline de treino do YOLOv5s via Ultralytics.

    Campos adicionados em pipeline 1.1:
      channels      — número de canais da imagem de entrada (3 = RGB)
      input_layout  — layout do tensor de entrada ("BCHW" = batch × canais × H × W)
      resize_mode   — modo de redimensionamento antes da inferência ("letterbox")

    Normalização YOLOv5 padrão:
      - Entrada: pixel [0, 255] uint8
      - Divisão por 255 → float [0.0, 1.0]
      - Sem subtração de média ou divisão por desvio padrão adicionais
      - mean = [0.0, 0.0, 0.0], std = [1.0, 1.0, 1.0] refletem isso
    """
    img_size:     int  = 640
    channels:     int  = 3            # RGB
    input_layout: str  = "BCHW"       # PyTorch padrão
    resize_mode:  str  = "letterbox"  # YOLOv5/Ultralytics: padding proporcional
    normalize:    bool = True
    mean:         list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    std:          list = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class PostprocessingInfo:
    """
    Parâmetros de pós-processamento aplicados após a saída bruta do modelo.

    Campos adicionados em pipeline 1.1:
      nms         — indica que NMS é aplicado (sempre True em YOLOv5)
      bbox_format — formato das caixas delimitadoras após NMS ("xyxy" = x1,y1,x2,y2)
    """
    nms:            bool  = True     # Non-Maximum Suppression sempre aplicado
    conf_threshold: float = 0.25
    iou_threshold:  float = 0.45
    max_det:        int   = 1000
    bbox_format:    str   = "xyxy"   # formato de saída: x_min, y_min, x_max, y_max


@dataclass
class TrainingMetrics:
    """Métricas de treinamento registradas ao final do treino."""
    epochs_trained: int
    best_epoch:     Optional[int]    # None quando não reportado pelo Ultralytics
    map50:          Optional[float]  # mAP @ IoU=0.50
    map50_95:       Optional[float]  # mAP @ IoU=0.50:0.95
    precision:      Optional[float]
    recall:         Optional[float]


@dataclass
class DatasetOriginInfo:
    """
    Origem do dataset utilizado no treino/refinamento.
    Substitui o campo dataset_origin (string) pelo pipeline 1.1.

    source: "local" | "roboflow" | "unknown"
    path:   caminho relativo à raiz do projeto, ou caminho original se não resolvível
    """
    name:   str           # nome do diretório do dataset (ex: "dataset_ativo")
    source: str           # origem declarada: "local", "roboflow", "unknown"
    path:   str           # caminho relativo ao projeto (ex: "datasets/dataset_ativo")


@dataclass
class RefinementInfo:
    """
    Presente apenas em pacotes gerados por refinamento (fine-tuning).
    Rastreia a origem do modelo base e o dataset de refinamento.
    """
    parent_package_name: str   # nome do experimento base
    parent_created_at:   str   # timestamp ISO do pacote base
    parent_dataset:      str   # dataset usado no treino original
    refinement_dataset:  str   # dataset usado no refinamento


@dataclass
class CompatibilityInfo:
    """
    Política de compatibilidade e prioridade de hardware para inferência.

    deploy_priority:
      "gpu_first" — usa GPU se disponível, fallback para CPU (padrão)
      "cpu_only"  — força CPU independentemente de GPU disponível
    """
    cpu_inference:     bool  = True
    gpu_inference:     bool  = True
    deploy_priority:   str   = "gpu_first"   # "gpu_first" | "cpu_only"
    min_vram_gb:       float = 0.0           # 0 = CPU-only suficiente para inferência
    torch_min_version: str   = "2.0.0"
    torchscript:       bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# Manifesto principal
# ─────────────────────────────────────────────────────────────────────────────

# Mapeamento campo → tipo para desserialização de nested dataclasses em load()
_NESTED_FIELD_TYPES: dict[str, type] = {
    "model":         ModelInfo,
    "classes":       ClassesInfo,
    "preprocessing": PreprocessingInfo,
    "postprocessing": PostprocessingInfo,
    "training":      TrainingMetrics,
    "compatibility": CompatibilityInfo,
    "refinement":    RefinementInfo,
    "dataset":       DatasetOriginInfo,
}


@dataclass
class ModelPackageManifest:
    """
    Manifesto completo do pacote de modelo RecycleAI.

    Campos obrigatórios para compatibilidade com a inferência:
      schema_version, pipeline_version, name, model, classes,
      preprocessing, postprocessing, compatibility

    Versionamento:
      pipeline_version "1.0" — formato original
      pipeline_version "1.1" — manifesto enriquecido (campos adicionais)
    """
    # ── Identificação ─────────────────────────────────────────────────────────
    name:             str
    schema_version:   str = SCHEMA_VERSION
    pipeline_version: str = PIPELINE_VERSION
    version:          str = "1.0.0"
    created_at:       str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    # ── Subseções (preenchidas pelo exporter) ─────────────────────────────────
    model:          Optional[ModelInfo]          = None
    classes:        Optional[ClassesInfo]        = None
    preprocessing:  Optional[PreprocessingInfo]  = None
    postprocessing: Optional[PostprocessingInfo] = None
    training:       Optional[TrainingMetrics]    = None
    compatibility:  Optional[CompatibilityInfo]  = None

    # ── Origem do dataset (pipeline 1.1) ──────────────────────────────────────
    dataset:        Optional[DatasetOriginInfo]  = None

    # ── Ambiente de treino (informativo) ──────────────────────────────────────
    trained_on_os:  str = ""
    trained_on_gpu: str = ""
    dataset_origin: str = ""   # mantido por compatibilidade com pipeline 1.0
    observations:   str = ""   # campo livre para anotações do operador

    # ── Modo de treinamento ───────────────────────────────────────────────────
    training_mode: str = "training"               # "training" | "refinement"
    refinement: Optional[RefinementInfo] = None   # presente apenas em refinamentos

    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serializa para dict JSON-serializável usando asdict() recursivo."""
        return asdict(self)

    def save(self, path: Path):
        """Salva manifest.json no caminho informado."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "ModelPackageManifest":
        """
        Carrega manifest.json existente.

        Campos desconhecidos são ignorados (forward-compat).
        Nested dicts são desserializados para os dataclasses corretos (fix de
        bug latente: sem isso, manifest.classes seria dict, não ClassesInfo).
        Campos ausentes usam os defaults do dataclass (backward-compat com v1.0).
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}

        # Desserializa nested dataclasses a partir de dicts JSON
        for field_name, nested_cls in _NESTED_FIELD_TYPES.items():
            raw = known.get(field_name)
            if isinstance(raw, dict):
                valid = nested_cls.__dataclass_fields__
                known[field_name] = nested_cls(
                    **{k: v for k, v in raw.items() if k in valid}
                )

        return cls(**known)


# ─────────────────────────────────────────────────────────────────────────────
# Validação de compatibilidade com a aplicação de inferência
# ─────────────────────────────────────────────────────────────────────────────

def check_inference_compatibility(
    manifest: ModelPackageManifest,
    pkg_dir:  Optional[Path] = None,
) -> tuple[bool, list[str]]:
    """
    Verifica se o pacote é compatível com a aplicação de inferência RecycleAI.

    Aceita tanto pipeline 1.0 (campos framework/format) quanto 1.1
    (campos architecture/deploy_format), verificando ambos quando presentes.

    Args:
        manifest: manifesto carregado do pacote.
        pkg_dir:  raiz do pacote (opcional). Se fornecido, verifica existência física
                  de best_ts.pt.

    Returns:
        (ok, issues) — ok=False se qualquer condição obrigatória falhar.
    """
    issues: list[str] = []
    ok = True

    def _fail(msg: str):
        nonlocal ok
        issues.append(msg)
        ok = False

    # 1. Schema
    if manifest.schema_version != SCHEMA_VERSION:
        _fail(f"schema_version incompatível: '{manifest.schema_version}' "
              f"(esperado '{SCHEMA_VERSION}')")

    # 2. Pipeline version
    try:
        pv     = tuple(int(x) for x in manifest.pipeline_version.split(".")[:2])
        pv_min = tuple(int(x) for x in PIPELINE_MIN_VERSION.split(".")[:2])
        if pv < pv_min:
            _fail(f"pipeline_version '{manifest.pipeline_version}' abaixo do mínimo "
                  f"'{PIPELINE_MIN_VERSION}'")
    except (ValueError, AttributeError):
        _fail(f"pipeline_version inválido: '{manifest.pipeline_version}'")

    # 3 & 4. Arquitetura e formato de deploy
    if manifest.model is None:
        _fail("model não definido no manifesto")
    else:
        # Aceita campo novo (deploy_format/architecture) ou antigo (format/framework)
        deploy_fmt = getattr(manifest.model, "deploy_format", None) or manifest.model.format
        arch       = getattr(manifest.model, "architecture", None) or manifest.model.framework

        if deploy_fmt != REQUIRED_DEPLOY_FORMAT:
            _fail(f"formato de deploy incompatível: '{deploy_fmt}' "
                  f"(esperado '{REQUIRED_DEPLOY_FORMAT}')")
        if arch != REQUIRED_ARCHITECTURE:
            _fail(f"arquitetura incompatível: '{arch}' "
                  f"(esperado '{REQUIRED_ARCHITECTURE}')")

    # 5. Classes definidas
    if manifest.classes is None:
        _fail("classes não definidas no manifesto")
    else:
        nc    = manifest.classes.nc    if hasattr(manifest.classes, "nc")    else manifest.classes.get("nc", 0)
        names = manifest.classes.names if hasattr(manifest.classes, "names") else manifest.classes.get("names", [])

        if nc < 1:
            _fail(f"nc={nc}: nenhuma classe definida")
        if not names:
            _fail("classes.names vazio")
        elif len(names) != nc:
            _fail(f"classes.names tem {len(names)} entradas mas nc={nc}")

    # 6. Existência física de best_ts.pt (se pkg_dir fornecido)
    if pkg_dir is not None:
        if manifest.model is not None:
            deploy_rel = manifest.model.deploy_file
        else:
            deploy_rel = "weights/best_ts.pt"
        deploy_abs = pkg_dir / deploy_rel
        if not deploy_abs.exists():
            _fail(f"arquivo de deploy não encontrado: {deploy_rel}")

    return ok, issues
