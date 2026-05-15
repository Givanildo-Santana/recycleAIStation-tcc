"""
Wrapper de treinamento YOLOv5s via biblioteca Ultralytics.

Requer: pip install ultralytics>=8.0.0 (inclui suporte a YOLOv5s)

Fluxo interno:
  1. Cria diretório de saída em runs/<nome_experimento>/
  2. Resolve peso base: weights/base/yolov5s.pt (controlado) ou download Ultralytics
  3. Verifica arquitetura real do modelo carregado — bloqueia se não for yolov5s estrito
  4. Invoca YOLO.train() com os parâmetros da config
  5. Retorna TrainingResult com pesos e métricas finais

Garantia de YOLOv5s estrito:
  - _resolve_base_weight_path() prefere peso local controlado (weights/base/yolov5s.pt)
  - _verify_yolov5s_strict() inspeciona yaml_file embutido no modelo carregado
  - Qualquer divergência (yolov5su, indeterminável) levanta ArchitectureViolationError
    e aborta o treino antes de qualquer época ser executada

Nota: a exportação para TorchScript é feita pelo exporter.py,
não aqui. O trainer apenas executa e retorna o caminho dos pesos.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.config import TrainingConfig

# Raiz da aplicação de treinamento (um nível acima de core/)
_ROOT = Path(__file__).resolve().parent.parent

# Diretório de pesos base controlados — yolov5s.pt verificado fica aqui
_WEIGHTS_BASE_DIR = _ROOT / "weights" / "base"


# ─────────────────────────────────────────────────────────────────────────────
# Exceção de violação de contrato de arquitetura
# ─────────────────────────────────────────────────────────────────────────────

class ArchitectureViolationError(RuntimeError):
    """
    Levantada quando o modelo carregado não corresponde à arquitetura declarada.

    Indica que o Ultralytics pode ter redirecionado ou substituído o peso base
    sem aviso explícito (ex: yolov5s.pt → yolov5su.pt).
    """


# ─────────────────────────────────────────────────────────────────────────────
# Controle de peso base e verificação de arquitetura
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_base_weight_path(model_arch: str) -> str:
    """
    Retorna o caminho a usar para inicializar YOLO() no novo treinamento.

    Preferência:
      1. weights/base/<model_arch>.pt — peso local controlado e já verificado
      2. "<model_arch>.pt"            — fallback: Ultralytics faz download na rede

    O peso local controlado evita que o Ultralytics substitua silenciosamente
    yolov5s.pt por yolov5su.pt em chamadas subsequentes ao treino.
    """
    _WEIGHTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    local = _WEIGHTS_BASE_DIR / f"{model_arch}.pt"
    if local.exists():
        print(f"  [ARCH] Usando peso base controlado: weights/base/{model_arch}.pt")
        return str(local)
    print(f"  [ARCH] Peso base local não encontrado — Ultralytics fará download de '{model_arch}.pt'.")
    return f"{model_arch}.pt"


def _detect_actual_arch(model) -> Optional[str]:
    """
    Inspeciona a arquitetura real do modelo YOLO carregado pelo Ultralytics.

    Estratégias (em ordem de confiabilidade):
      1. yaml_file no dicionário YAML embutido em model.model.yaml
         — gerado pelo Ultralytics ao montar o modelo; contém o nome do .yaml de origem
         — ex: '/path/to/yolov5s.yaml' → stem = 'yolov5s'
      2. model_name do objeto YOLO
         — atributo de instância com o nome do modelo solicitado

    Retorna o nome da arquitetura ('yolov5s', 'yolov5su', etc.) ou None se
    nenhuma estratégia produzir resultado confiável.
    """
    # Estratégia 1: yaml_file embutido
    try:
        inner = getattr(model, 'model', None)
        if inner is not None:
            yaml_data = getattr(inner, 'yaml', None)
            if isinstance(yaml_data, dict):
                yaml_file = yaml_data.get('yaml_file', '') or ''
                if yaml_file:
                    return Path(yaml_file).stem
    except Exception:
        pass

    # Estratégia 2: model_name do objeto YOLO
    try:
        model_name = getattr(model, 'model_name', '') or ''
        if model_name:
            return Path(model_name).stem
    except Exception:
        pass

    return None


def _verify_yolov5s_strict(model, expected_arch: str = "yolov5s") -> str:
    """
    Verifica que o modelo YOLO carregado é estritamente a arquitetura esperada.

    Bloqueia com ArchitectureViolationError se:
      - A arquitetura não puder ser determinada (indeterminável → bloqueio por segurança)
      - A arquitetura real for diferente de expected_arch (ex: yolov5su)

    Retorna o nome da arquitetura confirmada se a verificação passar.
    """
    actual = _detect_actual_arch(model)

    if actual is None:
        raise ArchitectureViolationError(
            "Arquitetura do modelo não pôde ser verificada — treino bloqueado.\n"
            f"  Esperado  : {expected_arch}\n"
            f"  Encontrado: indeterminável\n"
            f"  Causa     : O Ultralytics pode ter redirecionado '{expected_arch}.pt' "
            f"para uma variante não identificável.\n"
            f"  Ação      : Coloque '{expected_arch}.pt' verificado em "
            f"weights/base/ para forçar uso do peso correto, "
            f"ou verifique a versão do Ultralytics instalada (pip show ultralytics)."
        )

    if actual != expected_arch:
        raise ArchitectureViolationError(
            f"Arquitetura incorreta detectada — treino bloqueado.\n"
            f"  Esperado  : {expected_arch}\n"
            f"  Encontrado: {actual}\n"
            f"  Causa     : O Ultralytics substituiu '{expected_arch}.pt' por "
            f"'{actual}.pt' automaticamente.\n"
            f"  Ação      : Coloque o arquivo '{expected_arch}.pt' correto em "
            f"weights/base/ para forçar uso do peso base controlado."
        )

    return actual


def _cache_base_weight(model, model_arch: str) -> None:
    """
    Tenta salvar o peso base verificado em weights/base/ para uso futuro.

    Ao salvar localmente, execuções subsequentes usam o peso controlado diretamente,
    eliminando dependência de download e risco de redirecionamento pelo Ultralytics.
    Falha silenciosa: cache é otimização, não requisito.
    """
    cached = _WEIGHTS_BASE_DIR / f"{model_arch}.pt"
    if cached.exists():
        return
    try:
        import torch
        ckpt = getattr(model, 'ckpt', None)
        if ckpt is not None:
            _WEIGHTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, cached)
            print(
                f"  [ARCH] Peso base '{model_arch}.pt' verificado e salvo em "
                f"weights/base/ — próximos treinos usarão este arquivo."
            )
    except Exception as exc:
        print(f"  [ARCH] Aviso: não foi possível salvar peso base em cache ({exc}).")


# ─────────────────────────────────────────────────────────────────────────────
# Resultado do treinamento
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    success:        bool
    best_weights:   Optional[Path]   = None   # caminho de best.pt
    last_weights:   Optional[Path]   = None   # caminho de last.pt
    results_dir:    Optional[Path]   = None   # diretório completo do run
    epochs_trained: int              = 0
    best_epoch:     Optional[int]    = None   # None quando não reportado pelo Ultralytics
    map50:          Optional[float]  = None
    map50_95:       Optional[float]  = None
    precision:      Optional[float]  = None
    recall:         Optional[float]  = None
    error:          Optional[str]    = None


# ─────────────────────────────────────────────────────────────────────────────
# Treinamento
# ─────────────────────────────────────────────────────────────────────────────

def run_training(cfg: TrainingConfig) -> TrainingResult:
    """
    Executa o treinamento YOLOv5s com os parâmetros da configuração.

    Requer 'ultralytics' instalado e GPU NVIDIA disponível.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return TrainingResult(
            success=False,
            error=(
                "Biblioteca 'ultralytics' não instalada. "
                "Execute: pip install ultralytics"
            ),
        )

    cfg.runs_dir.mkdir(parents=True, exist_ok=True)

    # ── Verificação básica do data.yaml ───────────────────────────────────────
    raw_yaml = cfg.dataset_path / "data.yaml"
    if not raw_yaml.exists():
        return TrainingResult(
            success=False,
            error=f"data.yaml não encontrado em: {cfg.dataset_path}",
        )

    # ── Normalização para compatibilidade com Roboflow ────────────────────────
    # Cria data_normalized.yaml se o dataset usa 'valid' em vez de 'val',
    # ou se precisar de caminhos absolutos. Sem isso, Ultralytics pode rejeitar
    # o dataset por não encontrar o split 'val'.
    try:
        from core.dataset_adapter import resolve_dataset, get_normalized_data_yaml
        _ds_info  = resolve_dataset(cfg.dataset_path)
        _norm_dir = cfg.runs_dir / cfg.name / "_config"
        data_yaml = get_normalized_data_yaml(_ds_info, _norm_dir)
    except Exception:
        # Fallback: usa data.yaml original (dataset já está no formato correto)
        data_yaml = raw_yaml

    try:
        # ── Carregamento do modelo ────────────────────────────────────────────
        # Refinamento: carrega best.pt do pacote base (arquitetura já verificada no pacote)
        # Treinamento novo: prefere peso local controlado; fallback para download Ultralytics
        if cfg.pretrained_weights and cfg.pretrained_weights.exists():
            model = YOLO(str(cfg.pretrained_weights))
        else:
            weight_path = _resolve_base_weight_path(cfg.model_arch)
            model = YOLO(weight_path)

        # ── Verificação de arquitetura estrita ────────────────────────────────
        # Bloqueia antes de qualquer época se a arquitetura real divergir do contrato.
        # Aplica-se tanto ao novo treinamento quanto ao refinamento.
        _verify_yolov5s_strict(model, expected_arch=cfg.model_arch)
        print(f"  [ARCH] Arquitetura verificada: {cfg.model_arch} ✓")

        # ── Cache do peso base verificado ─────────────────────────────────────
        # Apenas no treinamento novo (não no refinamento, que parte de best.pt treinado).
        if not (cfg.pretrained_weights and cfg.pretrained_weights.exists()):
            _cache_base_weight(model, cfg.model_arch)

        results = model.train(
            data      = str(data_yaml),
            epochs    = cfg.epochs,
            batch     = cfg.batch_size,
            imgsz     = cfg.img_size,
            lr0       = cfg.lr0,
            device    = cfg.device,
            workers   = cfg.workers,
            patience  = cfg.patience,
            project   = str(cfg.runs_dir),
            name      = cfg.name,
            exist_ok  = True,
            augment   = cfg.augment,
            cache     = cfg.cache,
            verbose   = True,
        )

        run_dir = cfg.runs_dir / cfg.name
        best    = run_dir / "weights" / "best.pt"
        last    = run_dir / "weights" / "last.pt"

        # Extrai métricas do resultado Ultralytics
        map50    = _safe_metric(results, "metrics/mAP50(B)")
        map50_95 = _safe_metric(results, "metrics/mAP50-95(B)")
        precision = _safe_metric(results, "metrics/precision(B)")
        recall    = _safe_metric(results, "metrics/recall(B)")

        # best_epoch: extraído honestamente — None quando Ultralytics não reporta.
        # getattr(..., 0) or 0 seria errado: tornaria None em 0 (época válida).
        _raw_best = getattr(results, "best_epoch", None)
        try:
            best_epoch: Optional[int] = int(_raw_best) if _raw_best is not None else None
        except (TypeError, ValueError):
            best_epoch = None

        return TrainingResult(
            success        = best.exists(),
            best_weights   = best if best.exists() else None,
            last_weights   = last if last.exists() else None,
            results_dir    = run_dir,
            epochs_trained = cfg.epochs,
            best_epoch     = best_epoch,
            map50          = map50,
            map50_95       = map50_95,
            precision      = precision,
            recall         = recall,
        )

    except ArchitectureViolationError as exc:
        return TrainingResult(success=False, error=str(exc))
    except Exception as exc:
        return TrainingResult(success=False, error=str(exc))


def _safe_metric(results, key: str) -> Optional[float]:
    """Extrai métrica do objeto de resultados Ultralytics de forma segura."""
    try:
        val = results.results_dict.get(key)
        return float(val) if val is not None else None
    except Exception:
        return None
