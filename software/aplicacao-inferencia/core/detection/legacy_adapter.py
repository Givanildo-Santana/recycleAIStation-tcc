"""
Adaptador de interface para a GUI — camada de composição.

ETAPA 2: Este módulo é agora residual e simples.
  · Sem sys.path manipulation
  · Sem importações de scripts externos de yolov5
  · Sem lógica de inferência ou renderização — delegadas a módulos próprios

─── O que este módulo ainda faz ─────────────────────────────────────────────
  · Define DetectionBox e FrameResult (contratos de dados consumidos pela GUI)
  · run_frame(): compõe inference.run() + ROI filter por centróide
  · annotate_frame(): re-exporta de renderer.annotate_frame (sem duplicar)

─── O que foi separado nesta etapa ──────────────────────────────────────────
  · Inferência       → core/detection/inference.py
  · Pré/pós-proc.    → core/detection/yolo_utils.py  (nativo, sem yolov5)
  · Renderização     → core/detection/renderer.py

─── DÍVIDA TÉCNICA RESIDUAL (pós ETAPA 2) ───────────────────────────────────
  · Este adapter pode ser dissolvido futuramente:
    run_frame() pode ir para inference.py ou um módulo pipeline.py
    annotate_frame re-export pode ir direto para os callers
  · Scripts yolov5 externos foram removidos do projeto; dependência eliminada
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Contratos de dados (interface GUI — estáveis)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionBox:
    """Caixa de detecção individual anotada com flag de ROI."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    in_roi: bool = False

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2


@dataclass
class FrameResult:
    """
    Resultado de um frame processado exposto para a GUI.

    roi_label — label mais confiante cujo centróide está no ROI; "NONE" se vazio.
    roi_conf  — confiança do roi_label (0.0 se "NONE").
    boxes     — todas as caixas detectadas, com flag in_roi preenchida.
    """
    roi_label: str = "NONE"
    roi_conf:  float = 0.0
    boxes:     list[DetectionBox] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def run_frame(
    model,
    frame,
    *,
    device: str,
    img_size: int,
    conf_thres: float,
    iou_thres: float,
    names: list[str],
    roi: tuple[int, int, int, int],
) -> FrameResult:
    """
    Executa inferência e aplica filtro de ROI por centróide.

    Delega inferência a core.detection.inference.run() (pipeline nativo).
    Sem sys.path, sem dependência de scripts externos.
    """
    from core.detection.inference import run

    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    raw = run(
        model, frame,
        device=device,
        img_size=img_size,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        names=names,
    )

    boxes: list[DetectionBox] = [
        DetectionBox(
            x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
            label=det.label,
            confidence=det.confidence,
            in_roi=(
                roi_x1 <= (det.x1 + det.x2) // 2 <= roi_x2
                and roi_y1 <= (det.y1 + det.y2) // 2 <= roi_y2
            ),
        )
        for det in raw
    ]

    # raw já ordenado por confiança desc — primeiro in_roi é o mais confiante
    roi_boxes = [b for b in boxes if b.in_roi]
    if roi_boxes:
        best = roi_boxes[0]
        return FrameResult(roi_label=best.label, roi_conf=best.confidence, boxes=boxes)

    return FrameResult(boxes=boxes)


def annotate_frame(
    frame,
    result: FrameResult,
    roi: tuple[int, int, int, int],
) -> None:
    """Re-exporta renderer.annotate_frame — callers não precisam importar renderer."""
    from core.detection.renderer import annotate_frame as _render
    _render(frame, result, roi)
