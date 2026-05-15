"""
Renderização de detecções no frame de câmera.

Movido de core/detection/legacy_adapter.py (ETAPA 2).
Dependência única: cv2 (já no venv).

─── API pública ─────────────────────────────────────────────────────────────
  annotate_frame(frame, result, roi) — anota frame in-place
  ROI_COLOR / IN_ROI_COLOR / OUT_ROI_COLOR — constantes de cor exportadas

─── DÍVIDA TÉCNICA RESIDUAL ─────────────────────────────────────────────────
  · Cores e espessuras estão hardcoded — podem ser parametrizadas futuro.
  · Para empacotamento, considerar renderização opcional (modo headless).
"""
from __future__ import annotations

import cv2

# Cores BGR
ROI_COLOR     = (0, 255, 255)   # amarelo-ciano  — retângulo de ROI
IN_ROI_COLOR  = (0, 255, 0)     # verde          — detecções dentro do ROI
OUT_ROI_COLOR = (120, 120, 120) # cinza          — detecções fora do ROI


def annotate_frame(
    frame,
    result,
    roi: tuple[int, int, int, int],
) -> None:
    """
    Anota frame in-place com retângulo de ROI e caixas de detecção.

    Parâmetros:
      frame  — numpy array BGR, modificado in-place
      result — FrameResult (ou qualquer objeto com .boxes: list[DetectionBox])
      roi    — (x1, y1, x2, y2) em pixels

    Cores:
      ROI rect       → amarelo-ciano
      Caixas in_roi  → verde
      Caixas out_roi → cinza
    """
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), ROI_COLOR, 2)

    for box in result.boxes:
        color = IN_ROI_COLOR if box.in_roi else OUT_ROI_COLOR
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        cv2.putText(
            frame,
            f"{box.label} {box.confidence:.2f}",
            (box.x1, box.y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
