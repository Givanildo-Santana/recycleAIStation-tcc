"""
Pipeline de inferência nativo — sem dependência de scripts externos.

Implementa o fluxo completo frame→detecções:
  1. Pré-processamento : letterbox → tensor normalizado (0..1), BCHW
  2. Inferência        : model.forward() em torch.no_grad()
  3. Pós-processamento : NMS → scale_boxes → lista tipada de RawDetection

Utilitários YOLOv5 (letterbox, NMS, scale_boxes) consumidos via
core/detection/yolo_utils.py — ponto único e controlado.

─── Diferença em relação ao legado (scripts/modules/detection.py) ───────────
  · Retorna list[RawDetection] em vez de (detected_label, raw_tuples).
    O detected_label do legado era sobrescrito pela última detecção, não
    pela mais confiante — esse bug não existe aqui.
  · Sem dependência de sys.path externo.
  · Tipo de retorno explícito e verificável.

─── Estado pós ETAPA 2 ───────────────────────────────────────────────────────
  · yolo_utils.py agora é nativo: letterbox/NMS/scale_boxes reimplementados
    sem dependência de scripts yolov5 externos. Zero sys.path nesta cadeia.
  · torchvision.ops.nms é a única dependência nova (já estava no venv).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RawDetection:
    """
    Detecção individual retornada pelo pipeline de inferência.
    Coordenadas absolutas no frame original (antes de qualquer filtro).
    """
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float


def run(
    model,
    frame: np.ndarray,
    *,
    device: str,
    img_size: int,
    conf_thres: float,
    iou_thres: float,
    names: list[str],
) -> list[RawDetection]:
    """
    Executa o pipeline completo de inferência em um frame BGR.

    Parâmetros:
      model      — modelo TorchScript (via model_loader.load_active)
      frame      — frame BGR, numpy array (H, W, C)
      device     — "cpu" ou "cuda"
      img_size   — tamanho de entrada do modelo (ex: 640)
      conf_thres — limiar mínimo de confiança para aceitar detecção
      iou_thres  — limiar de IoU para supressão de sobreposições (NMS)
      names      — lista de nomes de classes do modelo ativo

    Retorna:
      Lista de RawDetection (pode ser vazia se nenhuma detecção).
      Ordenada por confiança decrescente.
    """
    from core.detection.yolo_utils import letterbox, non_max_suppression, scale_boxes

    # ── Pré-processamento ────────────────────────────────────────────────────
    # auto=False: garante saída exatamente (img_size x img_size).
    # Modelos YOLOv8/Ultralytics têm anchor points baked para o tamanho de
    # exportação; auto=True pode reduzir o padding a zero se dh/dw forem
    # múltiplos do stride, produzindo input não-quadrado e RuntimeError no
    # forward pass. YOLOv5 tolera qualquer tamanho múltiplo de stride.
    img = letterbox(frame, img_size, stride=32, auto=False)[0]  # (H', W', C) BGR
    img = img.transpose((2, 0, 1))                              # → (C, H', W')
    img = np.ascontiguousarray(img)
    img_t = torch.from_numpy(img).to(device).float() / 255.0   # normalizado [0,1]
    img_t = img_t.unsqueeze(0)                                  # → (1, C, H', W')

    # ── Inferência ───────────────────────────────────────────────────────────
    with torch.no_grad():
        raw = model(img_t)

    # ── Normalização do formato de saída ─────────────────────────────────────
    # YOLOv5 (legado):   retorna tupla ((1, num_preds, nc+5), features)
    #                    ou tensor    (1, num_preds, nc+5)   — shape[1] >> shape[2]
    # YOLOv8/Ultralytics: retorna tensor (1, nc+4, num_preds) diretamente
    #                    sem objectness score — shape[1] << shape[2]
    if isinstance(raw, (tuple, list)):
        pred = raw[0]
    else:
        pred = raw

    yolov8 = pred.ndim == 3 and pred.shape[1] < pred.shape[2]
    if yolov8:
        # Transpõe de (batch, nc+4, num_preds) → (batch, num_preds, nc+4)
        pred = pred.permute(0, 2, 1)

    # ── Pós-processamento: NMS ────────────────────────────────────────────────
    pred = non_max_suppression(pred, conf_thres, iou_thres, yolov8=yolov8)[0]

    if pred is None or not len(pred):
        return []

    # Reescala coordenadas de volta ao espaço do frame original
    # img_t.shape[2:] = (H', W') do frame letterboxed
    pred[:, :4] = scale_boxes(img_t.shape[2:], pred[:, :4], frame.shape).round()

    detections: list[RawDetection] = []
    for *xyxy, conf, cls_id in pred:
        x1, y1, x2, y2 = map(int, xyxy)
        label = names[int(cls_id)].upper()
        detections.append(RawDetection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            label=label,
            confidence=float(conf),
        ))

    # Ordenar por confiança decrescente (mais confiante primeiro)
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections
