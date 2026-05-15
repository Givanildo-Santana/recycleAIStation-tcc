"""
Utilitários de processamento de imagem para inferência YOLOv5/YOLOv8.

─── Funções ─────────────────────────────────────────────────────────────────
  letterbox           — redimensionamento com padding (aspect-ratio preservado)
  non_max_suppression — NMS puro PyTorch, sem torchvision (YOLOv5 e YOLOv8)
  scale_boxes         — reescala coordenadas para o frame original

─── NMS puro PyTorch ────────────────────────────────────────────────────────
  Implementação greedy equivalente a torchvision.ops.nms.
  Sem dependência de torchvision — evita circular import no bundle PyInstaller
  causado pelo module_collection_mode='pyz+py' do hook padrão do torchvision.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Pré-processamento
# ─────────────────────────────────────────────────────────────────────────────

def letterbox(
    im: np.ndarray,
    new_shape: int | tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    stride: int = 32,
    **_kwargs,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Redimensiona im para new_shape mantendo aspect-ratio e preenchendo com color.

    Retorna:
      (im_padded, ratio, (dw, dh))  — mesmo formato que yolov5.letterbox
    """
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    nh, nw = new_shape

    r = min(nh / h, nw / w)
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))

    dw = nw - new_unpad_w   # padding horizontal total
    dh = nh - new_unpad_h   # padding vertical total

    if auto:                 # reduz padding ao múltiplo de stride
        dw, dh = dw % stride, dh % stride

    dw /= 2.0
    dh /= 2.0

    if (w, h) != (new_unpad_w, new_unpad_h):
        im = cv2.resize(im, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


# ─────────────────────────────────────────────────────────────────────────────
# Pós-processamento
# ─────────────────────────────────────────────────────────────────────────────

def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Converte formato (cx, cy, w, h) → (x1, y1, x2, y2)."""
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2   # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2   # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2   # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2   # y2
    return y


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    NMS greedy puro PyTorch — sem torchvision.

    Evita circular import no bundle PyInstaller: o hook padrão do torchvision usa
    module_collection_mode='pyz+py', que em PyInstaller 6.x causa inicialização dupla
    do módulo quando importado em runtime → "partially initialized module 'torchvision'".
    """
    if not boxes.shape[0]:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        inter = (
            (x2[rest].clamp(max=x2[i].item()) - x1[rest].clamp(min=x1[i].item())).clamp(min=0)
            * (y2[rest].clamp(max=y2[i].item()) - y1[rest].clamp(min=y1[i].item())).clamp(min=0)
        )
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    yolov8: bool = False,
    **_kwargs,
) -> list[torch.Tensor]:
    """
    NMS para saída YOLOv5 e YOLOv8/Ultralytics TorchScript.

    prediction: [batch, num_preds, num_attrs]  (já normalizado pelo chamador)
      YOLOv5:  num_attrs = nc+5  →  [cx, cy, w, h, obj_conf, cls0…clsN]
      YOLOv8:  num_attrs = nc+4  →  [cx, cy, w, h, cls0…clsN]  (sem objectness)

    Retorna:
      list[Tensor] — um tensor [N, 6] por imagem do batch.
      Cada linha: [x1, y1, x2, y2, confidence, class_id]
    """
    bs     = prediction.shape[0]
    device = prediction.device
    output = [torch.zeros((0, 6), device=device)] * bs

    for xi, x in enumerate(prediction):
        if yolov8:
            # ── YOLOv8: sem objectness; confiança = max(class_scores) ────
            scores, cls_ids = x[:, 4:].max(dim=1)
            mask    = scores > conf_thres
            x       = x[mask]
            scores  = scores[mask]
            cls_ids = cls_ids[mask]
            if not x.shape[0]:
                continue
            boxes = _xywh2xyxy(x[:, :4])
        else:
            # ── YOLOv5: conf = objectness × max(class_scores) ─────────────
            x = x[x[:, 4] > conf_thres]
            if not x.shape[0]:
                continue
            x[:, 5:] *= x[:, 4:5]
            boxes = _xywh2xyxy(x[:, :4])
            scores, cls_ids = x[:, 5:].max(dim=1)
            mask    = scores > conf_thres
            boxes   = boxes[mask]
            scores  = scores[mask]
            cls_ids = cls_ids[mask]
            if not boxes.shape[0]:
                continue

        keep = _nms(boxes.float(), scores.float(), iou_thres)
        det  = torch.cat([
            boxes[keep],
            scores[keep].unsqueeze(1),
            cls_ids[keep].float().unsqueeze(1),
        ], dim=1)
        output[xi] = det

    return output


def scale_boxes(
    img1_shape: tuple[int, int],
    boxes: torch.Tensor,
    img0_shape: tuple,
    ratio_pad=None,
) -> torch.Tensor:
    """
    Reescala coordenadas de caixas de img1_shape → img0_shape.

    img1_shape — (H', W') do frame letterboxed (tensor.shape[2:])
    img0_shape — (H, W, C) ou (H, W) do frame original (frame.shape)
    boxes      — tensor [..., 4] em formato (x1, y1, x2, y2), modificado in-place

    Retorna boxes reescaladas (mesmo tensor).
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad  = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0]
        pad  = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]   # remove padding horizontal
    boxes[..., [1, 3]] -= pad[1]   # remove padding vertical
    boxes[..., :4]     /= gain      # reescala ao tamanho original

    # Clamp ao tamanho do frame original
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img0_shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img0_shape[0])

    return boxes


__all__ = ["letterbox", "non_max_suppression", "scale_boxes"]
