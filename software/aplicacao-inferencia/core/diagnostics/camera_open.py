"""
Abertura robusta de câmera com fallback automático de backend.

Contexto
--------
cv2.VideoCapture(n) sem backend explícito usa MSMF (Media Foundation) no
Windows por padrão. MSMF pode travar indefinidamente ao inicializar algumas
webcams USB — o construtor nunca retorna, sem lançar exceção nem devolver False.

Solução
-------
Tenta cada backend em ordem, cada um com timeout de thread individual:
  1. Backend padrão (CAP_ANY / MSMF no Windows)  — cobre câmera interna
  2. CAP_DSHOW (DirectShow)                       — cobre webcam USB no Windows

Regras de fallback:
  - Fallback CAP_DSHOW só é tentado no Windows para índices inteiros locais.
  - URLs (RTSP, arquivo) usam apenas backend padrão.
  - Se o backend padrão abre normalmente, CAP_DSHOW nunca é tentado.

Uso
---
    from core.diagnostics.camera_open import open_camera

    cap, detail = open_camera(source, timeout_s=10.0)
    if cap is None:
        # detail descreve a falha de cada tentativa
        return False, detail
    try:
        # usar cap normalmente — primeiro frame já consumido na validação
        ...
    finally:
        cap.release()
"""
from __future__ import annotations

import sys
import threading

_IS_WINDOWS = sys.platform == "win32"


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def open_camera(
    source: int | str,
    timeout_s: float = 10.0,
):
    """
    Abre câmera com fallback automático de backend.

    Parâmetros
    ----------
    source     : índice inteiro (câmera local) ou string (URL / caminho)
    timeout_s  : tempo máximo total; dividido igualmente entre backends tentados

    Retorna
    -------
    (cap, detalhe)
        cap é um cv2.VideoCapture aberto, isOpened() = True e com um frame
        de validação já lido. O chamador deve chamar cap.release().

    (None, detalhe)
        Falha em todos os backends; detalhe descreve cada tentativa.
    """
    import cv2

    # Fallback DSHOW apenas para índices locais inteiros no Windows
    use_dshow = _IS_WINDOWS and isinstance(source, int)

    if use_dshow:
        per_timeout = max(timeout_s / 2.0, 3.0)
        backends = [
            (cv2.CAP_ANY,   "padrão"),
            (cv2.CAP_DSHOW, "CAP_DSHOW"),
        ]
    else:
        per_timeout = timeout_s
        backends = [(cv2.CAP_ANY, "padrão")]

    errors: list[str] = []

    for backend_id, backend_name in backends:
        cap, err = _try_backend(source, backend_id, per_timeout)
        if cap is not None:
            return cap, f"Aberta via {backend_name} (source={source})"
        errors.append(f"{backend_name}: {err}")

    return None, " | ".join(errors)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _try_backend(
    source: int | str,
    backend: int,
    timeout_s: float,
):
    """
    Tenta abrir câmera com um único backend, com timeout de thread.

    Valida isOpened() + leitura de um frame para confirmar que o driver
    responde e entrega imagem real (não apenas cria o objeto sem resposta).

    Retorna
    -------
    (cap, None)    — sucesso; cap está aberto e pronto para uso pelo chamador
    (None, detail) — falha; detail descreve o motivo
    """
    import cv2

    result: list = [None]   # preenchido pela thread: (cap|None, err|None)

    def _do():
        try:
            cap = cv2.VideoCapture(source, backend)
            if not cap.isOpened():
                cap.release()
                result[0] = (None, "isOpened()=False")
                return
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                result[0] = (None, "sem frame válido")
                return
            # cap fica ABERTO — responsabilidade do chamador
            result[0] = (cap, None)
        except Exception as exc:
            result[0] = (None, str(exc))

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        # Thread travada: driver não respondeu dentro do timeout.
        # Não chamamos cap.release() aqui — o objeto cap ainda está em uso
        # pela thread daemon e será liberado pelo GC ao encerrar.
        return None, f"timeout ({timeout_s:.0f}s) — driver sem resposta"

    if result[0] is None:
        return None, "falha desconhecida"

    return result[0]
