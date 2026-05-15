"""
Sondagem individual de atuadores via comunicação serial.

Todos os métodos aceitam serial_handler=None e retornam ProbeResult
descritivo sem bloquear o sistema quando o Arduino não está conectado.

Protocolo com o Arduino (master.ino):
  Python → "Status\n"
  Arduino → "Status: A2:2,A3:2,A4:2,A5:2\n"
    onde 0=AVANCANDO, 1=RETORNANDO, 2=PARADO

Parser canônico do formato de status: diag_protocol.parse_actuator_status().
Este módulo importa e reutiliza essa implementação — não define parser próprio.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from core.diagnostics.diag_protocol import parse_actuator_status as _parse_status

ACTUATOR_IDS = ["A2", "A3", "A4", "A5"]
_CMD_STATUS = "Status"
_DEFAULT_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# Estrutura de resultado
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    target: str
    ok: bool | None  # None = inconclusivo (ex.: sem ack, confirmar visualmente)
    state: str = ""
    detail: str = ""

    def is_at_rest(self) -> bool:
        return self.state == "PARADO"


# ---------------------------------------------------------------------------
# Sondagem de atuadores via serial_handler
# ---------------------------------------------------------------------------

def probe_actuator_status(serial_handler, actuator_id: str) -> ProbeResult:
    """
    Solicita "Status" ao Arduino e interpreta a resposta para um atuador específico.

    Requer serial_handler com métodos:
      is_connected() -> bool
      send(msg: str)
      read_line(timeout: float) -> str | None

    Retorna ProbeResult(ok=False) se serial_handler for None.
    """
    if serial_handler is None or not _is_connected(serial_handler):
        return ProbeResult(
            target=actuator_id, ok=False,
            detail="Serial não disponível — Arduino não conectado"
        )

    try:
        serial_handler.send(_CMD_STATUS)
        t0 = time.monotonic()
        while time.monotonic() - t0 < _DEFAULT_TIMEOUT_S:
            line = _read_line(serial_handler, timeout=0.5)
            if line and "A2:" in line:
                states = _parse_status(line) or {}
                state = states.get(actuator_id, "DESCONHECIDO")
                ok = state == "PARADO"
                return ProbeResult(
                    target=actuator_id,
                    ok=ok,
                    state=state,
                    detail=f"Status: {line.strip()}"
                )
        return ProbeResult(
            target=actuator_id, ok=False,
            detail=f"Timeout aguardando status de {actuator_id}"
        )
    except Exception as exc:
        return ProbeResult(target=actuator_id, ok=False, detail=str(exc))


def probe_all_actuators(serial_handler) -> dict[str, ProbeResult]:
    """Sonda todos os atuadores com um único comando Status."""
    if serial_handler is None or not _is_connected(serial_handler):
        return {
            aid: ProbeResult(
                target=aid, ok=False,
                detail="Serial não disponível"
            )
            for aid in ACTUATOR_IDS
        }

    try:
        serial_handler.send(_CMD_STATUS)
        t0 = time.monotonic()
        while time.monotonic() - t0 < _DEFAULT_TIMEOUT_S:
            line = _read_line(serial_handler, timeout=0.5)
            if line and "A2:" in line:
                states = _parse_status(line) or {}
                return {
                    aid: ProbeResult(
                        target=aid,
                        ok=states.get(aid) == "PARADO",
                        state=states.get(aid, "DESCONHECIDO"),
                        detail="Status recebido"
                    )
                    for aid in ACTUATOR_IDS
                }
        return {
            aid: ProbeResult(target=aid, ok=False, detail="Timeout")
            for aid in ACTUATOR_IDS
        }
    except Exception as exc:
        return {
            aid: ProbeResult(target=aid, ok=False, detail=str(exc))
            for aid in ACTUATOR_IDS
        }


# ---------------------------------------------------------------------------
# Helpers de interface com serial_handler
# ---------------------------------------------------------------------------

def _is_connected(handler) -> bool:
    try:
        return handler.is_connected()
    except AttributeError:
        # SerialHandler legado usa serial_obj em vez de is_connected()
        return getattr(handler, "serial_obj", None) is not None


def _read_line(handler, timeout: float) -> str | None:
    try:
        return handler.read_line(timeout=timeout)
    except AttributeError:
        # Fallback defensivo para objetos que não implementem read_line()
        # (ex.: mocks em teste ou integrações não-padrão).
        # GerenciadorSerial (serial_handler.py) implementa read_line() —
        # este ramo nunca é ativado com o handler de produção atual.
        # Se AttributeError aparecer em uso normal, é sinal de regressão de API.
        return None
