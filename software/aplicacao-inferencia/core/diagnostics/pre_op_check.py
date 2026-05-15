"""
Checagens automáticas de pré-operação.

Cada check é isolado: uma falha não impede os outros de rodarem.
Resultados são gravados em diagnostics_repo e audit_log.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    elapsed_ms: int = 0

    def status_str(self) -> str:
        return "OK" if self.ok else "FALHA"


@dataclass
class PreOpReport:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    db_check:     CheckResult | None = None
    config_check: CheckResult | None = None
    model_check:  CheckResult | None = None
    camera_check: CheckResult | None = None
    serial_check: CheckResult | None = None
    actuators:    dict[str, CheckResult] = field(default_factory=dict)

    def critical_checks(self) -> list[CheckResult]:
        return [c for c in (
            self.db_check, self.config_check,
            self.model_check, self.camera_check, self.serial_check,
        ) if c is not None]

    def all_critical_ok(self) -> bool:
        checks = self.critical_checks()
        return bool(checks) and all(c.ok for c in checks)

    def summary(self) -> str:
        checks = self.critical_checks()
        ok = sum(1 for c in checks if c.ok)
        return f"{ok}/{len(checks)} checagens críticas OK"


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _timed(name: str, fn) -> CheckResult:
    t0 = time.monotonic()
    try:
        ok, detail = fn()
    except Exception as exc:
        ok, detail = False, str(exc)
    return CheckResult(
        name=name,
        ok=ok,
        detail=detail,
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# Checagens individuais
# ---------------------------------------------------------------------------

def _check_db() -> tuple[bool, str]:
    from db.database import get_connection
    get_connection().execute("SELECT 1")
    return True, "Conexão SQLite ativa"


def _check_config() -> tuple[bool, str]:
    from core.settings import settings_manager
    cfg = settings_manager.get_all()
    if not cfg:
        return False, "Nenhuma configuração encontrada no banco"
    return True, f"{len(cfg)} parâmetros carregados"


def _check_model() -> tuple[bool, str]:
    from core.detection.model_registry import get_active_path
    from db.repositories import model_repo

    row = model_repo.get_active()
    if row is None:
        return False, "Nenhum modelo ativo registrado no banco"

    path = get_active_path()
    if path is None or not path.exists():
        return False, f"Arquivo do modelo não encontrado: {row['file_path']}"

    size_mb = path.stat().st_size / (1024 * 1024)
    return True, f"{row['name']} — {path.name} ({size_mb:.1f} MB)"


def _check_camera() -> tuple[bool, str]:
    """
    Verifica disponibilidade da câmera com fallback automático de backend.

    Tenta o backend padrão (MSMF no Windows) primeiro. Se travar ou falhar,
    tenta CAP_DSHOW automaticamente — compatível com webcams USB cujo driver
    não responde ao MSMF. Cada tentativa tem timeout individual; total ≤ 10 s.

    Delega ao módulo camera_open que encapsula a lógica de threading e fallback.
    """
    import cv2
    from core.settings import settings_manager
    from core.diagnostics.camera_open import open_camera

    source_str = settings_manager.get("realtime.source")
    source = int(source_str) if source_str.isdigit() else source_str

    cap, detail = open_camera(source, timeout_s=10.0)
    if cap is None:
        return False, detail

    # Câmera aberta e frame validado — extrair dimensões antes de liberar
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        detail = f"{detail} — {w}×{h}"
    except Exception:
        pass
    finally:
        try:
            cap.release()
        except Exception:
            pass

    return True, detail


def _eval_handshake(response: str | None, port: str) -> tuple[bool, str]:
    """
    Interpreta a resposta do handshake PING_RECYCLEAI e devolve (ok, detalhe).

    Casos:
      None                      → firmware diferente ou porta errada → falha
      PONG_RECYCLEAI:OK         → master + slave saudáveis → OK
      PONG_RECYCLEAI:SLAVE_ERROR → master OK, slave com falha → falha
      qualquer outro PONG       → resposta inesperada → falha (defensivo)
    """
    if response is None:
        return False, (
            f"{port}: sem resposta ao PING_RECYCLEAI "
            "(firmware diferente ou porta serial incorreta)"
        )
    if "PONG_RECYCLEAI:OK" in response:
        return True, f"{port}: master OK, slave OK [handshake completo]"
    # DECISÃO DE DESIGN: pré-op rejeita SLAVE_ERROR porque triagem de produção
    # exige sistema I2C completo (master + slave). Comportamento assimétrico ao
    # diagnóstico (_ConexaoWorker em hardware_diag_dialog.py), que aceita
    # SLAVE_ERROR com aviso — pois permite inspecionar hardware degradado.
    if "PONG_RECYCLEAI:SLAVE_ERROR" in response:
        return False, (
            f"{port}: Arduino RecycleAI identificado, mas slave I2C com falha "
            "(SLAVE_ERROR). Verifique a conexão entre master e slave."
        )
    # PONG_RECYCLEAI presente mas conteúdo não reconhecido
    return False, (
        f"{port}: resposta PONG inesperada: '{response}'. "
        "Hardware não considerado apto."
    )


def _check_serial() -> tuple[bool, str]:
    from core.hardware.serial_handler import GerenciadorSerial

    # ── Caminho rápido: porta configurada ────────────────────────────────────
    handler = GerenciadorSerial.from_config()
    if handler.connect():
        response = handler.handshake()
        handler.stop()
        ok, detail = _eval_handshake(response, handler.port)
        if ok:
            return True, detail
        # Porta abre mas handshake falha — reporta motivo específico e continua
        # para varredura (pode ser que o Arduino mudou de COM).
        fast_fail_detail = detail
    else:
        fast_fail_detail = handler.connection_detail
        handler.stop()

    # ── Varredura automática de todas as COMs disponíveis ────────────────────
    found = GerenciadorSerial.scan_and_connect()
    if found is None:
        return False, (
            f"Arduino RecycleAI não encontrado em nenhuma porta COM. "
            f"Última tentativa na porta configurada: {fast_fail_detail}"
        )

    # Reutiliza a resposta já obtida durante o scan (evita segundo PING)
    response = found.last_handshake_response
    ok, detail = _eval_handshake(response, found.port)

    if ok:
        # Persiste a porta descoberta
        try:
            from db.repositories import config_repo
            config_repo.set("arduino.port", found.port)
        except Exception:
            pass

    found.stop()
    return ok, detail


# ---------------------------------------------------------------------------
# Ponto de entrada público
# ---------------------------------------------------------------------------

def run(user_id: int | None = None) -> PreOpReport:
    """
    Executa todas as checagens críticas e persiste o resultado no banco.
    Nunca lança exceção — erros individuais ficam no CheckResult.
    """
    report = PreOpReport()

    report.db_check     = _timed("Banco de dados",       _check_db)
    report.config_check = _timed("Configurações",         _check_config)
    report.model_check  = _timed("Modelo TorchScript",    _check_model)
    report.camera_check = _timed("Câmera",                _check_camera)
    report.serial_check = _timed("Arduino (serial)",      _check_serial)

    _persist(report, user_id)
    return report


def _persist(report: PreOpReport, user_id: int | None):
    try:
        from db.repositories import diagnostics_repo, audit_repo
        actuators_dict = {
            k: {"ok": v.ok, "detail": v.detail}
            for k, v in report.actuators.items()
        }
        diagnostics_repo.insert(
            session_id=report.session_id,
            camera_ok=report.camera_check.ok if report.camera_check else False,
            model_ok=report.model_check.ok   if report.model_check  else False,
            serial_ok=report.serial_check.ok if report.serial_check else False,
            actuators=actuators_dict,
            user_id=user_id,
        )
        audit_repo.record("DIAG_RUN", description=report.summary(), user_id=user_id)
    except Exception:
        pass  # diagnóstico não pode falhar por causa do banco
