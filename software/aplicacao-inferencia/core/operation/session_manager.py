"""
Gerenciador de sessões operacionais.

Camada de negócio entre OperationScreen e session_repo.
Stateless: nenhum estado é mantido entre chamadas.
O chamador (OperationScreen) guarda op_session_id e started_at.

Responsabilidades:
  - Abrir sessão operacional formal com snapshot de configuração e modelo ativo
  - Fechar sessão com totais, duração e status
  - Registrar eventos (erros/avisos) da sessão
  - Armazenar métricas técnicas pré-computadas (perfil maintenance)
  - Auditar início e fim via audit_repo

Esta aplicação NÃO treina modelos. Métricas são de INFERÊNCIA, não de treino.
"""
from __future__ import annotations

from datetime import datetime

from core.auth.session import Session
from db.repositories import session_repo, audit_repo

# Chaves de configuração capturadas no snapshot da sessão
_SNAPSHOT_KEYS = [
    "realtime.conf_thres",
    "realtime.iou_thres",
    "realtime.source",
    "realtime.device",
    "realtime.roi_x_start",
    "realtime.roi_x_end",
    "realtime.roi_y_start",
    "realtime.roi_y_end",
    "roi_timer.seconds",
    "arduino.port",
    "arduino.baudrate",
    "conveyor.delay_vidro_ms",
    "conveyor.delay_papel_ms",
    "conveyor.delay_plastico_ms",
    "conveyor.delay_metal_ms",
    "conveyor.delay_nao_identificado_ms",
]


def open_session(auth_session: Session) -> tuple[int, datetime]:
    """
    Abre uma sessão operacional formal no banco.

    Captura:
      - modelo ativo no momento da abertura
      - snapshot das configurações operacionais relevantes

    Returns:
        (op_session_id, started_at)
        op_session_id: PK gerada em op_sessions
        started_at: datetime de início (para cálculo de duração ao fechar)
    """
    from core.settings import settings_manager
    from db.repositories import model_repo

    # ── Modelo ativo ──
    model_id   = None
    model_name = None
    try:
        row = model_repo.get_active()
        if row:
            model_id   = row["id"]
            model_name = row["name"]
    except Exception:
        pass

    # ── Snapshot de configuração ──
    snapshot: dict[str, str] = {}
    for key in _SNAPSHOT_KEYS:
        try:
            snapshot[key] = settings_manager.get(key)
        except Exception:
            pass

    started_at = datetime.now()

    op_session_id = session_repo.open_session(
        session_id      = auth_session.session_id,
        user_id         = auth_session.user_id,
        user_login      = auth_session.login,
        profile         = auth_session.role,
        model_id        = model_id,
        model_name      = model_name,
        config_snapshot = snapshot,
    )

    audit_repo.record(
        "OPERATION_SESSION_OPEN",
        description=(
            f"Sessao operacional aberta — operador={auth_session.login}"
            f" perfil={auth_session.role} modelo={model_name or 'N/A'}"
        ),
        user_id=auth_session.user_id,
    )

    return op_session_id, started_at


def close_session(
    op_session_id: int,
    started_at: datetime,
    counts: dict[str, int],
    auth_session: Session,
    status: str = "closed",
    notes: str | None = None,
) -> None:
    """
    Fecha a sessão com os totais finais e duração calculada.

    Args:
        op_session_id: PK da sessão em op_sessions
        started_at:    datetime retornado por open_session()
        counts:        dict {classe: quantidade} acumulado durante a operação
        auth_session:  sessão de autenticação (para auditoria)
        status:        'closed' | 'error'
        notes:         observação opcional
    """
    ended_at   = datetime.now()
    duration_s = (ended_at - started_at).total_seconds()
    total      = sum(counts.values())

    session_repo.close_session(
        op_session_id = op_session_id,
        duration_s    = duration_s,
        total_items   = total,
        counts        = counts,
        status        = status,
        notes         = notes,
    )

    audit_repo.record(
        "OPERATION_SESSION_CLOSE",
        description=(
            f"Sessao encerrada — {total} item(s)"
            f" duracao={duration_s:.1f}s operador={auth_session.login}"
        ),
        user_id=auth_session.user_id,
    )


def record_event(
    op_session_id: int,
    event_type: str,
    detail: str | None = None,
) -> None:
    """
    Registra um evento (ERROR/WARNING/INFO) associado à sessão.
    Falha silenciosa: não deve interromper a operação.
    """
    try:
        session_repo.add_event(op_session_id, event_type, detail)
    except Exception:
        pass


def record_metrics(op_session_id: int, stats: dict) -> None:
    """
    Armazena métricas técnicas pré-computadas pelo _TrabalhadorInferencia.

    Chamado apenas quando o perfil é `maintenance`.
    Recebe um dict com as seguintes chaves (todas opcionais):

      infer_count     int    — total de inferências executadas
      infer_time_min  float  — tempo mínimo por inferência (segundos)
      infer_time_avg  float  — tempo médio por inferência
      infer_time_max  float  — tempo máximo por inferência
      conf_min        float  — confiança mínima das detecções
      conf_avg        float  — confiança média
      conf_max        float  — confiança máxima
      fps_avg         float  — FPS médio da sessão
      serial_port     str    — porta serial usada
      model_path      str    — caminho do modelo ativo
      extra           dict   — campos livres adicionais

    Não lança exceção — falha é registrada silenciosamente.
    """
    try:
        session_repo.upsert_metrics(
            op_session_id   = op_session_id,
            infer_count     = stats.get("infer_count", 0),
            infer_time_min  = stats.get("infer_time_min"),
            infer_time_avg  = stats.get("infer_time_avg"),
            infer_time_max  = stats.get("infer_time_max"),
            conf_min        = stats.get("conf_min"),
            conf_avg        = stats.get("conf_avg"),
            conf_max        = stats.get("conf_max"),
            fps_avg         = stats.get("fps_avg"),
            serial_port     = stats.get("serial_port"),
            model_path      = stats.get("model_path"),
            extra           = stats.get("extra"),
        )
    except Exception:
        pass
