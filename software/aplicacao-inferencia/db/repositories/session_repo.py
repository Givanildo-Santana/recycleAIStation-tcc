"""
Repositório de sessões operacionais.

Tabelas: op_sessions, session_events, session_metrics
Sem lógica de negócio — apenas SQL.
"""
from __future__ import annotations

import json
from datetime import datetime

from db.database import get_connection


# ─────────────────────────────────────────────────────── op_sessions ──────────

def open_session(
    session_id: str,
    user_id: int,
    user_login: str,
    profile: str,
    model_id: int | None = None,
    model_name: str | None = None,
    config_snapshot: dict | None = None,
) -> int:
    """Insere uma sessão com status='open'. Retorna o id gerado."""
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO op_sessions
            (session_id, user_id, user_login, profile, model_id, model_name,
             config_snapshot, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id, user_id, user_login, profile,
            model_id, model_name,
            json.dumps(config_snapshot, ensure_ascii=False) if config_snapshot else None,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    return cur.lastrowid


def close_session(
    op_session_id: int,
    duration_s: float,
    total_items: int,
    counts: dict[str, int],
    status: str = "closed",
    notes: str | None = None,
) -> None:
    """Atualiza a sessão com os totais finais e a marca como encerrada."""
    conn = get_connection()
    conn.execute(
        """
        UPDATE op_sessions SET
            ended_at    = datetime('now', 'localtime'),
            duration_s  = ?,
            total_items = ?,
            counts_json = ?,
            status      = ?,
            notes       = ?
        WHERE id = ?
        """,
        (
            round(duration_s, 2),
            total_items,
            json.dumps(counts, ensure_ascii=False),
            status,
            notes,
            op_session_id,
        ),
    )
    conn.commit()


def mark_error(op_session_id: int, notes: str | None = None) -> None:
    """Marca a sessão como encerrada com erro."""
    conn = get_connection()
    conn.execute(
        """
        UPDATE op_sessions
        SET status='error', ended_at=datetime('now', 'localtime'), notes=?
        WHERE id=?
        """,
        (notes, op_session_id),
    )
    conn.commit()


def get_by_id(op_session_id: int):
    return get_connection().execute(
        "SELECT * FROM op_sessions WHERE id = ?", (op_session_id,)
    ).fetchone()


def list_recent(limit: int = 50):
    return get_connection().execute(
        "SELECT * FROM op_sessions ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()


# ──────────────────────────────────────────────────── session_events ──────────

def add_event(
    op_session_id: int,
    event_type: str,
    detail: str | None = None,
) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO session_events (op_session_id, event_type, detail, ts) VALUES (?, ?, ?, ?)",
        (op_session_id, event_type, detail, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()


def get_events(op_session_id: int):
    return get_connection().execute(
        "SELECT * FROM session_events WHERE op_session_id=? ORDER BY ts",
        (op_session_id,),
    ).fetchall()


# ──────────────────────────────────────────────────── session_metrics ─────────

def upsert_metrics(
    op_session_id: int,
    infer_count: int = 0,
    infer_time_min: float | None = None,
    infer_time_avg: float | None = None,
    infer_time_max: float | None = None,
    conf_min: float | None = None,
    conf_avg: float | None = None,
    conf_max: float | None = None,
    fps_avg: float | None = None,
    serial_port: str | None = None,
    model_path: str | None = None,
    extra: dict | None = None,
) -> None:
    """Insere ou atualiza métricas de uma sessão (UNIQUE em op_session_id)."""
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO session_metrics
            (op_session_id, infer_count,
             infer_time_min, infer_time_avg, infer_time_max,
             conf_min, conf_avg, conf_max,
             fps_avg, serial_port, model_path, extra_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(op_session_id) DO UPDATE SET
            infer_count     = excluded.infer_count,
            infer_time_min  = excluded.infer_time_min,
            infer_time_avg  = excluded.infer_time_avg,
            infer_time_max  = excluded.infer_time_max,
            conf_min        = excluded.conf_min,
            conf_avg        = excluded.conf_avg,
            conf_max        = excluded.conf_max,
            fps_avg         = excluded.fps_avg,
            serial_port     = excluded.serial_port,
            model_path      = excluded.model_path,
            extra_json      = excluded.extra_json
        """,
        (
            op_session_id, infer_count,
            infer_time_min, infer_time_avg, infer_time_max,
            conf_min, conf_avg, conf_max,
            fps_avg, serial_port, model_path,
            json.dumps(extra, ensure_ascii=False) if extra else None,
        ),
    )
    conn.commit()


def get_metrics(op_session_id: int):
    return get_connection().execute(
        "SELECT * FROM session_metrics WHERE op_session_id = ?", (op_session_id,)
    ).fetchone()
