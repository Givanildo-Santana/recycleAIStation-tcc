from datetime import datetime

from db.database import get_connection


def record(
    event_type: str,
    description: str = None,
    param_key: str = None,
    old_value=None,
    new_value=None,
    user_id: int = None,
):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO audit_log
            (event_type, description, param_key, old_value, new_value, user_id, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_type,
            description,
            param_key,
            str(old_value) if old_value is not None else None,
            str(new_value) if new_value is not None else None,
            user_id,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def get_recent(limit: int = 100):
    return get_connection().execute(
        """
        SELECT al.*, u.login
        FROM audit_log al
        LEFT JOIN users u ON u.id = al.user_id
        ORDER BY al.ts DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def get_by_event(event_type: str, limit: int = 50):
    return get_connection().execute(
        "SELECT * FROM audit_log WHERE event_type = ? ORDER BY ts DESC LIMIT ?",
        (event_type, limit),
    ).fetchall()
