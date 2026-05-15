from db.database import get_connection


def get(param_key: str) -> str | None:
    row = get_connection().execute(
        "SELECT param_value FROM system_config WHERE param_key = ?", (param_key,)
    ).fetchone()
    return row["param_value"] if row else None


def get_all() -> dict:
    rows = get_connection().execute(
        "SELECT param_key, param_value FROM system_config"
    ).fetchall()
    return {row["param_key"]: row["param_value"] for row in rows}


def set(param_key: str, param_value: str):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO system_config (param_key, param_value, updated_at)
        VALUES (?, ?, datetime('now', 'localtime'))
        ON CONFLICT(param_key) DO UPDATE SET
            param_value = excluded.param_value,
            updated_at  = excluded.updated_at
        """,
        (param_key, param_value),
    )
    conn.commit()


def record_history(param_key: str, old_value, new_value, changed_by: int):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO config_history (param_key, old_value, new_value, changed_by)
        VALUES (?, ?, ?, ?)
        """,
        (
            param_key,
            str(old_value) if old_value is not None else None,
            str(new_value),
            changed_by,
        ),
    )
    conn.commit()


def history(param_key: str, limit: int = 50):
    return get_connection().execute(
        """
        SELECT ch.*, u.login
        FROM config_history ch
        LEFT JOIN users u ON u.id = ch.changed_by
        WHERE ch.param_key = ?
        ORDER BY ch.changed_at DESC
        LIMIT ?
        """,
        (param_key, limit),
    ).fetchall()
