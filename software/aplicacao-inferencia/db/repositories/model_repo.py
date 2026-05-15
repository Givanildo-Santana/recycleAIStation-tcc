import json
from db.database import get_connection


def register(
    name: str,
    file_path: str,
    fmt: str = "torchscript",
    nc: int = None,
    class_names: list = None,
    origin: str = "manual",
    notes: str = None,
    registered_by: int = None,
    package_dir: str = None,
) -> int:
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO models
            (name, file_path, format, status, nc, class_names, origin, notes,
             registered_by, package_dir)
        VALUES (?, ?, ?, 'inactive', ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            file_path,
            fmt,
            nc,
            json.dumps(class_names) if class_names else None,
            origin,
            notes,
            registered_by,
            package_dir,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_active():
    return get_connection().execute(
        "SELECT * FROM models WHERE status = 'active' LIMIT 1"
    ).fetchone()


def get_by_id(model_id: int):
    return get_connection().execute(
        "SELECT * FROM models WHERE id = ?", (model_id,)
    ).fetchone()


def list_all():
    return get_connection().execute(
        "SELECT * FROM models ORDER BY registered_at DESC"
    ).fetchall()


def set_active(model_id: int):
    conn = get_connection()
    conn.execute("UPDATE models SET status = 'inactive' WHERE status = 'active'")
    conn.execute(
        "UPDATE models SET status = 'active' WHERE id = ?", (model_id,)
    )
    conn.commit()


def update_status(model_id: int, status: str, notes: str = None):
    conn = get_connection()
    if notes is not None:
        conn.execute(
            "UPDATE models SET status = ?, notes = ? WHERE id = ?",
            (status, notes, model_id),
        )
    else:
        conn.execute(
            "UPDATE models SET status = ? WHERE id = ?", (status, model_id)
        )
    conn.commit()


def exists_any() -> bool:
    row = get_connection().execute(
        "SELECT COUNT(*) AS cnt FROM models"
    ).fetchone()
    return row["cnt"] > 0
