import json
from db.database import get_connection


def insert(
    session_id: str,
    camera_ok: bool,
    model_ok: bool,
    serial_ok: bool,
    actuators: dict,
    conveyor_ok: bool = None,
    user_id: int = None,
):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO diagnostics
            (session_id, camera_ok, model_ok, serial_ok, actuators_json, conveyor_ok, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            int(camera_ok),
            int(model_ok),
            int(serial_ok),
            json.dumps(actuators),
            int(conveyor_ok) if conveyor_ok is not None else None,
            user_id,
        ),
    )
    conn.commit()


def update_conveyor(session_id: str, conveyor_ok: bool):
    conn = get_connection()
    conn.execute(
        "UPDATE diagnostics SET conveyor_ok = ? WHERE session_id = ?",
        (int(conveyor_ok), session_id),
    )
    conn.commit()


def get_latest(session_id: str):
    return get_connection().execute(
        "SELECT * FROM diagnostics WHERE session_id = ? ORDER BY run_at DESC LIMIT 1",
        (session_id,),
    ).fetchone()
