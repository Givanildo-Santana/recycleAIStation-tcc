from db.database import get_connection


def insert(label: str, confidence: float, session_id: str, user_id: int):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO detections (label, confidence, session_id, user_id)
        VALUES (?, ?, ?, ?)
        """,
        (label, confidence, session_id, user_id),
    )
    conn.commit()


def get_by_session(session_id: str):
    return get_connection().execute(
        "SELECT * FROM detections WHERE session_id = ? ORDER BY confirmed_at",
        (session_id,),
    ).fetchall()


def count_by_label(session_id: str) -> dict:
    rows = get_connection().execute(
        """
        SELECT label, COUNT(*) AS total
        FROM detections WHERE session_id = ?
        GROUP BY label
        """,
        (session_id,),
    ).fetchall()
    return {row["label"]: row["total"] for row in rows}
