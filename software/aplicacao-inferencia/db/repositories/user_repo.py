from db.database import get_connection


def get_by_login(login: str):
    return get_connection().execute(
        "SELECT * FROM users WHERE login = ? AND active = 1", (login,)
    ).fetchone()


def get_by_id(user_id: int):
    return get_connection().execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()


def create(login: str, password_hash: str, role: str,
           must_change_password: bool = False, created_by: int = None):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO users (login, password_hash, role, must_change_password, created_by)
        VALUES (?, ?, ?, ?, ?)
        """,
        (login, password_hash, role, int(must_change_password), created_by),
    )
    conn.commit()


def update_password(user_id: int, new_hash: str, must_change_password: bool = False):
    conn = get_connection()
    conn.execute(
        "UPDATE users SET password_hash = ?, must_change_password = ? WHERE id = ?",
        (new_hash, int(must_change_password), user_id),
    )
    conn.commit()


def list_active():
    return get_connection().execute(
        "SELECT id, login, role, created_at FROM users WHERE active = 1 ORDER BY login"
    ).fetchall()


def list_all():
    """Lista todos os usuários (ativos e inativos) — para painel administrativo."""
    return get_connection().execute(
        """SELECT id, login, role, active, must_change_password, created_at
           FROM users ORDER BY login"""
    ).fetchall()


def activate(user_id: int):
    conn = get_connection()
    conn.execute("UPDATE users SET active = 1 WHERE id = ?", (user_id,))
    conn.commit()


def deactivate(user_id: int):
    conn = get_connection()
    conn.execute("UPDATE users SET active = 0 WHERE id = ?", (user_id,))
    conn.commit()


def count_active_maintenance() -> int:
    """Conta contas maintenance ativas — usado para impedir desativar a última."""
    row = get_connection().execute(
        "SELECT COUNT(*) AS cnt FROM users WHERE role='maintenance' AND active=1"
    ).fetchone()
    return row["cnt"]


def delete(user_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()


def exists_any() -> bool:
    row = get_connection().execute("SELECT COUNT(*) AS cnt FROM users").fetchone()
    return row["cnt"] > 0
