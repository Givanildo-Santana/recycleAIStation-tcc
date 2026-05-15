import sqlite3
import threading
from pathlib import Path

from core.utils.paths import project_root, bundle_data_root

# DB fica num local gravável (ao lado do .exe em bundle; raiz do projeto em dev).
_DB_PATH        = project_root() / "data" / "recycleai.db"
# Migrations são somente-leitura — podem estar dentro do _MEIPASS em bundle.
_MIGRATIONS_DIR = bundle_data_root() / "db" / "migrations"

_local = threading.local()


def get_connection() -> sqlite3.Connection:
    if getattr(_local, "conn", None) is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        _local.conn = conn
    return _local.conn


def initialize():
    conn = get_connection()
    _ensure_migrations_table(conn)
    _run_pending_migrations(conn)


def _ensure_migrations_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.commit()


def _run_pending_migrations(conn: sqlite3.Connection):
    applied = {
        row["version"]
        for row in conn.execute("SELECT version FROM schema_migrations")
    }
    for path in sorted(_MIGRATIONS_DIR.glob("v*.sql")):
        version = path.stem
        if version in applied:
            continue
        conn.executescript(path.read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO schema_migrations (version) VALUES (?)", (version,)
        )
        conn.commit()
        print(f"[DB] Migration aplicada: {version}")


def close():
    conn = getattr(_local, "conn", None)
    if conn:
        conn.close()
        _local.conn = None
