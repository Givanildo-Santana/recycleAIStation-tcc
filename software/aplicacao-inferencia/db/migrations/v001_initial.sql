-- v001_initial: schema base do RecycleAI-Station

CREATE TABLE IF NOT EXISTS users (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    login                TEXT    UNIQUE NOT NULL,
    password_hash        TEXT    NOT NULL,
    role                 TEXT    NOT NULL CHECK(role IN ('operator','maintenance')),
    active               INTEGER NOT NULL DEFAULT 1,
    must_change_password INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    created_by           INTEGER REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS detections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    label        TEXT    NOT NULL,
    confidence   REAL    NOT NULL,
    confirmed_at TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    session_id   TEXT    NOT NULL,
    user_id      INTEGER REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS system_config (
    param_key   TEXT PRIMARY KEY,
    param_value TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS config_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    param_key   TEXT    NOT NULL,
    old_value   TEXT,
    new_value   TEXT    NOT NULL,
    changed_by  INTEGER REFERENCES users(id),
    changed_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT    NOT NULL,
    description TEXT,
    param_key   TEXT,
    old_value   TEXT,
    new_value   TEXT,
    user_id     INTEGER REFERENCES users(id),
    ts          TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS diagnostics (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT    NOT NULL,
    camera_ok      INTEGER,
    model_ok       INTEGER,
    serial_ok      INTEGER,
    actuators_json TEXT,
    conveyor_ok    INTEGER,
    run_at         TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    user_id        INTEGER REFERENCES users(id)
);
