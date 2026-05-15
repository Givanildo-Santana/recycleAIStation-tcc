-- v003_sessions: sessões operacionais formais, eventos e métricas técnicas
--
-- Objetivo:
--   Registrar cada operação como uma sessão formal com totais, snapshot de
--   configuração e métricas técnicas de inferência (apenas perfil maintenance).
--
-- Relacionamento com tabela existente `detections`:
--   detections.session_id (TEXT UUID) referencia op_sessions.session_id — sem FK
--   para manter compatibilidade com registros anteriores à migração.

-- ── Sessão operacional principal ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS op_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL UNIQUE,    -- UUID da Session de autenticação
    user_id         INTEGER REFERENCES users(id),
    user_login      TEXT    NOT NULL,
    profile         TEXT    NOT NULL CHECK(profile IN ('operator','maintenance')),
    model_id        INTEGER REFERENCES models(id),
    model_name      TEXT,
    config_snapshot TEXT,                       -- JSON das configs relevantes no início
    started_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    ended_at        TEXT,
    duration_s      REAL,
    total_items     INTEGER NOT NULL DEFAULT 0,
    counts_json     TEXT,                       -- JSON {"metal":3,"papel":2,...}
    status          TEXT    NOT NULL DEFAULT 'open'
                            CHECK(status IN ('open','closed','error')),
    notes           TEXT
);

-- ── Eventos da sessão (erros, avisos, informações relevantes) ─────────────────
CREATE TABLE IF NOT EXISTS session_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    op_session_id   INTEGER NOT NULL REFERENCES op_sessions(id),
    event_type      TEXT    NOT NULL,           -- ERROR | WARNING | INFO
    detail          TEXT,
    ts              TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- ── Métricas técnicas de inferência (principalmente perfil maintenance) ────────
-- UNIQUE em op_session_id: upsert garante apenas 1 linha por sessão
CREATE TABLE IF NOT EXISTS session_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    op_session_id   INTEGER NOT NULL REFERENCES op_sessions(id) UNIQUE,
    infer_count     INTEGER NOT NULL DEFAULT 0,
    infer_time_min  REAL,                       -- segundos
    infer_time_avg  REAL,
    infer_time_max  REAL,
    conf_min        REAL,                       -- confiança das detecções
    conf_avg        REAL,
    conf_max        REAL,
    fps_avg         REAL,
    serial_port     TEXT,
    model_path      TEXT,
    extra_json      TEXT                        -- JSON livre para dados adicionais
);
