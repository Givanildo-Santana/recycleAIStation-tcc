-- v005: Remove UNIQUE constraint de op_sessions.session_id
--
-- Problema:
--   A coluna session_id armazena o UUID da sessão de autenticação.
--   O operador pode iniciar e parar a triagem várias vezes na mesma
--   sessão de login, gerando múltiplas op_sessions com o mesmo session_id.
--   A constraint UNIQUE impedia o segundo INSERT, causando falha silenciosa
--   em _do_start() a partir da segunda tentativa.
--
-- Solução:
--   Recriar op_sessions sem UNIQUE em session_id.
--   A unicidade de cada registro operacional é garantida pelo PK id (autoincrement).
--   session_id permanece como referência de qual sessão de auth gerou a operação.
--
-- Efeito colateral corrigido:
--   Sessões operacionais com status='open' e ended_at IS NULL são marcadas
--   como 'error' para não deixar registros órfãos de tentativas falhas anteriores.

-- Fechar sessões órfãs antes da recriação da tabela
UPDATE op_sessions
SET    status    = 'error',
       ended_at  = datetime('now', 'localtime'),
       notes     = 'Encerrada automaticamente — sessao orfa (migracao v005)'
WHERE  status = 'open' AND ended_at IS NULL;

-- Recriar op_sessions sem UNIQUE em session_id
CREATE TABLE op_sessions_v2 (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    user_id         INTEGER REFERENCES users(id),
    user_login      TEXT    NOT NULL,
    profile         TEXT    NOT NULL CHECK(profile IN ('operator','maintenance')),
    model_id        INTEGER REFERENCES models(id),
    model_name      TEXT,
    config_snapshot TEXT,
    started_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    ended_at        TEXT,
    duration_s      REAL,
    total_items     INTEGER NOT NULL DEFAULT 0,
    counts_json     TEXT,
    status          TEXT    NOT NULL DEFAULT 'open'
                            CHECK(status IN ('open','closed','error')),
    notes           TEXT
);

INSERT INTO op_sessions_v2 SELECT * FROM op_sessions;

DROP TABLE op_sessions;

ALTER TABLE op_sessions_v2 RENAME TO op_sessions;
