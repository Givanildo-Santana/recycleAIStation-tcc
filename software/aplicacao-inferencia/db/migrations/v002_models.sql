-- v002_models: registro de modelos operacionais
-- Esta aplicação é SOMENTE de operação — sem treinamento.
-- Modelos são artefatos externos já treinados, importados aqui para uso.

CREATE TABLE IF NOT EXISTS models (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    file_path     TEXT    NOT NULL,     -- relativo à raiz do projeto
    format        TEXT    NOT NULL DEFAULT 'torchscript'
                          CHECK(format IN ('torchscript','pt','onnx')),
    status        TEXT    NOT NULL DEFAULT 'inactive'
                          CHECK(status IN ('active','inactive','invalid')),
    nc            INTEGER,              -- número de classes
    class_names   TEXT,                 -- JSON array ex.: ["metal","papel","plastico","vidro"]
    origin        TEXT,                 -- ex.: 'manual','imported'
    notes         TEXT,                 -- observações de compatibilidade
    registered_at TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    registered_by INTEGER REFERENCES users(id)
);
