"""
Teste de integração da camada de sessão operacional.

Testa sem Qt, sem câmera e sem hardware.
Usa um DB temporário isolado para não tocar no banco de desenvolvimento.

Cobertura:
  T1  Migração v003 aplica as 3 novas tabelas
  T2  session_repo.open_session() cria registro com status='open'
  T3  session_repo.close_session() atualiza totais e status='closed'
  T4  session_repo.get_by_id() retorna dados corretos
  T5  session_repo.add_event() / get_events() funcionam
  T6  session_repo.upsert_metrics() insere e retorna métricas
  T7  session_manager.open_session() — integração com modelo e configs
  T8  session_manager.close_session() — duração calculada corretamente
  T9  session_manager.record_metrics() — armazenamento via upsert
  T10 report_service.get_operational_report() — estrutura completa
  T11 report_service.get_technical_report() — superset com métricas
  T12 report_service.export_csv() — arquivo gerado, campos presentes
  T13 Sessão maintenance vs operator — diferenciação no banco
  T14 Sessão sem encerramento formal — status permanece 'open'
"""
import json
import os
import shutil
import sys
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path

# ── Adicionar raiz do projeto ao sys.path ─────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

# ── Redirecionar DB e migrations para temporários ─────────────────────────────
_TMP = Path(tempfile.mkdtemp(prefix="recycleai_test_"))
_DB  = _TMP / "data" / "test.db"
_DB.parent.mkdir(parents=True, exist_ok=True)

# Patch antes de qualquer import do projeto
import db.database as _dbmod
_dbmod._DB_PATH        = _DB
_dbmod._MIGRATIONS_DIR = _ROOT / "db" / "migrations"
_dbmod._local          = threading.local()

# ── Imports do projeto (após patch) ──────────────────────────────────────────
from db.database import initialize
from db.repositories import session_repo, audit_repo
from core.auth.session import Session
from core.operation import session_manager, report_service

# ─────────────────────────────────────────────────────────────────────────────
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
        FAIL += 1


# ─────────────────────────────────────────────────────────────────────────────
def run():
    print("\n=== test_session_layer.py ===\n")

    # ── Setup: inicializar banco e dados mínimos ──────────────────────────────
    initialize()
    conn = _dbmod.get_connection()

    # Criar usuário de teste mínimo
    conn.execute(
        "INSERT OR IGNORE INTO users (login, password_hash, role) VALUES (?,?,?)",
        ("test_op", "x", "operator"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO users (login, password_hash, role) VALUES (?,?,?)",
        ("test_maint", "x", "maintenance"),
    )
    conn.commit()

    user_op_id = conn.execute(
        "SELECT id FROM users WHERE login='test_op'"
    ).fetchone()["id"]
    user_maint_id = conn.execute(
        "SELECT id FROM users WHERE login='test_maint'"
    ).fetchone()["id"]

    # ── T1: Migração v003 criou as 3 tabelas ─────────────────────────────────
    tables = {
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    check("T1a  op_sessions criada",       "op_sessions"     in tables)
    check("T1b  session_events criada",    "session_events"  in tables)
    check("T1c  session_metrics criada",   "session_metrics" in tables)

    # ── T2: open_session cria registro com status='open' ─────────────────────
    import uuid
    sid = str(uuid.uuid4())
    op_id = session_repo.open_session(
        session_id      = sid,
        user_id         = user_op_id,
        user_login      = "test_op",
        profile         = "operator",
        config_snapshot = {"realtime.conf_thres": "0.5"},
    )
    row = session_repo.get_by_id(op_id)
    check("T2a  open_session retorna id int",  isinstance(op_id, int) and op_id > 0)
    check("T2b  status='open'",                row["status"] == "open")
    check("T2c  session_id salvo",             row["session_id"] == sid)
    check("T2d  profile='operator'",           row["profile"] == "operator")
    check("T2e  config_snapshot JSON",
          json.loads(row["config_snapshot"])["realtime.conf_thres"] == "0.5")

    # ── T3: close_session atualiza totais e status ────────────────────────────
    counts = {"metal": 3, "papel": 2, "plastico": 1}
    session_repo.close_session(
        op_session_id = op_id,
        duration_s    = 45.7,
        total_items   = sum(counts.values()),
        counts        = counts,
        status        = "closed",
    )
    row = session_repo.get_by_id(op_id)
    check("T3a  status='closed'",             row["status"] == "closed")
    check("T3b  total_items=6",               row["total_items"] == 6)
    check("T3c  duration_s=45.7",             abs(row["duration_s"] - 45.7) < 0.01)
    check("T3d  counts_json correto",
          json.loads(row["counts_json"])["metal"] == 3)
    check("T3e  ended_at preenchido",         row["ended_at"] is not None)

    # ── T4: list_recent ───────────────────────────────────────────────────────
    rows = session_repo.list_recent(10)
    check("T4   list_recent retorna lista",   len(rows) >= 1)

    # ── T5: session_events ────────────────────────────────────────────────────
    session_repo.add_event(op_id, "ERROR",   "Camera perdeu frame")
    session_repo.add_event(op_id, "WARNING", "Serial offline")
    evs = session_repo.get_events(op_id)
    check("T5a  2 eventos inseridos",         len(evs) == 2)
    check("T5b  primeiro evento=ERROR",       evs[0]["event_type"] == "ERROR")

    # ── T6: session_metrics upsert ────────────────────────────────────────────
    maint_sid = str(uuid.uuid4())
    maint_op_id = session_repo.open_session(
        session_id  = maint_sid,
        user_id     = user_maint_id,
        user_login  = "test_maint",
        profile     = "maintenance",
    )
    session_repo.upsert_metrics(
        op_session_id  = maint_op_id,
        infer_count    = 150,
        infer_time_min = 0.012,
        infer_time_avg = 0.018,
        infer_time_max = 0.034,
        conf_min       = 0.52,
        conf_avg       = 0.74,
        conf_max       = 0.96,
        fps_avg        = 28.5,
        serial_port    = "COM5",
        model_path     = "modelos_treinados/modelo_base/weights/best_ts.pt",
        extra          = {"roi": [100, 70, 480, 450]},
    )
    m = session_repo.get_metrics(maint_op_id)
    check("T6a  metrics salvas",              m is not None)
    check("T6b  infer_count=150",             m["infer_count"] == 150)
    check("T6c  fps_avg=28.5",               abs(m["fps_avg"] - 28.5) < 0.01)
    check("T6d  extra_json decodifica",
          json.loads(m["extra_json"])["roi"] == [100, 70, 480, 450])

    # Upsert (segunda chamada — deve atualizar, não duplicar)
    session_repo.upsert_metrics(op_session_id=maint_op_id, infer_count=200)
    m2 = session_repo.get_metrics(maint_op_id)
    check("T6e  upsert atualiza (sem duplicar)", m2["infer_count"] == 200)

    # ── T7: session_manager.open_session() ───────────────────────────────────
    auth_op = Session(user_id=user_op_id, login="test_op", role="operator")
    sm_op_id, sm_started = session_manager.open_session(auth_op)
    sm_row = session_repo.get_by_id(sm_op_id)
    check("T7a  session_manager retorna (int, datetime)",
          isinstance(sm_op_id, int) and isinstance(sm_started, datetime))
    check("T7b  status='open'",               sm_row["status"] == "open")
    check("T7c  user_login='test_op'",        sm_row["user_login"] == "test_op")
    check("T7d  config_snapshot presente",    sm_row["config_snapshot"] is not None)

    # ── T8: session_manager.close_session() — duração calculada ──────────────
    started_fake = datetime.now() - timedelta(seconds=30)
    session_manager.close_session(
        op_session_id = sm_op_id,
        started_at    = started_fake,
        counts        = {"vidro": 1, "metal": 2},
        auth_session  = auth_op,
    )
    sm_row2 = session_repo.get_by_id(sm_op_id)
    check("T8a  status='closed'",             sm_row2["status"] == "closed")
    check("T8b  duracao >= 30s",              sm_row2["duration_s"] >= 30.0)
    check("T8c  total_items=3",               sm_row2["total_items"] == 3)

    # ── T9: session_manager.record_metrics() ─────────────────────────────────
    auth_maint = Session(user_id=user_maint_id, login="test_maint", role="maintenance")
    sm_m_id, _ = session_manager.open_session(auth_maint)
    session_manager.record_metrics(sm_m_id, {
        "infer_count":    80,
        "infer_time_min": 0.010,
        "infer_time_avg": 0.015,
        "infer_time_max": 0.025,
        "conf_min": 0.55, "conf_avg": 0.72, "conf_max": 0.93,
        "fps_avg": 30.1,
        "serial_port": "COM3",
        "model_path": "modelos_treinados/modelo_base/weights/best_ts.pt",
        "extra": {"conf_thres": 0.5},
    })
    sm_metrics = session_repo.get_metrics(sm_m_id)
    check("T9a  metrics registradas via session_manager",  sm_metrics is not None)
    check("T9b  infer_count=80",              sm_metrics["infer_count"] == 80)
    check("T9c  serial_port='COM3'",          sm_metrics["serial_port"] == "COM3")

    # ── T10: report_service.get_operational_report() ─────────────────────────
    session_repo.close_session(sm_m_id, 25.0, 5, {"plastico": 5}, "closed")
    op_report = report_service.get_operational_report(sm_op_id)
    check("T10a  report_type='operational'",  op_report["report_type"] == "operational")
    check("T10b  operator='test_op'",         op_report["operator"] == "test_op")
    check("T10c  total_items=3",              op_report["total_items"] == 3)
    check("T10d  counts_by_class presente",   "metal" in op_report["counts_by_class"])
    check("T10e  duration_fmt presente",
          isinstance(op_report["duration_fmt"], str) and len(op_report["duration_fmt"]) > 0)
    check("T10f  generated_at presente",      op_report.get("generated_at") is not None)

    # ── T11: report_service.get_technical_report() ───────────────────────────
    tech_report = report_service.get_technical_report(sm_m_id)
    check("T11a  report_type='technical'",    tech_report["report_type"] == "technical")
    check("T11b  metrics presente",           tech_report["metrics"] is not None)
    check("T11c  infer_count=80",             tech_report["metrics"]["infer_count"] == 80)
    check("T11d  config_snapshot presente",   isinstance(tech_report["config_snapshot"], dict))
    check("T11e  fps_avg presente",           tech_report["metrics"]["fps_avg"] is not None)

    # ── T12: report_service.export_csv() ─────────────────────────────────────
    csv_op   = _TMP / "report_op.csv"
    csv_tech = _TMP / "report_tech.csv"
    report_service.export_csv(op_report,   csv_op)
    report_service.export_csv(tech_report, csv_tech)

    csv_op_text   = csv_op.read_text(encoding="utf-8-sig")
    csv_tech_text = csv_tech.read_text(encoding="utf-8-sig")

    check("T12a  CSV operacional gerado",       csv_op.exists())
    check("T12b  CSV técnico gerado",           csv_tech.exists())
    check("T12c  CSV contém operador",          "test_op" in csv_op_text)
    check("T12d  CSV contém total_items",       "3" in csv_op_text)
    check("T12e  CSV técnico contém metricas",
          "METRICAS TECNICAS" in csv_tech_text)
    check("T12f  CSV técnico contém snapshot",
          "SNAPSHOT" in csv_tech_text)

    # ── T13: Diferenciação operator vs maintenance no banco ───────────────────
    all_rows = session_repo.list_recent(50)
    profiles = {r["profile"] for r in all_rows}
    check("T13a  perfil 'operator' registrado",    "operator"    in profiles)
    check("T13b  perfil 'maintenance' registrado", "maintenance" in profiles)

    # Somente sessões maintenance têm métricas
    maint_rows = [r for r in all_rows if r["profile"] == "maintenance"]
    maint_with_metrics = [
        r for r in maint_rows
        if session_repo.get_metrics(r["id"]) is not None
    ]
    check("T13c  maintenance tem metricas associadas", len(maint_with_metrics) >= 1)

    # ── T14: Sessão sem encerramento formal — status permanece 'open' ─────────
    open_sid = str(uuid.uuid4())
    open_id  = session_repo.open_session(
        session_id="abandoned-" + open_sid[:8],
        user_id=user_op_id, user_login="test_op", profile="operator",
    )
    open_row = session_repo.get_by_id(open_id)
    check("T14a  sessao nao encerrada tem status='open'", open_row["status"] == "open")
    check("T14b  ended_at = None",              open_row["ended_at"] is None)

    # ── Resultado final ───────────────────────────────────────────────────────
    total = PASS + FAIL
    print(f"\n{'='*44}")
    print(f"  {PASS}/{total} PASS  |  {FAIL} FAIL")
    print(f"{'='*44}")
    if FAIL:
        print(f"\n  Arquivos de teste em: {_TMP}\n")
    else:
        print(f"\n  CSV gerados em: {_TMP}\n")

    return FAIL == 0


if __name__ == "__main__":
    try:
        ok = run()
    finally:
        # Limpar banco de teste
        try:
            _dbmod.close()
        except Exception:
            pass
    sys.exit(0 if ok else 1)
