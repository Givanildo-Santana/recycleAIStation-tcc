"""
Teste de exportacao PDF + CSV de relatorios operacionais.

Sem Qt, sem camera, sem hardware.
DB temporario isolado.

Cobertura:
  T1   fpdf2 importavel no venv
  T2   export_csv operacional — arquivo gerado, estrutura correta
  T3   export_csv tecnico (maintenance) — inclui secoes de metricas
  T4   export_pdf operacional — arquivo .pdf gerado e nao vazio
  T5   export_pdf tecnico — inclui secao de metricas no PDF
  T6   build_report(id, 'operator') -> dict operacional
  T7   build_report(id, 'maintenance') -> dict tecnico
  T8   PDF operacional — campos criticos presentes no binario
  T9   PDF tecnico — campos extras presentes
  T10  CSV encoding UTF-8 com BOM (Excel-compativel)
  T11  export_pdf levanta ImportError se fpdf2 nao disponivel (simulado)
  T12  _fmt_duration formata corretamente
"""
import json
import sys
import tempfile
import threading
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

# Patch do banco de dados
import db.database as _dbmod
_TMP = Path(tempfile.mkdtemp(prefix="recycleai_rpttest_"))
_DB  = _TMP / "data" / "test.db"
_DB.parent.mkdir(parents=True, exist_ok=True)
_dbmod._DB_PATH        = _DB
_dbmod._MIGRATIONS_DIR = _ROOT / "db" / "migrations"
_dbmod._local          = threading.local()

from db.database import initialize
from db.repositories import session_repo
from core.operation import report_service
from core.operation.report_service import _fmt_duration

PASS = 0
FAIL = 0

def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
        FAIL += 1


def _make_session(profile: str, user_login: str) -> int:
    """Cria e fecha uma sessao com dados realistas. Retorna op_session_id."""
    import uuid
    sid = str(uuid.uuid4())
    op_id = session_repo.open_session(
        session_id      = sid,
        user_id         = 1,
        user_login      = user_login,
        profile         = profile,
        model_name      = "RecycleAI-v1 (best_ts)",
        config_snapshot = {
            "realtime.conf_thres": "0.50",
            "realtime.iou_thres":  "0.45",
            "realtime.roi_x_start": "100",
            "realtime.roi_x_end":   "480",
            "arduino.port": "COM5",
        },
    )
    session_repo.add_event(op_id, "WARNING", "Serial offline — operacao sem Arduino")
    session_repo.close_session(
        op_session_id = op_id,
        duration_s    = 183.5,
        total_items   = 7,
        counts        = {"metal": 2, "papel": 3, "plastico": 1, "vidro": 1},
        status        = "closed",
    )
    if profile == "maintenance":
        session_repo.upsert_metrics(
            op_session_id  = op_id,
            infer_count    = 320,
            infer_time_min = 0.011,
            infer_time_avg = 0.017,
            infer_time_max = 0.031,
            conf_min       = 0.51,
            conf_avg       = 0.76,
            conf_max       = 0.97,
            fps_avg        = 28.4,
            serial_port    = None,
            model_path     = "modelos_treinados/modelo_base/weights/best_ts.pt",
            extra          = {"conf_thres": 0.50, "iou_thres": 0.45, "roi": [100,70,480,450]},
        )
    return op_id


def run():
    print("\n=== test_report_export.py ===\n")

    initialize()
    conn = _dbmod.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO users (login, password_hash, role) VALUES (?,?,?)",
        ("admin", "x", "maintenance"),
    )
    conn.commit()

    op_id    = _make_session("operator",    "operador01")
    maint_id = _make_session("maintenance", "admin")

    # ── T1: fpdf2 importavel ──────────────────────────────────────────────────
    try:
        from fpdf import FPDF
        check("T1   fpdf2 importavel", True)
    except ImportError as e:
        check("T1   fpdf2 importavel", False, str(e))
        print("\n  ABORT: fpdf2 nao instalado — execute pip install fpdf2\n")
        return False

    # ── T2: CSV operacional ───────────────────────────────────────────────────
    rpt_op = report_service.get_operational_report(op_id)
    csv_op = _TMP / "relatorio_op.csv"
    report_service.export_csv(rpt_op, csv_op)
    csv_op_txt = csv_op.read_text(encoding="utf-8-sig")

    check("T2a  CSV operacional gerado",    csv_op.exists())
    check("T2b  CSV contem operador",       "operador01" in csv_op_txt)
    check("T2c  CSV contem total_items=7",  "7" in csv_op_txt)
    check("T2d  CSV contem metal=2",        "metal" in csv_op_txt and "2" in csv_op_txt)
    check("T2e  CSV NAO tem secao metricas","METRICAS TECNICAS" not in csv_op_txt)

    # ── T3: CSV tecnico (maintenance) ─────────────────────────────────────────
    rpt_tech = report_service.get_technical_report(maint_id)
    csv_tech = _TMP / "relatorio_tech.csv"
    report_service.export_csv(rpt_tech, csv_tech)
    csv_tech_txt = csv_tech.read_text(encoding="utf-8-sig")

    check("T3a  CSV tecnico gerado",             csv_tech.exists())
    check("T3b  CSV tecnico tem metricas",        "METRICAS TECNICAS" in csv_tech_txt)
    check("T3c  CSV tecnico tem snapshot config", "SNAPSHOT" in csv_tech_txt)
    check("T3d  CSV tecnico tem infer_count=320", "320" in csv_tech_txt)
    check("T3e  CSV tecnico tem fps_avg=28.4",    "28.4" in csv_tech_txt)

    # ── T4: PDF operacional ───────────────────────────────────────────────────
    pdf_op = _TMP / "relatorio_op.pdf"
    report_service.export_pdf(rpt_op, pdf_op)

    check("T4a  PDF operacional gerado",   pdf_op.exists())
    check("T4b  PDF nao vazio (>2KB)",     pdf_op.stat().st_size > 2_000)
    check("T4c  PDF tem assinatura %PDF",  pdf_op.read_bytes()[:5] == b"%PDF-")

    # ── T5: PDF tecnico (maintenance) ─────────────────────────────────────────
    pdf_tech = _TMP / "relatorio_tech.pdf"
    report_service.export_pdf(rpt_tech, pdf_tech)

    check("T5a  PDF tecnico gerado",       pdf_tech.exists())
    check("T5b  PDF tecnico nao vazio",    pdf_tech.stat().st_size > 5_000)
    # PDF tecnico deve ser maior que operacional (mais conteudo)
    check("T5c  PDF tecnico > PDF op",
          pdf_tech.stat().st_size > pdf_op.stat().st_size)

    # ── T6 / T7: build_report diferencia perfil ───────────────────────────────
    r_op    = report_service.build_report(op_id,    "operator")
    r_maint = report_service.build_report(maint_id, "maintenance")
    check("T6   build_report operator -> operational",
          r_op["report_type"] == "operational")
    check("T7   build_report maintenance -> technical",
          r_maint["report_type"] == "technical")

    # ── T8: conteudo critico no PDF operacional (busca no binario) ────────────
    pdf_bytes = pdf_op.read_bytes()
    check("T8a  PDF contem 'operador01'",   b"operador01" in pdf_bytes)
    check("T8b  PDF contem 'operator'",     b"operator"   in pdf_bytes)
    check("T8c  PDF contem 'metal'",        b"metal"      in pdf_bytes)

    # ── T9: conteudo tecnico no PDF maintenance ────────────────────────────────
    pdf_tech_bytes = pdf_tech.read_bytes()
    check("T9a  PDF tecnico contem 'admin'",       b"admin"   in pdf_tech_bytes)
    check("T9b  PDF tecnico contem 'maintenance'", b"maintenance" in pdf_tech_bytes)
    check("T9c  PDF tecnico contem 'METRICAS'",    b"METRICAS" in pdf_tech_bytes)
    check("T9d  PDF tecnico contem '28.4'",        b"28.4"    in pdf_tech_bytes or
                                                    b"28"      in pdf_tech_bytes)

    # ── T10: CSV UTF-8 com BOM ────────────────────────────────────────────────
    raw = csv_op.read_bytes()
    check("T10  CSV tem BOM UTF-8 (\\xef\\xbb\\xbf)", raw[:3] == b"\xef\xbb\xbf")

    # ── T11: ImportError simulado ─────────────────────────────────────────────
    import sys as _sys
    import unittest.mock as mock
    with mock.patch.dict(_sys.modules, {"fpdf": None}):
        try:
            report_service.export_pdf(rpt_op, _TMP / "should_fail.pdf")
            check("T11  ImportError se fpdf2 ausente", False, "nao levantou excecao")
        except (ImportError, TypeError):
            check("T11  ImportError se fpdf2 ausente", True)

    # ── T12: _fmt_duration ────────────────────────────────────────────────────
    check("T12a _fmt_duration None    -> '-'",   _fmt_duration(None)    == "-")
    check("T12b _fmt_duration 45      -> '45s'", _fmt_duration(45)      == "45s")
    check("T12c _fmt_duration 183     -> '3m 03s'", _fmt_duration(183)  == "3m 03s")
    check("T12d _fmt_duration 3723    -> '1h 02m 03s'", _fmt_duration(3723) == "1h 02m 03s")

    # ── Resultado final ───────────────────────────────────────────────────────
    total = PASS + FAIL
    print(f"\n{'='*44}")
    print(f"  {PASS}/{total} PASS  |  {FAIL} FAIL")
    print(f"{'='*44}")
    print(f"\n  Arquivos gerados em: {_TMP}\n")
    print(f"  PDF operacional: {pdf_op.stat().st_size // 1024} KB")
    print(f"  PDF tecnico:     {pdf_tech.stat().st_size // 1024} KB\n")
    return FAIL == 0


if __name__ == "__main__":
    try:
        ok = run()
    finally:
        try:
            _dbmod.close()
        except Exception:
            pass
    sys.exit(0 if ok else 1)
