"""
Teste da camada de gestão de usuários (sem Qt, sem hardware).

Cobertura:
  T1  user_repo.list_all() retorna todos (ativos e inativos)
  T2  create_user() cria com must_change_password=True
  T3  create_user() rejeita login duplicado (LoginAlreadyExists)
  T4  reset_password() força must_change_password=True e audita PASSWORD_RESET
  T5  set_active(False) desativa com auditoria USER_DISABLED
  T6  set_active(True)  reativa com auditoria USER_ENABLED
  T7  count_active_maintenance() conta corretamente
  T8  login com senha resetada -> MustChangePassword
  T9  verify() rejeita usuário inativo (InvalidCredentials)
  T10 create_user operator + create_user maintenance -> list_all tem ambos
  T11 auditoria registra USER_CREATED, PASSWORD_RESET, USER_DISABLED, USER_ENABLED
"""
import sys
import tempfile
import threading
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

# Patch do banco de dados
import db.database as _dbmod
_TMP = Path(tempfile.mkdtemp(prefix="recycleai_usrmgt_"))
_DB  = _TMP / "data" / "test.db"
_DB.parent.mkdir(parents=True, exist_ok=True)
_dbmod._DB_PATH        = _DB
_dbmod._MIGRATIONS_DIR = _ROOT / "db" / "migrations"
_dbmod._local          = threading.local()

from db.database import initialize
from db.repositories import user_repo, audit_repo
from core.auth.authenticator import (
    ensure_admin_user, verify, create_user, reset_password, set_active,
    LoginAlreadyExists, InvalidCredentials, MustChangePassword,
)

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


def run():
    print("\n=== test_user_management.py ===\n")

    initialize()
    conn = _dbmod.get_connection()

    # Setup: cria admin via ensure_admin_user (bootstrap normal)
    ensure_admin_user()
    admin_row = conn.execute("SELECT id FROM users WHERE login='admin'").fetchone()
    admin_id  = admin_row["id"]

    # ── T1: list_all retorna o admin ──────────────────────────────────────────
    all_users = user_repo.list_all()
    check("T1   list_all retorna admin",   any(u["login"] == "admin" for u in all_users))
    check("T1b  list_all inclui must_chg", "must_change_password" in all_users[0].keys())

    # ── T2: create_user operator ──────────────────────────────────────────────
    op_id = create_user("op_teste", "operator", "Senha@123", created_by=admin_id)
    op_row = user_repo.get_by_id(op_id)
    check("T2a  create_user retorna int",        isinstance(op_id, int) and op_id > 0)
    check("T2b  role=operator",                  op_row["role"] == "operator")
    check("T2c  must_change_password=1",         op_row["must_change_password"] == 1)
    check("T2d  active=1",                       op_row["active"] == 1)

    # ── T3: create_user duplicado ─────────────────────────────────────────────
    try:
        create_user("op_teste", "operator", "outra", created_by=admin_id)
        check("T3   LoginAlreadyExists levantado", False, "nao levantou excecao")
    except LoginAlreadyExists:
        check("T3   LoginAlreadyExists levantado", True)

    # ── T4: reset_password ────────────────────────────────────────────────────
    # Primeiro faz login com senha original para confirmar que funciona
    try:
        verify("op_teste", "Senha@123")
        check("T4-pre login inicial ok", False, "devia levantar MustChangePassword")
    except MustChangePassword:
        check("T4-pre MustChangePassword no login inicial", True)

    # Reseta para nova senha temporária
    reset_password(op_id, "Novo@456", admin_user_id=admin_id)
    op_row2 = user_repo.get_by_id(op_id)
    check("T4a  must_change ainda=1 apos reset", op_row2["must_change_password"] == 1)

    # Testa login com nova senha → deve dar MustChangePassword
    try:
        verify("op_teste", "Novo@456")
        check("T4b  login apos reset -> MustChangePassword", False, "nao levantou")
    except MustChangePassword:
        check("T4b  login apos reset -> MustChangePassword", True)

    # ── T5: set_active(False) ─────────────────────────────────────────────────
    set_active(op_id, active=False, admin_user_id=admin_id)
    op_row3 = user_repo.get_by_id(op_id)
    check("T5a  active=0 apos desativacao", op_row3["active"] == 0)

    # Login de usuario inativo deve falhar
    try:
        verify("op_teste", "Novo@456")
        check("T5b  usuario inativo rejeita login", False, "nao levantou excecao")
    except InvalidCredentials:
        check("T5b  usuario inativo -> InvalidCredentials", True)

    # ── T6: set_active(True) ─────────────────────────────────────────────────
    set_active(op_id, active=True, admin_user_id=admin_id)
    op_row4 = user_repo.get_by_id(op_id)
    check("T6   active=1 apos reativacao", op_row4["active"] == 1)

    # ── T7: count_active_maintenance ─────────────────────────────────────────
    cnt_before = user_repo.count_active_maintenance()
    check("T7a  count maintenance >= 1 (admin existe)", cnt_before >= 1)

    maint_id = create_user("maint_extra", "maintenance", "Manut@789", created_by=admin_id)
    cnt_after = user_repo.count_active_maintenance()
    check("T7b  count aumentou apos criar maintenance", cnt_after == cnt_before + 1)

    # ── T8: login apos reset → MustChangePassword ────────────────────────────
    # Já testado em T4b, apenas confirma que maint_extra tb exige troca
    try:
        verify("maint_extra", "Manut@789")
        check("T8   maint_extra login -> MustChangePassword", False)
    except MustChangePassword:
        check("T8   maint_extra login -> MustChangePassword", True)

    # ── T9: verify rejeita usuario inexistente ───────────────────────────────
    try:
        verify("nao_existe", "qualquer")
        check("T9   usuario inexistente -> InvalidCredentials", False)
    except InvalidCredentials:
        check("T9   usuario inexistente -> InvalidCredentials", True)

    # ── T10: list_all inclui operator e maintenance ──────────────────────────
    all_now = user_repo.list_all()
    logins  = [u["login"] for u in all_now]
    roles   = [u["role"]  for u in all_now]
    check("T10a  op_teste em list_all",      "op_teste"    in logins)
    check("T10b  maint_extra em list_all",   "maint_extra" in logins)
    check("T10c  operator role presente",    "operator"    in roles)
    check("T10d  maintenance role presente", "maintenance" in roles)

    # ── T11: auditoria registrou os eventos esperados ────────────────────────
    events = audit_repo.get_recent(limit=100)
    evt_types = [e["event_type"] for e in events]
    check("T11a  USER_CREATED em auditoria",    "USER_CREATED"    in evt_types)
    check("T11b  PASSWORD_RESET em auditoria",  "PASSWORD_RESET"  in evt_types)
    check("T11c  USER_DISABLED em auditoria",   "USER_DISABLED"   in evt_types)
    check("T11d  USER_ENABLED em auditoria",    "USER_ENABLED"    in evt_types)

    # ── Resultado ─────────────────────────────────────────────────────────────
    total = PASS + FAIL
    print(f"\n{'='*44}")
    print(f"  {PASS}/{total} PASS  |  {FAIL} FAIL")
    print(f"{'='*44}\n")
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
