import bcrypt
from db.database import get_connection
from db.repositories import user_repo, audit_repo
from core.auth.session import Session


class AuthError(Exception):
    pass


class AccountLocked(AuthError):
    pass


class InvalidCredentials(AuthError):
    pass


class LoginAlreadyExists(AuthError):
    pass


class MustChangePassword(Exception):
    def __init__(self, user_id: int):
        self.user_id = user_id


_MAX_ATTEMPTS = 3
_failed_attempts: dict[str, int] = {}


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify(login: str, password: str) -> Session:
    if _failed_attempts.get(login, 0) >= _MAX_ATTEMPTS:
        raise AccountLocked("Conta bloqueada por excesso de tentativas. Contacte o administrador do sistema.")

    user = user_repo.get_by_login(login)
    if user is None:
        _register_fail(login)
        raise InvalidCredentials()

    if not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        _register_fail(login, user["id"])
        raise InvalidCredentials()

    _failed_attempts.pop(login, None)
    audit_repo.record("LOGIN_OK", user_id=user["id"])

    if user["must_change_password"]:
        raise MustChangePassword(user["id"])

    return Session(user_id=user["id"], login=user["login"], role=user["role"])


def change_password(user_id: int, new_password: str):
    new_hash = hash_password(new_password)
    user_repo.update_password(user_id, new_hash, must_change_password=False)
    audit_repo.record("PASSWORD_CHANGED", user_id=user_id)


def ensure_admin_user():
    if user_repo.exists_any():
        return
    default_hash = hash_password("admin123")
    user_repo.create(
        login="admin",
        password_hash=default_hash,
        role="maintenance",
        must_change_password=True,
    )
    audit_repo.record("USER_CREATED", description="Admin padrão criado no primeiro boot")
    print("[AUTH] Usuário admin criado. Login: admin / Senha: admin123")
    print("[AUTH] Troca de senha obrigatória no primeiro acesso.")


def ensure_operator_user():
    """
    Garante que o usuário operador inicial existe.

    Criado apenas se o login 'operador' ainda não estiver cadastrado
    (ativo ou inativo). Idempotente — não duplica nem sobrescreve senha
    em boots subsequentes.

    Perfil: role='operator' (sem privilégios administrativos).
    Senha inicial: operador123 (temporária — troca obrigatória no primeiro login).
    """
    existing = get_connection().execute(
        "SELECT id FROM users WHERE login = ?", ("operador",)
    ).fetchone()
    if existing is not None:
        return
    op_hash = hash_password("operador123")
    user_repo.create(
        login="operador",
        password_hash=op_hash,
        role="operator",
        must_change_password=True,   # senha temporária → troca obrigatória no 1º login
    )
    audit_repo.record("USER_CREATED", description="Operador padrão criado no primeiro boot")
    print("[AUTH] Usuário operador criado. Login: operador / Senha: operador123")
    print("[AUTH] Troca de senha obrigatória no primeiro acesso.")


def create_user(login: str, role: str, temp_password: str, created_by: int) -> int:
    """
    Cria novo usuário com senha temporária e troca obrigatória no primeiro login.

    Raises:
        LoginAlreadyExists — se o login já existir (ativo ou inativo).
    Returns:
        id do usuário criado.
    """
    # Verifica colisão incluindo inativos (evita reativação silenciosa)
    existing = get_connection().execute(
        "SELECT id FROM users WHERE login = ?", (login,)
    ).fetchone()
    if existing is not None:
        raise LoginAlreadyExists(f"Login '{login}' já está em uso.")

    new_hash = hash_password(temp_password)
    user_repo.create(
        login=login,
        password_hash=new_hash,
        role=role,
        must_change_password=True,
        created_by=created_by,
    )
    new_user = get_connection().execute(
        "SELECT id FROM users WHERE login = ?", (login,)
    ).fetchone()
    audit_repo.record(
        "USER_CREATED",
        description=f"login={login} role={role}",
        user_id=created_by,
    )
    return new_user["id"]


def reset_password(target_user_id: int, temp_password: str, admin_user_id: int):
    """
    Reseta a senha de outro usuário para uma temporária com must_change_password=True.
    Registra PASSWORD_RESET na auditoria.
    """
    new_hash = hash_password(temp_password)
    user_repo.update_password(target_user_id, new_hash, must_change_password=True)
    audit_repo.record(
        "PASSWORD_RESET",
        description=f"target_user_id={target_user_id}",
        user_id=admin_user_id,
    )


def set_active(target_user_id: int, active: bool, admin_user_id: int):
    """
    Ativa ou desativa conta com auditoria.
    Não verifica limite mínimo de maintenance — cabe ao chamador fazer isso.
    """
    if active:
        user_repo.activate(target_user_id)
        audit_repo.record(
            "USER_ENABLED",
            description=f"target_user_id={target_user_id}",
            user_id=admin_user_id,
        )
    else:
        user_repo.deactivate(target_user_id)
        audit_repo.record(
            "USER_DISABLED",
            description=f"target_user_id={target_user_id}",
            user_id=admin_user_id,
        )


class CannotDeleteAdmin(AuthError):
    pass


def delete_user(target_user_id: int, admin_user_id: int):
    """
    Exclui permanentemente um usuário do banco.

    Regras de proteção (ambas levantam AuthError):
      · O usuário com login='admin' nunca pode ser excluído.
      · A exclusão não pode deixar zero contas maintenance ativas.
    """
    target = user_repo.get_by_id(target_user_id)
    if target is None:
        raise AuthError(f"Usuário id={target_user_id} não encontrado.")

    if target["login"] == "admin":
        raise CannotDeleteAdmin(
            "O usuário 'admin' é protegido e não pode ser excluído."
        )

    if target["role"] == "maintenance" and target["active"]:
        if user_repo.count_active_maintenance() <= 1:
            raise AuthError(
                "Não é possível excluir a única conta de administrador ativa.\n"
                "Crie ou ative outro administrador antes de excluir esta conta."
            )

    user_repo.delete(target_user_id)
    audit_repo.record(
        "USER_DELETED",
        description=f"login={target['login']} role={target['role']}",
        user_id=admin_user_id,
    )


def _register_fail(login: str, user_id: int = None):
    _failed_attempts[login] = _failed_attempts.get(login, 0) + 1
    audit_repo.record("LOGIN_FAIL", description=f"login={login}", user_id=user_id)
