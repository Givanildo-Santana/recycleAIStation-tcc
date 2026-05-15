from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass(frozen=True)
class Session:
    user_id: int
    login: str
    role: str           # 'operator' | 'maintenance'
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def is_maintenance(self) -> bool:
        return self.role == "maintenance"

    def is_operator(self) -> bool:
        return self.role == "operator"

    def is_admin(self) -> bool:
        # "admin" não é um role separado: é o usuário com login="admin"
        # e role="maintenance", criado por ensure_admin_user().
        # Outros usuários maintenance NÃO são admin.
        return self.login == "admin"

    @property
    def display_role(self) -> str:
        return {"maintenance": "Administrador", "operator": "Operador"}.get(self.role, self.role)
