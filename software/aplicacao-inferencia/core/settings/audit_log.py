"""
Wrapper de auditoria para eventos não relacionados a configurações.
Alterações de config devem passar por settings_manager.set() — não por aqui.
"""
from db.repositories import audit_repo


def record(event_type: str, description: str = None, user_id: int = None):
    audit_repo.record(event_type=event_type, description=description, user_id=user_id)


def get_recent(limit: int = 100):
    return audit_repo.get_recent(limit)
