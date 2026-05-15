"""
Executado uma vez após a instalação para preparar o ambiente de runtime.
Idempotente: seguro para rodar múltiplas vezes.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from db.database import initialize as db_init
from core.auth.authenticator import ensure_admin_user
from core.settings.settings_manager import bootstrap_defaults
from core.detection.model_registry import seed_default as seed_model


def main():
    print("[POST-INSTALL] Criando banco de dados e aplicando migrations...")
    db_init()

    print("[POST-INSTALL] Garantindo usuário admin inicial...")
    ensure_admin_user()

    print("[POST-INSTALL] Gravando configurações padrão...")
    bootstrap_defaults()

    print("[POST-INSTALL] Registrando modelo operacional padrão...")
    seed_model()

    print("[POST-INSTALL] Instalação base concluída com sucesso.")
    print("[POST-INSTALL] Login inicial: admin / Senha: admin123")
    print("[POST-INSTALL] Troca de senha obrigatória no primeiro acesso.")


if __name__ == "__main__":
    main()
