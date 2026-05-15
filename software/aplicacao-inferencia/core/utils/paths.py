"""
Resolução de caminhos — compatível com desenvolvimento e bundle PyInstaller.

Em desenvolvimento:
  project_root()     → raiz do repositório (pai de app/, core/, db/, …)
  bundle_data_root() → igual a project_root()

Em bundle PyInstaller (--onedir, modo de produção):
  project_root()     → diretório do executável (gravável: DB, modelos importados)
  bundle_data_root() → sys._MEIPASS  (somente-leitura: migrations, modelos embutidos)

Regra de resolução de caminhos relativos armazenados no banco:
  1. Tenta bundle_data_root() / path  (modelos embutidos no bundle)
  2. Tenta project_root() / path       (modelos importados, em dev)
  3. Retorna bundle_data_root() / path  como fallback (para erros legíveis)
"""
from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """
    Raiz gravável do projeto.

    · Desenvolvimento : diretório raiz do repositório (três níveis acima deste arquivo).
    · Bundle frozen   : diretório do executável (.exe), que é gravável pelo operador.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent


def bundle_data_root() -> Path:
    """
    Raiz dos dados somente-leitura embutidos no bundle.

    · Desenvolvimento : igual a project_root().
    · Bundle frozen   : sys._MEIPASS  (_internal/ no PyInstaller 6.x --onedir).
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return project_root()


def resolve_model_path(rel_path: str) -> Path:
    """
    Resolve um caminho de modelo relativo para absoluto.

    Tenta, em ordem:
      1. bundle_data_root() / rel_path  — modelos embutidos no bundle
      2. project_root() / rel_path      — modelos importados/gravados pelo operador
    Se nenhum existir, retorna bundle_data_root() / rel_path (erro legível downstream).
    """
    path = Path(rel_path)
    if path.is_absolute():
        return path
    for base in (bundle_data_root(), project_root()):
        candidate = base / path
        if candidate.exists():
            return candidate
    return bundle_data_root() / path
