"""
Registro de modelos operacionais.

Esta aplicação é SOMENTE de operação. Modelos são artefatos externos
já treinados. Não há treinamento, dataset management nem hyperparameter
tuning neste módulo ou em qualquer parte desta aplicação.

Responsabilidades:
  - Registrar modelos no banco (nome, caminho, formato, classes)
  - Selecionar modelo ativo
  - Validar compatibilidade antes de ativar
  - Semear o modelo padrão (best_ts.pt) no primeiro boot

Fluxo de importação via pacote completo (padrão atual):
  1. Operador seleciona pasta <nome>_package/ produzida pela aplicação de treinamento
  2. _ImportWorker copia o pacote para runtime_inferencia/modelos_importados/<pkg>/
  3. register_package() lê o manifest.json e registra no banco com classes e deploy_file
  4. Operador ativa via set_active()
  5. pipeline de inferência usa get_active_path() → resolve file_path do manifest

Fluxo legado (arquivo único .pt/.onnx):
  1. register() ainda funciona para modelos importados antes desta versão
"""
from __future__ import annotations

import json
from pathlib import Path

from core.detection.model_validator import validate, ValidationResult
from core.detection.package_validator import validate_package, PackageValidationResult
from db.repositories import model_repo, audit_repo

from core.utils.paths import project_root, bundle_data_root, resolve_model_path
_ROOT      = bundle_data_root()   # dados embutidos somente-leitura (data.yaml, modelos bundled)
_PROJ_ROOT = project_root()       # diretório gravável (onde ficam os modelos importados)


# ---------------------------------------------------------------------------
# Semear modelo padrão (chamado no bootstrap)
# ---------------------------------------------------------------------------

def seed_default(user_id: int | None = None):
    """
    Registra e ativa modelos_treinados/modelo_base como modelo inicial
    no primeiro boot. Idempotente: não faz nada se já houver qualquer
    modelo registrado no banco — preserva qualquer escolha posterior do admin.

    Fluxo de decisão:
      1. model_repo.exists_any() → True  → retorna imediatamente (preserva admin)
      2. Localiza modelos_treinados/modelo_base em bundle_data_root() ou project_root()
      3. validate_package() superficial (sem torch.jit.load) para extrair metadados
      4. Registra e ativa usando os metadados do manifest.json
      5. Se pasta ausente ou inválida → loga aviso e retorna sem crash
    """
    if model_repo.exists_any():
        # Modelos já registrados — o admin pode ter escolhido um diferente;
        # não interferir em nenhum cenário de boot subsequente.
        return

    # Localiza modelos_treinados/modelo_base: tenta bundle_data_root() primeiro
    # (correto no .exe), depois project_root() (correto em dev).
    pkg_dir: Path | None = None
    for base in (_ROOT, _PROJ_ROOT):
        candidate = base / "modelos_treinados" / "modelo_base"
        if candidate.is_dir():
            pkg_dir = candidate
            break

    if pkg_dir is None:
        print("[MODEL] AVISO: modelos_treinados/modelo_base não encontrado em nenhum")
        print("[MODEL]   caminho raiz. Seed ignorado.")
        print("[MODEL]   Importe um modelo manualmente pelo painel de administração.")
        return

    # Validação superficial: lê manifest, verifica estrutura, NÃO carrega o modelo
    # com torch.jit.load para não atrasar o boot.
    result = validate_package(pkg_dir, deep=False)
    if not result.ok:
        print(f"[MODEL] AVISO: modelo_base em '{pkg_dir}' é inválido — {result.detail}")
        print("[MODEL]   Importe um modelo manualmente pelo painel de administração.")
        return

    # Caminho relativo armazenado no banco.
    # resolve_model_path() resolve para absoluto em runtime, tentando
    # bundle_data_root() primeiro (correto no .exe) e project_root() em seguida.
    file_path_rel = f"modelos_treinados/modelo_base/{result.deploy_file}"

    model_id = model_repo.register(
        name=result.pkg_name,
        file_path=file_path_rel,
        fmt="torchscript",
        nc=result.nc,
        class_names=result.class_names,
        origin="seed",
        notes=result.detail,
        registered_by=user_id,
        package_dir="modelos_treinados/modelo_base",
    )

    model_repo.set_active(model_id)
    audit_repo.record(
        "MODEL_ACTIVATED",
        description=(
            f"Modelo base registrado e ativado no primeiro boot: {file_path_rel} "
            f"({result.nc} classes, {result.size_mb:.1f} MB)"
        ),
        user_id=user_id,
    )
    print(f"[MODEL] Modelo base registrado e ativado (id={model_id}): {file_path_rel}")
    print(f"[MODEL]   Nome: {result.pkg_name}")
    print(f"[MODEL]   Classes ({result.nc}): {result.class_names}")
    print(f"[MODEL]   Tamanho: {result.size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Registro de novo modelo externo
# ---------------------------------------------------------------------------

def register(
    name: str,
    file_path: str,
    fmt: str = "torchscript",
    nc: int = None,
    class_names: list[str] = None,
    origin: str = "imported",
    notes: str = None,
    user_id: int = None,
) -> tuple[int, ValidationResult]:
    """
    Valida e registra um modelo externo já treinado.
    Retorna (model_id, ValidationResult).
    Não ativa automaticamente — operador escolhe via set_active().
    """
    result = validate(file_path)
    status = "inactive" if result.ok else "invalid"
    combined_notes = f"{notes}\n{result.detail}" if notes else result.detail

    model_id = model_repo.register(
        name=name,
        file_path=str(file_path),
        fmt=fmt,
        nc=nc,
        class_names=class_names,
        origin=origin,
        notes=combined_notes,
        registered_by=user_id,
    )
    if not result.ok:
        model_repo.update_status(model_id, "invalid", notes=result.detail)

    audit_repo.record(
        "MODEL_REGISTERED",
        description=f"Modelo '{name}' registrado (status={status}): {file_path}",
        user_id=user_id,
    )
    return model_id, result


# ---------------------------------------------------------------------------
# Registro de pacote completo (fluxo padrão a partir da v1.1)
# ---------------------------------------------------------------------------

def register_package(
    pkg_dir: Path,
    name: str = None,
    user_id: int = None,
) -> tuple[int, PackageValidationResult]:
    """
    Registra um pacote RecycleAI completo já copiado para dentro da aplicação.

    Lê o manifest.json do pacote para extrair:
      - deploy_file  → file_path relativo para o modelo ativo
      - nc           → número de classes
      - class_names  → nomes das classes na ordem correta do manifest

    Args:
        pkg_dir:  Path absoluto para a pasta do pacote (deve existir)
        name:     nome de exibição; usa manifest.name se omitido
        user_id:  ID do operador que está registrando

    Returns:
        (model_id, PackageValidationResult) — result.ok=True se inserido com sucesso.

    Lança:
        ValueError se o pacote não for válido (validação superficial).
    """
    # Revalida o pacote já copiado (superficial — deep foi feito no worker)
    result = validate_package(pkg_dir, deep=False)
    if not result.ok:
        raise ValueError(f"Pacote inválido após cópia: {result.detail}")

    display_name = (name or "").strip() or result.pkg_name

    # Calcula caminhos relativos ao diretório gravável do projeto
    try:
        file_path_rel = (pkg_dir / result.deploy_file).resolve().relative_to(
            _PROJ_ROOT
        ).as_posix()
        pkg_dir_rel = pkg_dir.resolve().relative_to(_PROJ_ROOT).as_posix()
    except ValueError:
        # Pacote fora da árvore do projeto — salva absoluto como fallback
        file_path_rel = str(pkg_dir / result.deploy_file)
        pkg_dir_rel   = str(pkg_dir)

    model_id = model_repo.register(
        name         = display_name,
        file_path    = file_path_rel,
        fmt          = "torchscript",
        nc           = result.nc,
        class_names  = result.class_names,
        origin       = "imported",
        notes        = result.detail,
        registered_by = user_id,
        package_dir  = pkg_dir_rel,
    )

    audit_repo.record(
        "MODEL_REGISTERED",
        description=(
            f"Pacote '{display_name}' registrado "
            f"({result.nc} classes, {result.size_mb:.1f} MB): {pkg_dir_rel}"
        ),
        user_id=user_id,
    )
    return model_id, result


# ---------------------------------------------------------------------------
# Ativação
# ---------------------------------------------------------------------------

def set_active(model_id: int, user_id: int = None) -> ValidationResult:
    """
    Valida o modelo e o ativa se compatível.
    Desativa todos os outros.
    """
    row = model_repo.get_by_id(model_id)
    if row is None:
        raise ValueError(f"Modelo id={model_id} não encontrado")

    result = validate(row["file_path"])
    if not result.ok:
        model_repo.update_status(model_id, "invalid", notes=result.detail)
        raise ValueError(f"Modelo inválido: {result.detail}")

    model_repo.set_active(model_id)
    audit_repo.record(
        "MODEL_ACTIVATED",
        description=f"Modelo ativado: {row['name']} ({row['file_path']})",
        user_id=user_id,
    )
    return result


# ---------------------------------------------------------------------------
# Consulta do modelo ativo
# ---------------------------------------------------------------------------

def get_active_path() -> Path | None:
    """Retorna o caminho absoluto do modelo ativo, ou None se não houver."""
    row = model_repo.get_active()
    if row is None:
        return None
    # resolve_model_path tenta bundle_data_root() e project_root() em ordem
    return resolve_model_path(row["file_path"])


def get_active_classes() -> list[str]:
    """Retorna a lista de classes do modelo ativo."""
    row = model_repo.get_active()
    if row is None or not row["class_names"]:
        return []
    return json.loads(row["class_names"])


def list_models() -> list[dict]:
    """Lista todos os modelos registrados com campos principais."""
    rows = model_repo.list_all()
    return [
        {
            "id":           r["id"],
            "name":         r["name"],
            "file_path":    r["file_path"],
            "format":       r["format"],
            "status":       r["status"],
            "nc":           r["nc"],
            "class_names":  json.loads(r["class_names"]) if r["class_names"] else [],
            "origin":       r["origin"],
            "notes":        r["notes"],
            "registered_at": r["registered_at"],
            # package_dir: None para modelos legados, posix-relativo para pacotes
            "package_dir":  r["package_dir"] if "package_dir" in r.keys() else None,
        }
        for r in rows
    ]


