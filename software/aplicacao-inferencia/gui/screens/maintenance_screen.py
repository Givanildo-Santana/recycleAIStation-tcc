"""
Tela de Manutenção — perfil `maintenance`.

Responsabilidades:
  · Exibir usuário logado, perfil e sessão
  · Importar modelo externo pronto (file picker + cópia + validação profunda)
  · Exibir e ativar modelos registrados no banco
  · Editar configurações do sistema com validação, persistência e auditoria
  · Exibir log de auditoria recente
  · Navegação segura: logout volta ao login

Escopo operacional — definição explícita:
  Esta aplicação NÃO treina modelos.
  Esta aplicação NÃO gerencia datasets de treino.
  Esta aplicação NÃO faz hyperparameter tuning.
  Esta aplicação recebe pastas de pacote RecycleAI (<nome>_package/ com manifest.json)
  produzidas pela aplicação de treinamento e as registra para uso no pipeline de inferência.

Integração:
  · model_registry       — lista, registro e ativação de modelos
  · model_validator      — validação nível 1 (rápida) e nível 2 (torch.jit.load)
  · settings_manager     — leitura e escrita de configurações
  · audit_repo           — leitura + MAINTENANCE_ENTER/EXIT/MODEL_IMPORT_*/CONFIG_CHANGE/DIAG_MANUAL_*
  · config_repo.history()— histórico por parâmetro

Funcionalidades previstas para versões futuras:
  · Importação em lote de modelos
  · Restauração de padrões de fábrica via interface
  · Validação cruzada entre campos de ROI (x_start < x_end)
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QScrollArea, QSplitter,
    QLineEdit, QFrame, QFileDialog,
    QDialog, QDialogButtonBox, QComboBox, QFormLayout, QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont

from core.auth.session import Session

from core.utils.paths import project_root
_PROJECT_ROOT = project_root()
_IMPORT_DIR   = _PROJECT_ROOT / "runtime_inferencia" / "modelos_importados"


# ─────────────────────────────────────────────────────────────────────────────
# Especificação dos parâmetros editáveis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ParamSpec:
    key: str
    label: str
    group: str
    dtype: str          # 'float' | 'int' | 'str'
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    hint: str = ""


_PARAM_SPECS: list[_ParamSpec] = [
    _ParamSpec("realtime.conf_thres",       "Confiança mínima",    "Inferência",  "float", 0.1,  1.0,    "0.10 – 1.00"),
    _ParamSpec("realtime.iou_thres",        "IoU NMS",             "Inferência",  "float", 0.1,  0.9,    "0.10 – 0.90"),
    _ParamSpec("realtime.device",           "Dispositivo",         "Inferência",  "str",                  hint="'cpu' ou 'cuda'"),
    _ParamSpec("realtime.source",           "Fonte de câmera",     "Inferência",  "str",                  hint="índice (0,1) ou URL RTSP"),
    _ParamSpec("realtime.roi_x_start",      "ROI X início",        "ROI",         "int",   0, 3840,       "pixels"),
    _ParamSpec("realtime.roi_x_end",        "ROI X fim",           "ROI",         "int",   0, 3840,       "pixels"),
    _ParamSpec("realtime.roi_y_start",      "ROI Y início",        "ROI",         "int",   0, 2160,       "pixels"),
    _ParamSpec("realtime.roi_y_end",        "ROI Y fim",           "ROI",         "int",   0, 2160,       "pixels"),
    _ParamSpec("roi_timer.seconds",         "Segundos confirmação","Timer ROI",   "int",   1,  60,        "segundos (1–60)"),
    _ParamSpec("arduino.port",              "Porta Serial",        "Arduino",     "str",                  hint="Ex.: COM5"),
    _ParamSpec("arduino.baudrate",          "Baudrate",            "Arduino",     "int",   1200, 115200,  "Ex.: 9600"),
    _ParamSpec("conveyor.delay_vidro_ms",   "Delay Vidro",         "Esteira",     "int",   0, 30000,      "ms"),
    _ParamSpec("conveyor.delay_papel_ms",   "Delay Papel",         "Esteira",     "int",   0, 30000,      "ms"),
    _ParamSpec("conveyor.delay_plastico_ms","Delay Plástico",      "Esteira",     "int",   0, 30000,      "ms"),
    _ParamSpec("conveyor.delay_metal_ms",            "Delay Metal",            "Esteira",     "int",   0, 30000,      "ms"),
    _ParamSpec("conveyor.delay_nao_identificado_ms", "Delay Não Identificado", "Esteira",     "int",   0, 30000,      "ms"),
]

_GROUP_ORDER = ["Inferência", "ROI", "Timer ROI", "Arduino", "Esteira"]


# ─────────────────────────────────────────────────────────────────────────────
# Validação de parâmetros de configuração (isolada de widgets)
# ─────────────────────────────────────────────────────────────────────────────

def _validate(spec: _ParamSpec, raw: str) -> tuple[bool, str]:
    raw = raw.strip()
    if not raw:
        return False, "Campo obrigatório."
    if spec.dtype == "float":
        try:
            v = float(raw.replace(",", "."))
        except ValueError:
            return False, "Valor deve ser numérico (ex: 0.50)."
        if spec.min_val is not None and v < spec.min_val:
            return False, f"Mínimo permitido: {spec.min_val}."
        if spec.max_val is not None and v > spec.max_val:
            return False, f"Máximo permitido: {spec.max_val}."
    elif spec.dtype == "int":
        try:
            v = int(raw)
        except ValueError:
            return False, "Valor deve ser inteiro."
        if spec.min_val is not None and v < spec.min_val:
            return False, f"Mínimo permitido: {int(spec.min_val)}."
        if spec.max_val is not None and v > spec.max_val:
            return False, f"Máximo permitido: {int(spec.max_val)}."
    elif spec.dtype == "str":
        if len(raw) > 128:
            return False, "Valor muito longo (máx. 128 chars)."
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Widget de linha editável de parâmetro
# ─────────────────────────────────────────────────────────────────────────────

class _ParamRow(QWidget):
    value_changed = Signal()

    def __init__(self, spec: _ParamSpec, parent=None):
        super().__init__(parent)
        self._spec = spec
        self._original: str = ""
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(8)

        lbl = QLabel(self._spec.label)
        lbl.setFixedWidth(165)
        lbl.setToolTip(f"Chave: {self._spec.key}")

        self._edit = QLineEdit()
        self._edit.setFixedWidth(140)
        self._edit.setPlaceholderText(self._spec.hint)
        self._edit.textChanged.connect(self._on_changed)

        hint_lbl = QLabel(self._spec.hint)
        hint_lbl.setStyleSheet("font-size: 10px;")
        hint_lbl.setFixedWidth(130)

        self._status = QLabel("")
        self._status.setFixedWidth(20)
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setFont(QFont("", 12))

        for w in (lbl, self._edit, hint_lbl, self._status):
            lay.addWidget(w)
        lay.addStretch()

    def set_value(self, value: str):
        self._original = value
        self._edit.blockSignals(True)
        self._edit.setText(value)
        self._edit.blockSignals(False)
        self._update_status()

    @property
    def key(self) -> str:
        return self._spec.key

    @property
    def current_text(self) -> str:
        return self._edit.text().strip()

    @property
    def is_dirty(self) -> bool:
        return self.current_text != self._original

    @property
    def is_valid(self) -> bool:
        ok, _ = _validate(self._spec, self.current_text)
        return ok

    def validation_error(self) -> str:
        _, msg = _validate(self._spec, self.current_text)
        return msg

    def reset(self):
        self._edit.setText(self._original)

    def _on_changed(self):
        self._update_status()
        self.value_changed.emit()

    def _update_status(self):
        if not self.is_dirty:
            self._status.setText("")
            self._edit.setStyleSheet("")
            return
        ok, _ = _validate(self._spec, self.current_text)
        if ok:
            self._status.setText("✔")
            self._status.setStyleSheet("color: #81c784;")          # verde claro (tema escuro)
            self._edit.setStyleSheet("border: 1px solid #81c784;")
        else:
            self._status.setText("✘")
            self._status.setStyleSheet("color: #ef9a9a;")          # vermelho claro (tema escuro)
            self._edit.setStyleSheet("border: 1px solid #ef9a9a;")
            self._edit.setToolTip(_validate(self._spec, self.current_text)[1])


# ─────────────────────────────────────────────────────────────────────────────
# Workers assíncronos
# ─────────────────────────────────────────────────────────────────────────────

class _LoadWorker(QThread):
    # IMPORTANTE: não usar "finished" — QThread já tem QThread::finished() (sem args)
    # no C++. O conflito de nomes causa TypeError silencioso e dados nunca são exibidos.
    result_ready = Signal(object)

    def run(self):
        from core.detection import model_registry
        from core.settings import settings_manager
        from db.repositories import audit_repo, user_repo
        self.result_ready.emit(_LoadResult(
            models=model_registry.list_models(),
            settings=settings_manager.get_all(),
            audit=audit_repo.get_recent(limit=50),
            users=user_repo.list_all(),
        ))


class _LoadResult:
    def __init__(self, models, settings, audit, users):
        self.models = models
        self.settings = settings
        self.audit = audit
        self.users = users


class _ActivateWorker(QThread):
    done = Signal(bool, str)

    def __init__(self, model_id: int, user_id: int):
        super().__init__()
        self._model_id = model_id
        self._user_id = user_id

    def run(self):
        try:
            from core.detection import model_registry, model_loader
            model_registry.set_active(self._model_id, user_id=self._user_id)
            model_loader.invalidate_cache()
            self.done.emit(True, "Modelo ativado com sucesso.")
        except Exception as exc:
            self.done.emit(False, str(exc))


class _SaveWorker(QThread):
    done = Signal(list, list)

    def __init__(self, changes: list[tuple[str, str]], user_id: int):
        super().__init__()
        self._changes = changes
        self._user_id = user_id

    def run(self):
        from core.settings import settings_manager
        saved: list[str] = []
        errors: list[tuple[str, str]] = []
        for key, value in self._changes:
            try:
                settings_manager.set(key, value, self._user_id)
                saved.append(key)
            except Exception as exc:
                errors.append((key, str(exc)))
        self.done.emit(saved, errors)


class _ImportWorker(QThread):
    """
    Importa uma pasta de pacote RecycleAI completo para operação.

    Passos:
      1. Valida o pacote fonte superficialmente (manifest + estrutura)
      2. Cria runtime_inferencia/modelos_importados/ se não existir
      3. Copia a pasta inteira para runtime_inferencia/modelos_importados/<pkg_name>/
      4. Valida o destino copiado com deep=True (torch.jit.load)
      5. Se inválido: remove cópia, registra MODEL_IMPORT_FAILED e aborta
      6. Se válido: chama model_registry.register_package() → MODEL_REGISTERED

    Sem treinamento, sem dataset, sem conversão de formato.
    Aceita apenas pastas de pacote produzidas pela aplicação de treinamento RecycleAI.
    """
    # (success, message, model_id)  — model_id=-1 se falhou
    done = Signal(bool, str, int)

    def __init__(self, src_dir: str, name: str, user_id: int):
        super().__init__()
        self._src_dir  = Path(src_dir)
        self._name     = name.strip()
        self._user_id  = user_id

    def run(self):
        from core.detection.package_validator import validate_package
        from core.detection import model_registry
        from db.repositories import audit_repo

        # ── 1. Validação superficial da pasta fonte ───────────────────────
        quick = validate_package(self._src_dir, deep=False)
        if not quick.ok:
            self._fail(audit_repo, quick.detail)
            return

        pkg_name = quick.pkg_name or self._src_dir.name

        # ── 2. Copiar pasta inteira para imported/ ────────────────────────
        try:
            _IMPORT_DIR.mkdir(parents=True, exist_ok=True)
            dest_dir = self._unique_dest_dir(_IMPORT_DIR, pkg_name)
            shutil.copytree(str(self._src_dir), str(dest_dir))
        except Exception as exc:
            self._fail(audit_repo, f"Falha ao copiar pacote: {exc}")
            return

        # ── 3. Validação profunda no destino (torch.jit.load) ────────────
        deep = validate_package(dest_dir, deep=True)
        if not deep.ok:
            try:
                shutil.rmtree(str(dest_dir), ignore_errors=True)
            except Exception:
                pass
            self._fail(audit_repo, f"Validação profunda falhou: {deep.detail}")
            return

        # ── 4. Registrar no banco via model_registry ──────────────────────
        display_name = self._name or deep.pkg_name
        try:
            model_id, _ = model_registry.register_package(
                pkg_dir  = dest_dir,
                name     = display_name,
                user_id  = self._user_id,
            )
        except Exception as exc:
            self._fail(audit_repo, f"Falha ao registrar no banco: {exc}")
            return

        self.done.emit(
            True,
            f"Pacote '{display_name}' importado com sucesso "
            f"({deep.nc} classes, {deep.size_mb:.1f} MB).",
            model_id,
        )

    # ── helpers ────────────────────────────────────────────────────────────

    def _fail(self, audit_repo, msg: str):
        try:
            audit_repo.record(
                "MODEL_IMPORT_FAILED",
                description=f"Importação de pacote falhou: {msg}",
                user_id=self._user_id,
            )
        except Exception:
            pass
        self.done.emit(False, msg, -1)

    @staticmethod
    def _unique_dest_dir(parent: Path, pkg_name: str) -> Path:
        """Retorna destino único para a pasta do pacote: adiciona _1, _2, … se já existe."""
        dest = parent / pkg_name
        if not dest.exists():
            return dest
        counter = 1
        while dest.exists():
            dest = parent / f"{pkg_name}_{counter}"
            counter += 1
        return dest


# ─────────────────────────────────────────────────────────────────────────────
# Tela principal
# ─────────────────────────────────────────────────────────────────────────────

class MaintenanceScreen(QWidget):
    """
    Tela do perfil `maintenance`.

    Parâmetros:
      session — sessão autenticada com role='maintenance'
      on_back — callback para retornar ao login
    """

    def __init__(self, session: Session, on_back=None, on_to_operation=None, parent=None):
        super().__init__(parent)
        self._session         = session
        self._on_back         = on_back
        self._on_to_operation = on_to_operation
        self._models: list[dict] = []
        self._users:  list[dict] = []
        self._param_rows: dict[str, _ParamRow] = {}

        self._load_worker:     _LoadWorker | None     = None
        self._activate_worker: _ActivateWorker | None = None
        self._save_worker:     _SaveWorker | None     = None
        self._import_worker:   _ImportWorker | None   = None

        self._build_ui()
        self._record_entry()
        self._reload_data()

    # ──────────────────────────────────────────────────────────── UI builder ──

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        root.addWidget(self._build_header())
        root.addWidget(self._build_status_bar())

        body = QSplitter(Qt.Horizontal)
        body.setContentsMargins(10, 10, 10, 10)
        body.setHandleWidth(6)

        body.addWidget(self._build_models_panel())

        right = QSplitter(Qt.Vertical)
        right.addWidget(self._build_settings_panel())
        right.addWidget(self._build_users_panel())
        right.addWidget(self._build_audit_panel())
        right.setSizes([300, 280, 180])
        body.addWidget(right)

        body.setSizes([460, 580])
        root.addWidget(body, stretch=1)
        root.addWidget(self._build_footer())

    # ── Cabeçalho ─────────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setStyleSheet("background: #1565c0;")
        lay = QHBoxLayout(header)
        lay.setContentsMargins(20, 14, 20, 14)

        title = QLabel("RecycleAI-Station — Administração")
        title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")

        user_lbl = QLabel(
            f"Usuário: <b>{self._session.login}</b>"
            f"&nbsp;|&nbsp;Perfil: <b>{self._session.display_role}</b>"
            f"&nbsp;|&nbsp;Sessão: {self._session.session_id[:8]}…"
        )
        user_lbl.setStyleSheet("color: #cce; font-size: 12px;")

        lay.addWidget(title)
        lay.addStretch()
        lay.addWidget(user_lbl)
        return header

    def _build_status_bar(self) -> QLabel:
        self._status_bar = QLabel("  Carregando dados…")
        self._status_bar.setStyleSheet("padding: 6px 20px; font-size: 12px;")
        return self._status_bar

    # ── Painel de modelos (com formulário de importação embutido) ─────────────

    def _build_models_panel(self) -> QWidget:
        box = QGroupBox("Modelos Registrados")
        box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        lay = QVBoxLayout(box)
        lay.setSpacing(8)

        # Label modelo ativo — cores adaptadas ao tema escuro
        self._active_model_lbl = QLabel("Modelo ativo: —")
        self._active_model_lbl.setStyleSheet(
            "color: #a5d6a7; font-size: 12px; padding: 4px 8px;"
            "background: #1a3a1a; border-radius: 4px;"
        )
        lay.addWidget(self._active_model_lbl)

        # Tabela de modelos
        self._model_table = QTableWidget(0, 5)
        self._model_table.setHorizontalHeaderLabels(["ID", "Nome", "Formato", "Status", "nc"])
        self._model_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for col in (0, 2, 3, 4):
            self._model_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )
        self._model_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._model_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._model_table.setAlternatingRowColors(True)
        self._model_table.verticalHeader().setVisible(False)
        self._model_table.setMinimumHeight(150)
        self._model_table.selectionModel().selectionChanged.connect(
            self._on_model_selection_changed
        )
        lay.addWidget(self._model_table)

        # Detalhes + botão ativar
        self._model_detail_lbl = QLabel("Selecione um modelo para ver detalhes.")
        self._model_detail_lbl.setStyleSheet("font-size: 11px; padding: 4px;")
        self._model_detail_lbl.setWordWrap(True)
        lay.addWidget(self._model_detail_lbl)

        self._btn_activate = QPushButton("✔  Ativar Modelo Selecionado")
        self._btn_activate.setEnabled(False)
        self._btn_activate.setStyleSheet(
            "QPushButton:enabled { background:#1565c0; color:white; font-weight:bold;"
            "  padding:6px 16px; border-radius:4px; }"
            "QPushButton:disabled { background:#bbb; color:#eee; }"
            "QPushButton:enabled:hover { background:#0d47a1; }"
        )
        self._btn_activate.clicked.connect(self._do_activate)
        lay.addWidget(self._btn_activate)

        # Separador
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #ddd;")
        lay.addWidget(sep)

        # Botão toggle do formulário de importação
        self._btn_toggle_import = QPushButton("📥  Importar Modelo Externo…")
        self._btn_toggle_import.setStyleSheet(
            "QPushButton { background:#f57f17; color:white; font-weight:bold;"
            "  padding:6px 16px; border-radius:4px; }"
            "QPushButton:hover { background:#e65100; }"
        )
        self._btn_toggle_import.clicked.connect(self._toggle_import_form)
        lay.addWidget(self._btn_toggle_import)

        # Formulário de importação (oculto por padrão)
        lay.addWidget(self._build_import_form())

        return box

    def _build_import_form(self) -> QFrame:
        """Formulário embutido de importação — oculto até o usuário expandir."""
        self._import_frame = QFrame()
        self._import_frame.setFrameShape(QFrame.StyledPanel)
        self._import_frame.setStyleSheet(
            "QFrame { border: 1px solid #888;"
            "  border-radius: 6px; padding: 4px; }"
        )
        self._import_frame.setVisible(False)

        lay = QVBoxLayout(self._import_frame)
        lay.setSpacing(8)
        lay.setContentsMargins(10, 10, 10, 10)

        # Nota de escopo
        scope_lbl = QLabel(
            "<b>Importação de pacote operacional pronto.</b><br>"
            "<span style='color:#e65100; font-size:11px;'>"
            "Esta aplicação não treina modelos — importe pastas de pacote RecycleAI "
            "(<i>&lt;nome&gt;_package/</i> com manifest.json), geradas pela aplicação de treinamento."
            "</span>"
        )
        scope_lbl.setWordWrap(True)
        lay.addWidget(scope_lbl)

        # Linha: pasta do pacote
        file_row = QHBoxLayout()
        file_lbl = QLabel("Pacote:")
        file_lbl.setFixedWidth(60)
        self._import_path_edit = QLineEdit()
        self._import_path_edit.setReadOnly(True)
        self._import_path_edit.setPlaceholderText("Nenhuma pasta selecionada…")
        btn_browse = QPushButton("Selecionar…")
        btn_browse.setFixedWidth(90)
        btn_browse.clicked.connect(self._do_browse_file)
        file_row.addWidget(file_lbl)
        file_row.addWidget(self._import_path_edit)
        file_row.addWidget(btn_browse)
        lay.addLayout(file_row)

        # Linha: nome do modelo
        name_row = QHBoxLayout()
        name_lbl = QLabel("Nome:")
        name_lbl.setFixedWidth(60)
        self._import_name_edit = QLineEdit()
        self._import_name_edit.setPlaceholderText("Nome para identificação no sistema…")
        self._import_name_edit.setMaxLength(128)
        name_row.addWidget(name_lbl)
        name_row.addWidget(self._import_name_edit)
        lay.addLayout(name_row)

        # Status da seleção (tamanho, formato, resultado de validação rápida)
        self._import_status_lbl = QLabel("")
        self._import_status_lbl.setStyleSheet("font-size: 11px; padding: 2px 4px;")
        self._import_status_lbl.setWordWrap(True)
        lay.addWidget(self._import_status_lbl)

        # Botões
        btn_row = QHBoxLayout()
        self._btn_import_cancel = QPushButton("Cancelar")
        self._btn_import_cancel.clicked.connect(self._do_cancel_import)

        self._btn_import_confirm = QPushButton("✔  Importar e Registrar")
        self._btn_import_confirm.setEnabled(False)
        self._btn_import_confirm.setStyleSheet(
            "QPushButton:enabled { background:#2e7d32; color:white; font-weight:bold;"
            "  padding:6px 14px; border-radius:4px; }"
            "QPushButton:disabled { background:#bbb; color:#eee; }"
            "QPushButton:enabled:hover { background:#1b5e20; }"
        )
        self._btn_import_confirm.clicked.connect(self._do_confirm_import)

        btn_row.addWidget(self._btn_import_cancel)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_import_confirm)
        lay.addLayout(btn_row)

        return self._import_frame

    # ── Painel de configurações editável ──────────────────────────────────────

    def _build_settings_panel(self) -> QWidget:
        outer = QGroupBox("Configurações do Sistema")
        outer.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        outer_lay = QVBoxLayout(outer)
        outer_lay.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        c_lay = QVBoxLayout(container)
        c_lay.setSpacing(10)
        c_lay.setContentsMargins(4, 4, 4, 4)

        groups: dict[str, list[_ParamSpec]] = {}
        for spec in _PARAM_SPECS:
            groups.setdefault(spec.group, []).append(spec)

        for group_name in _GROUP_ORDER:
            specs = groups.get(group_name, [])
            if not specs:
                continue
            grp_box = QGroupBox(group_name)
            grp_box.setStyleSheet(
                "QGroupBox { font-weight:normal; font-size:12px;"
                "  border:1px solid #3a3a3a; border-radius:4px; margin-top:8px; }"
                "QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; }"
            )
            grp_lay = QVBoxLayout(grp_box)
            grp_lay.setSpacing(2)
            for spec in specs:
                row = _ParamRow(spec)
                row.value_changed.connect(self._on_param_changed)
                grp_lay.addWidget(row)
                self._param_rows[spec.key] = row
            c_lay.addWidget(grp_box)

        c_lay.addStretch()
        scroll.setWidget(container)
        outer_lay.addWidget(scroll, stretch=1)

        action_bar = QHBoxLayout()
        action_bar.setContentsMargins(0, 4, 0, 0)

        self._dirty_lbl = QLabel("")
        self._dirty_lbl.setStyleSheet("color: #e65100; font-size: 11px;")

        self._btn_discard = QPushButton("↩  Descartar")
        self._btn_discard.setEnabled(False)
        self._btn_discard.clicked.connect(self._do_discard)

        self._btn_save = QPushButton("💾  Salvar Alterações")
        self._btn_save.setEnabled(False)
        self._btn_save.setStyleSheet(
            "QPushButton:enabled { background:#2e7d32; color:white; font-weight:bold;"
            "  padding:6px 16px; border-radius:4px; }"
            "QPushButton:disabled { background:#bbb; color:#eee; }"
            "QPushButton:enabled:hover { background:#1b5e20; }"
        )
        self._btn_save.clicked.connect(self._do_save)

        action_bar.addWidget(self._dirty_lbl)
        action_bar.addStretch()
        action_bar.addWidget(self._btn_discard)
        action_bar.addWidget(self._btn_save)
        outer_lay.addLayout(action_bar)

        return outer

    # ── Painel de auditoria ───────────────────────────────────────────────────

    def _build_audit_panel(self) -> QWidget:
        box = QGroupBox("Auditoria Recente (últimos 50 eventos)")
        box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        lay = QVBoxLayout(box)

        self._audit_table = QTableWidget(0, 4)
        self._audit_table.setHorizontalHeaderLabels(
            ["Timestamp", "Evento", "Usuário", "Descrição / Parâmetro"]
        )
        self._audit_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._audit_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._audit_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._audit_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._audit_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._audit_table.setAlternatingRowColors(True)
        self._audit_table.verticalHeader().setVisible(False)
        self._audit_table.setMinimumHeight(120)
        lay.addWidget(self._audit_table)

        return box

    # ── Painel de usuários ────────────────────────────────────────────────────

    def _build_users_panel(self) -> QWidget:
        box = QGroupBox("Usuários do Sistema")
        box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        lay = QVBoxLayout(box)
        lay.setSpacing(6)

        # Tabela de usuários
        self._user_table = QTableWidget(0, 4)
        self._user_table.setHorizontalHeaderLabels(
            ["Login", "Perfil", "Status", "Senha Temp"]
        )
        self._user_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3):
            self._user_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )
        self._user_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._user_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._user_table.setAlternatingRowColors(True)
        self._user_table.verticalHeader().setVisible(False)
        self._user_table.setMinimumHeight(100)
        self._user_table.selectionModel().selectionChanged.connect(
            self._on_user_selection_changed
        )
        lay.addWidget(self._user_table)

        # Barra de ações
        btn_bar = QHBoxLayout()

        self._btn_new_user = QPushButton("➕  Novo Usuário")
        self._btn_new_user.setStyleSheet(
            "QPushButton { background:#1565c0; color:white; font-weight:bold;"
            "  padding:5px 12px; border-radius:4px; }"
            "QPushButton:hover { background:#0d47a1; }"
        )
        self._btn_new_user.clicked.connect(self._do_create_user)

        self._btn_reset_pwd = QPushButton("🔑  Resetar Senha")
        self._btn_reset_pwd.setEnabled(False)
        self._btn_reset_pwd.clicked.connect(self._do_reset_password)

        self._btn_toggle_active = QPushButton("Desativar")
        self._btn_toggle_active.setEnabled(False)
        self._btn_toggle_active.clicked.connect(self._do_toggle_active)

        self._btn_delete_user = QPushButton("🗑  Excluir")
        self._btn_delete_user.setEnabled(False)
        self._btn_delete_user.setStyleSheet(
            "QPushButton { background:#c62828; color:white; font-weight:bold;"
            "  padding:5px 12px; border-radius:4px; }"
            "QPushButton:hover { background:#b71c1c; }"
            "QPushButton:disabled { background:#bbb; color:#eee; }"
        )
        self._btn_delete_user.clicked.connect(self._do_delete_user)

        btn_bar.addWidget(self._btn_new_user)
        btn_bar.addWidget(self._btn_reset_pwd)
        btn_bar.addWidget(self._btn_toggle_active)
        btn_bar.addWidget(self._btn_delete_user)
        btn_bar.addStretch()
        lay.addLayout(btn_bar)

        return box

    _ROLE_DISPLAY = {"maintenance": "Administrador", "operator": "Operador"}

    def _populate_users(self, users):
        self._user_table.setRowCount(0)
        for u in users:
            row = self._user_table.rowCount()
            self._user_table.insertRow(row)
            active       = bool(u["active"])
            must_chg     = bool(u["must_change_password"])
            status       = "ativo" if active else "inativo"
            temp_lbl     = "sim" if must_chg else "não"
            display_role = self._ROLE_DISPLAY.get(u["role"], u["role"])
            for col, val in enumerate([u["login"], display_role, status, temp_lbl]):
                item = QTableWidgetItem(val)
                item.setData(Qt.UserRole, u["id"])
                if col == 1:
                    item.setData(Qt.UserRole + 1, u["role"])  # raw role for guard checks
                if not active:
                    item.setForeground(QColor("#888"))       # cinza para inativos
                elif u["role"] == "maintenance":
                    item.setForeground(QColor("#90caf9"))    # azul claro para administradores
                self._user_table.setItem(row, col, item)

    def _on_user_selection_changed(self):
        selected = self._user_table.selectedItems()
        if not selected:
            self._btn_reset_pwd.setEnabled(False)
            self._btn_toggle_active.setEnabled(False)
            self._btn_delete_user.setEnabled(False)
            return
        row      = selected[0].row()
        uid      = self._user_table.item(row, 0).data(Qt.UserRole)
        login    = self._user_table.item(row, 0).text()
        status   = self._user_table.item(row, 2).text()
        active   = status == "ativo"
        is_self  = (uid == self._session.user_id)
        is_admin = (login == "admin")

        self._btn_reset_pwd.setEnabled(True)
        self._btn_toggle_active.setEnabled(not is_self)
        self._btn_toggle_active.setText("Desativar" if active else "Ativar")
        # Admin e conta própria nunca podem ser excluídos pela UI
        self._btn_delete_user.setEnabled(not is_self and not is_admin)

    def _do_create_user(self):
        dlg = _CreateUserDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        login, role, pwd = dlg.result_data()
        try:
            from core.auth.authenticator import create_user, LoginAlreadyExists
            create_user(login, role, pwd, created_by=self._session.user_id)
            self._set_status(
                f"Usuário '{login}' ({role}) criado com senha temporária.",
                loading=False, ok=True,
            )
            self._reload_data()
        except LoginAlreadyExists as exc:
            QMessageBox.warning(self, "Login em uso", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Erro", f"Falha ao criar usuário:\n{exc}")

    def _do_reset_password(self):
        selected = self._user_table.selectedItems()
        if not selected:
            return
        row   = selected[0].row()
        uid   = self._user_table.item(row, 0).data(Qt.UserRole)
        login = self._user_table.item(row, 0).text()

        dlg = _ResetPasswordDialog(login, self)
        if dlg.exec() != QDialog.Accepted:
            return
        pwd = dlg.result_password()
        try:
            from core.auth.authenticator import reset_password
            reset_password(uid, pwd, admin_user_id=self._session.user_id)
            self._set_status(
                f"Senha de '{login}' resetada. Troca obrigatória no próximo login.",
                loading=False, ok=True,
            )
            self._reload_data()
        except Exception as exc:
            QMessageBox.critical(self, "Erro", f"Falha ao resetar senha:\n{exc}")

    def _do_toggle_active(self):
        selected = self._user_table.selectedItems()
        if not selected:
            return
        row      = selected[0].row()
        uid      = self._user_table.item(row, 0).data(Qt.UserRole)
        login    = self._user_table.item(row, 0).text()
        raw_role = self._user_table.item(row, 1).data(Qt.UserRole + 1)
        status   = self._user_table.item(row, 2).text()
        active   = status == "ativo"

        # Guarda de segurança: impede desativar a última conta maintenance ativa
        if active and raw_role == "maintenance":
            from db.repositories import user_repo
            if user_repo.count_active_maintenance() <= 1:
                QMessageBox.warning(
                    self,
                    "Ação bloqueada",
                    "Não é possível desativar a última conta de administrador ativa.\n"
                    "Crie outro administrador antes de desativar esta conta.",
                )
                return

        action = "desativar" if active else "ativar"
        reply  = QMessageBox.question(
            self, "Confirmar",
            f"Deseja {action} o usuário '{login}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            from core.auth.authenticator import set_active
            set_active(uid, active=not active, admin_user_id=self._session.user_id)
            verb = "ativado" if not active else "desativado"
            self._set_status(
                f"Usuário '{login}' {verb}.",
                loading=False, ok=True,
            )
            self._reload_data()
        except Exception as exc:
            QMessageBox.critical(self, "Erro", f"Falha ao {action} usuário:\n{exc}")

    def _do_delete_user(self):
        selected = self._user_table.selectedItems()
        if not selected:
            return
        row   = selected[0].row()
        uid   = self._user_table.item(row, 0).data(Qt.UserRole)
        login = self._user_table.item(row, 0).text()

        reply = QMessageBox.warning(
            self,
            "Excluir usuário",
            f"Esta ação é irreversível.\n\n"
            f"Excluir permanentemente o usuário '{login}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            from core.auth.authenticator import delete_user, CannotDeleteAdmin, AuthError
            delete_user(uid, admin_user_id=self._session.user_id)
            self._set_status(
                f"Usuário '{login}' excluído permanentemente.",
                loading=False, ok=True,
            )
            self._reload_data()
        except CannotDeleteAdmin as exc:
            QMessageBox.critical(self, "Ação bloqueada", str(exc))
        except AuthError as exc:
            QMessageBox.warning(self, "Ação bloqueada", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Erro", f"Falha ao excluir usuário:\n{exc}")

    def _do_go_operation(self):
        if self._on_to_operation:
            self._on_to_operation(self._session)

    # ── Rodapé ────────────────────────────────────────────────────────────────

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        footer.setStyleSheet("border-top: 1px solid #888;")
        lay = QHBoxLayout(footer)
        lay.setContentsMargins(20, 10, 20, 10)

        self._btn_reload = QPushButton("↺  Atualizar dados")
        self._btn_reload.clicked.connect(self._reload_data)

        self._btn_logout = QPushButton("← Logout")
        self._btn_logout.setStyleSheet(
            "QPushButton { background:#c62828; color:white; font-weight:bold;"
            "  padding:6px 18px; border-radius:4px; }"
            "QPushButton:hover { background:#b71c1c; }"
        )
        self._btn_logout.clicked.connect(self._do_logout)

        self._btn_go_op = QPushButton("▶  Ir para Operação")
        self._btn_go_op.setStyleSheet(
            "QPushButton { background:#2e7d32; color:white; font-weight:bold;"
            "  padding:6px 18px; border-radius:4px; }"
            "QPushButton:hover { background:#1b5e20; }"
        )
        self._btn_go_op.setVisible(self._on_to_operation is not None)
        self._btn_go_op.clicked.connect(self._do_go_operation)

        self._btn_diag = QPushButton("🔧  Diagnóstico de Hardware")
        self._btn_diag.setStyleSheet(
            "QPushButton { background:#4a148c; color:white; font-weight:bold;"
            "  padding:6px 18px; border-radius:4px; }"
            "QPushButton:hover { background:#6a1b9a; }"
        )
        self._btn_diag.clicked.connect(self._do_abrir_diagnostico)

        lay.addWidget(self._btn_logout)
        lay.addStretch()
        lay.addWidget(self._btn_reload)
        lay.addWidget(self._btn_diag)
        lay.addWidget(self._btn_go_op)
        return footer

    # ─────────────────────────────────────────────── Data load / populate ────

    def _record_entry(self):
        try:
            from db.repositories import audit_repo
            audit_repo.record(
                "MAINTENANCE_ENTER",
                description=f"Usuário '{self._session.login}' entrou na Administração",
                user_id=self._session.user_id,
            )
        except Exception:
            pass

    def _reload_data(self):
        self._set_status("Carregando dados…", loading=True)
        self._btn_reload.setEnabled(False)
        self._btn_activate.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_discard.setEnabled(False)

        self._load_worker = _LoadWorker()
        self._load_worker.result_ready.connect(self._on_data_loaded)
        self._load_worker.start()

    def _on_data_loaded(self, result: _LoadResult):
        self._models = result.models
        self._users  = result.users
        self._populate_models(result.models)
        self._fill_param_rows(result.settings)
        self._populate_audit(result.audit)
        self._populate_users(result.users)
        self._set_status("Dados atualizados.", loading=False, ok=True)
        self._btn_reload.setEnabled(True)

    # ── Modelos ───────────────────────────────────────────────────────────────

    def _populate_models(self, models: list[dict]):
        self._model_table.setRowCount(0)
        active_name = "—"
        for m in models:
            row = self._model_table.rowCount()
            self._model_table.insertRow(row)
            status    = m["status"]
            is_active = status == "active"
            if is_active:
                active_name = m["name"]
            for col, val in enumerate([
                str(m["id"]), m["name"], m["format"], status,
                str(m["nc"]) if m["nc"] else "?",
            ]):
                item = QTableWidgetItem(val)
                item.setData(Qt.UserRole, m["id"])
                if is_active:
                    # Fundo verde escuro para legibilidade no tema escuro
                    item.setBackground(QColor("#1a3a1a"))
                    item.setForeground(QColor("#a5d6a7"))
                    f = item.font(); f.setBold(True); item.setFont(f)
                elif status == "invalid":
                    # Fundo vermelho escuro para tema escuro
                    item.setBackground(QColor("#3a0a0a"))
                    item.setForeground(QColor("#ef9a9a"))
                self._model_table.setItem(row, col, item)
        self._active_model_lbl.setText(f"Modelo ativo: <b>{active_name}</b>")

    def _on_model_selection_changed(self):
        selected = self._model_table.selectedItems()
        if not selected:
            self._model_detail_lbl.setText("Selecione um modelo para ver detalhes.")
            self._btn_activate.setEnabled(False)
            return
        model_id = self._model_table.item(selected[0].row(), 0).data(Qt.UserRole)
        model    = next((m for m in self._models if m["id"] == model_id), None)
        if not model:
            return
        classes = ", ".join(model["class_names"]) if model["class_names"] else "—"
        detail  = (
            f"<b>Caminho:</b> {model['file_path']}<br>"
            f"<b>Classes:</b> {classes}<br>"
            f"<b>Origem:</b> {model['origin']}&nbsp;&nbsp;"
            f"<b>Registrado:</b> {model.get('registered_at', '—')}"
        )
        if model.get("notes"):
            detail += f"<br><b>Notas:</b> {model['notes']}"
        self._model_detail_lbl.setText(detail)
        is_active  = model["status"] == "active"
        is_invalid = model["status"] == "invalid"
        self._btn_activate.setEnabled(not is_active and not is_invalid)
        self._btn_activate.setText(
            "✔  Ativar Modelo Selecionado" if not is_active else "✔  Já ativo"
        )

    # ── Importação de modelo ──────────────────────────────────────────────────

    def _toggle_import_form(self):
        visible = not self._import_frame.isVisible()
        self._import_frame.setVisible(visible)
        if not visible:
            self._reset_import_form()

    def _reset_import_form(self):
        self._import_path_edit.clear()
        self._import_name_edit.clear()
        self._import_status_lbl.setText("")
        self._btn_import_confirm.setEnabled(False)

    def _do_browse_file(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Selecionar pasta do pacote RecycleAI (<nome>_package/)",
            str(Path.home()),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not path:
            return
        self._on_package_selected(path)

    def _on_package_selected(self, path: str):
        """Preenche o formulário e executa validação rápida (nível 1) ao selecionar a pasta."""
        from core.detection.package_validator import validate_package

        p = Path(path)
        self._import_path_edit.setText(path)

        # Pré-preenche nome com o nome da pasta sem sufixo "_package" (editável)
        if not self._import_name_edit.text().strip():
            folder_name = p.name
            suggested = folder_name[:-len("_package")] if folder_name.endswith("_package") else folder_name
            self._import_name_edit.setText(suggested)

        # Validação superficial (nível 1 — sem carregar o modelo)
        result = validate_package(p, deep=False)

        if result.ok:
            classes_preview = ", ".join(result.class_names[:6])
            if len(result.class_names) > 6:
                classes_preview += "…"
            self._import_status_lbl.setText(
                f"<span style='color:#81c784;'>✔ Pacote OK: {result.pkg_name}"
                f" — {result.nc} classes — {result.size_mb:.1f} MB</span><br>"
                f"<span style='color:#bbb; font-size:10px;'>"
                f"Classes: {classes_preview}"
                f" — Validação profunda (torch.jit.load) será executada ao importar.</span>"
            )
            self._btn_import_confirm.setEnabled(True)
        else:
            self._import_status_lbl.setText(
                f"<span style='color:#ef9a9a;'>✘ {result.detail}</span>"
            )
            self._btn_import_confirm.setEnabled(False)

    def _do_cancel_import(self):
        self._import_frame.setVisible(False)
        self._reset_import_form()

    def _do_confirm_import(self):
        src  = self._import_path_edit.text().strip()
        name = self._import_name_edit.text().strip()

        if not src:
            return
        if not name:
            name = Path(src).name

        self._btn_import_confirm.setEnabled(False)
        self._btn_import_cancel.setEnabled(False)
        self._btn_reload.setEnabled(False)
        self._import_status_lbl.setText(
            "<span style='color:#90caf9;'>⏳ Copiando pacote e validando "
            "(torch.jit.load)… aguarde.</span>"
        )
        self._set_status("Importando pacote…", loading=True)

        self._import_worker = _ImportWorker(src, name, self._session.user_id)
        self._import_worker.done.connect(self._on_import_done)
        self._import_worker.start()

    def _on_import_done(self, success: bool, message: str, model_id: int):
        self._btn_import_cancel.setEnabled(True)
        self._btn_reload.setEnabled(True)

        if success:
            self._import_status_lbl.setText(
                f"<span style='color:#81c784;'>✔ {message}</span>"
            )
            self._set_status(message, loading=False, ok=True)
            # Oculta formulário e recarrega lista
            self._import_frame.setVisible(False)
            self._reset_import_form()
            self._reload_data()
        else:
            self._import_status_lbl.setText(
                f"<span style='color:#ef9a9a;'>✘ {message}</span>"
            )
            self._set_status(f"Importação falhou: {message}", loading=False, ok=False)
            self._btn_import_confirm.setEnabled(True)   # permite tentar novamente

    # ── Configurações ─────────────────────────────────────────────────────────

    def _fill_param_rows(self, settings: dict[str, str]):
        for key, row in self._param_rows.items():
            row.set_value(settings.get(key, ""))
        self._update_save_state()

    def _on_param_changed(self):
        self._update_save_state()

    def _update_save_state(self):
        dirty_rows   = [r for r in self._param_rows.values() if r.is_dirty]
        invalid_dirty = [r for r in dirty_rows if not r.is_valid]
        n_dirty   = len(dirty_rows)
        n_invalid = len(invalid_dirty)

        if n_dirty == 0:
            self._dirty_lbl.setText("")
            self._btn_save.setEnabled(False)
            self._btn_discard.setEnabled(False)
        elif n_invalid > 0:
            self._dirty_lbl.setText(
                f"{n_dirty} campo(s) alterado(s) — {n_invalid} com erro(s) (✘)"
            )
            self._btn_save.setEnabled(False)
            self._btn_discard.setEnabled(True)
        else:
            self._dirty_lbl.setText(f"{n_dirty} campo(s) alterado(s) — pronto para salvar")
            self._btn_save.setEnabled(True)
            self._btn_discard.setEnabled(True)

    def _do_discard(self):
        for row in self._param_rows.values():
            if row.is_dirty:
                row.reset()
        self._update_save_state()
        self._set_status("Alterações descartadas.", loading=False, ok=True)

    def _do_save(self):
        changes = [
            (r.key, r.current_text)
            for r in self._param_rows.values()
            if r.is_dirty and r.is_valid
        ]
        if not changes:
            return
        self._btn_save.setEnabled(False)
        self._btn_discard.setEnabled(False)
        self._btn_reload.setEnabled(False)
        self._set_status(f"Salvando {len(changes)} parâmetro(s)…", loading=True)

        self._save_worker = _SaveWorker(changes, self._session.user_id)
        self._save_worker.done.connect(self._on_save_done)
        self._save_worker.start()

    def _on_save_done(self, saved: list[str], errors: list[tuple[str, str]]):
        if errors:
            err_keys = ", ".join(k for k, _ in errors)
            self._set_status(
                f"Salvo: {len(saved)} | Erro: {len(errors)} ({err_keys})",
                loading=False, ok=False,
            )
        else:
            self._set_status(
                f"{len(saved)} parâmetro(s) salvo(s) com sucesso.",
                loading=False, ok=True,
            )
        self._reload_data()

    # ── Auditoria ─────────────────────────────────────────────────────────────

    def _populate_audit(self, rows):
        self._audit_table.setRowCount(0)
        for r in rows:
            idx = self._audit_table.rowCount()
            self._audit_table.insertRow(idx)
            ts    = str(r["ts"])[:19] if r["ts"] else "—"
            event = r["event_type"] or "—"
            user  = r["login"] or "sistema"
            if event == "CONFIG_CHANGE" and r["param_key"]:
                desc = f"{r['param_key']}: {r['old_value']} → {r['new_value']}"
            else:
                desc = r["description"] or r["param_key"] or "—"
            for col, val in enumerate([ts, event, user, desc]):
                item = QTableWidgetItem(val)
                # Cores claras para legibilidade no tema escuro
                if "ERROR" in event or "FAIL" in event:
                    item.setForeground(QColor("#ef9a9a"))   # vermelho claro
                elif event in ("MODEL_ACTIVATED", "MODEL_REGISTERED"):
                    item.setForeground(QColor("#a5d6a7"))   # verde claro
                elif event == "CONFIG_CHANGE":
                    item.setForeground(QColor("#90caf9"))   # azul claro
                elif event == "MODEL_IMPORT_FAILED":
                    item.setForeground(QColor("#ffab91"))   # laranja claro
                self._audit_table.setItem(idx, col, item)

    # ── Ações de modelo ───────────────────────────────────────────────────────

    def _do_activate(self):
        selected = self._model_table.selectedItems()
        if not selected:
            return
        model_id = self._model_table.item(selected[0].row(), 0).data(Qt.UserRole)
        self._btn_activate.setEnabled(False)
        self._btn_reload.setEnabled(False)
        self._set_status("Ativando modelo, aguarde…", loading=True)

        self._activate_worker = _ActivateWorker(model_id, self._session.user_id)
        self._activate_worker.done.connect(self._on_activate_done)
        self._activate_worker.start()

    def _on_activate_done(self, success: bool, message: str):
        self._set_status(
            ("OK: " if success else "ERRO: ") + message,
            loading=False, ok=success,
        )
        self._reload_data()

    def _do_abrir_diagnostico(self):
        """Abre o diálogo de diagnóstico manual de hardware."""
        from gui.screens.hardware_diag_dialog import HardwareDiagDialog
        dlg = HardwareDiagDialog(session=self._session, parent=self)
        dlg.exec()
        # Recarrega o log de auditoria após o diagnóstico encerrar
        self._reload_data()

    def _do_logout(self):
        try:
            from db.repositories import audit_repo
            audit_repo.record(
                "MAINTENANCE_EXIT",
                description=f"Usuário '{self._session.login}' saiu da Administração",
                user_id=self._session.user_id,
            )
        except Exception:
            pass
        if self._on_back:
            self._on_back()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, text: str, *, loading: bool, ok: bool = True):
        if loading:
            style = "color:#64b5f6;"
        elif ok:
            style = "color:#81c784;"
        else:
            style = "color:#e57373;"
        self._status_bar.setStyleSheet(f"{style} padding:6px 20px; font-size:12px;")
        self._status_bar.setText(f"  {text}")

    def closeEvent(self, event):
        for worker in (
            self._load_worker, self._activate_worker,
            self._save_worker, self._import_worker,
        ):
            if worker and worker.isRunning():
                worker.wait(2000)
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Dialogs de gestão de usuários
# ─────────────────────────────────────────────────────────────────────────────

class _CreateUserDialog(QDialog):
    """Formulário de criação de novo usuário com senha temporária."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Criar Novo Usuário")
        self.setModal(True)
        self.setMinimumWidth(340)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        form = QFormLayout()

        self._login_edit = QLineEdit()
        self._login_edit.setPlaceholderText("login (sem espaços)")
        self._login_edit.setMaxLength(64)
        form.addRow("Login:", self._login_edit)

        self._role_combo = QComboBox()
        self._role_combo.addItems(["operator"])
        form.addRow("Perfil:", self._role_combo)

        self._pwd_edit = QLineEdit()
        self._pwd_edit.setEchoMode(QLineEdit.Password)
        self._pwd_edit.setPlaceholderText("mínimo 6 caracteres")
        form.addRow("Senha temporária:", self._pwd_edit)

        self._pwd_confirm = QLineEdit()
        self._pwd_confirm.setEchoMode(QLineEdit.Password)
        self._pwd_confirm.setPlaceholderText("repita a senha")
        form.addRow("Confirmar senha:", self._pwd_confirm)

        lay.addLayout(form)

        self._status = QLabel("")
        self._status.setStyleSheet("color:#c62828; font-size:11px;")
        lay.addWidget(self._status)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _validate_and_accept(self):
        login = self._login_edit.text().strip()
        pwd   = self._pwd_edit.text()
        conf  = self._pwd_confirm.text()

        if not login:
            self._status.setText("Login obrigatório.")
            return
        if " " in login:
            self._status.setText("Login não pode conter espaços.")
            return
        if len(pwd) < 6:
            self._status.setText("Senha deve ter ao menos 6 caracteres.")
            return
        if pwd != conf:
            self._status.setText("Senhas não coincidem.")
            return
        self.accept()

    def result_data(self) -> tuple[str, str, str]:
        """Retorna (login, role, password)."""
        return (
            self._login_edit.text().strip(),
            self._role_combo.currentText(),
            self._pwd_edit.text(),
        )


class _ResetPasswordDialog(QDialog):
    """Formulário de reset de senha temporária para outro usuário."""

    def __init__(self, target_login: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Resetar Senha — {target_login}")
        self.setModal(True)
        self.setMinimumWidth(320)
        self._build(target_login)

    def _build(self, target_login: str):
        lay = QVBoxLayout(self)

        lay.addWidget(QLabel(
            f"<b>Usuário:</b> {target_login}<br>"
            "<span style='color:#555; font-size:11px;'>"
            "A senha temporária exigirá troca no próximo login.</span>"
        ))

        form = QFormLayout()
        self._pwd_edit = QLineEdit()
        self._pwd_edit.setEchoMode(QLineEdit.Password)
        self._pwd_edit.setPlaceholderText("mínimo 6 caracteres")
        form.addRow("Nova senha temporária:", self._pwd_edit)

        self._pwd_confirm = QLineEdit()
        self._pwd_confirm.setEchoMode(QLineEdit.Password)
        self._pwd_confirm.setPlaceholderText("repita a senha")
        form.addRow("Confirmar senha:", self._pwd_confirm)

        lay.addLayout(form)

        self._status = QLabel("")
        self._status.setStyleSheet("color:#c62828; font-size:11px;")
        lay.addWidget(self._status)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _validate_and_accept(self):
        pwd  = self._pwd_edit.text()
        conf = self._pwd_confirm.text()
        if len(pwd) < 6:
            self._status.setText("Senha deve ter ao menos 6 caracteres.")
            return
        if pwd != conf:
            self._status.setText("Senhas não coincidem.")
            return
        self.accept()

    def result_password(self) -> str:
        return self._pwd_edit.text()
