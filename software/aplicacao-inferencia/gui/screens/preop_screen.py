from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox,
)
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont

from core.auth.session import Session
from core.diagnostics.pre_op_check import CheckResult, PreOpReport, run as run_diag


# ---------------------------------------------------------------------------
# Worker thread para não travar a GUI durante as checagens
# ---------------------------------------------------------------------------

class _TrabalhadorDiagnostico(QThread):
    # IMPORTANTE: NÃO usar "finished" como nome — QThread já possui QThread::finished()
    # (sem argumentos) embutido no C++.  No PySide6, nomear o sinal customizado igual
    # cria conflito: o slot conectado recebe chamada sem args → TypeError silencioso →
    # _on_diag_done nunca é chamado corretamente → tela fica presa em "aguarde...".
    result_ready = Signal(object)  # emite PreOpReport

    def __init__(self, user_id: int):
        super().__init__()
        self._user_id = user_id

    def run(self):
        try:
            report = run_diag(user_id=self._user_id)
        except Exception:
            report = PreOpReport()
        self.result_ready.emit(report)


# ---------------------------------------------------------------------------
# Linha de resultado de uma checagem
# ---------------------------------------------------------------------------

class _LinhaDiagnostico(QWidget):
    def __init__(self, result: CheckResult, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)

        # Cores claras para legibilidade no tema escuro
        color = "#81c784" if result.ok else "#ef9a9a"
        icon  = "✔" if result.ok else "✘"

        lbl_icon = QLabel(f'<span style="color:{color}; font-size:15px"><b>{icon}</b></span>')
        lbl_icon.setFixedWidth(24)

        lbl_name = QLabel(f"<b>{result.name}</b>")
        lbl_name.setFixedWidth(180)

        lbl_detail = QLabel(result.detail)

        lbl_ms = QLabel(f"{result.elapsed_ms} ms")
        lbl_ms.setStyleSheet("font-size: 11px;")
        lbl_ms.setFixedWidth(60)
        lbl_ms.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        for w in (lbl_icon, lbl_name, lbl_detail):
            layout.addWidget(w)
        layout.addStretch()
        layout.addWidget(lbl_ms)


# ---------------------------------------------------------------------------
# Tela principal de pré-operação
# ---------------------------------------------------------------------------

class PreOpScreen(QWidget):
    """
    Exibe o resultado do diagnóstico pré-operação e controla o avanço
    para a triagem. Roda as checagens em QThread para não bloquear a GUI.

    Sinais:
      on_start(session, report) — emitido quando operador confirma início
      on_back()                 — volta para o login
    """

    def __init__(
        self,
        session: Session,
        on_start=None,
        on_back=None,
        back_label: str = "← Sair",
        parent=None,
    ):
        super().__init__(parent)
        self._session = session
        self._on_start = on_start
        self._on_back = on_back
        self._back_label = back_label
        self._report: PreOpReport | None = None
        self._worker: _TrabalhadorDiagnostico | None = None
        self._watchdog: QTimer | None = None
        self._build_ui()
        self._run_diagnostics()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(24, 24, 24, 24)

        # Cabeçalho
        title = QLabel("<h2>Diagnóstico Pré-Operação</h2>")
        subtitle = QLabel(
            f"Usuário: <b>{self._session.login}</b> "
            f"&nbsp;|&nbsp; Perfil: <b>{self._session.display_role}</b>"
        )
        root.addWidget(title)
        root.addWidget(subtitle)

        # Grupo de checagens
        self._checks_group = QGroupBox("Verificações automáticas")
        self._checks_layout = QVBoxLayout(self._checks_group)
        self._checks_layout.setSpacing(4)
        self._pending_lbl = QLabel("  Verificando pré-requisitos, aguarde...")
        self._checks_layout.addWidget(self._pending_lbl)
        root.addWidget(self._checks_group)

        # Resumo
        self._summary_lbl = QLabel("")
        self._summary_lbl.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(11)
        self._summary_lbl.setFont(font)
        root.addWidget(self._summary_lbl)

        root.addStretch()

        # Botões
        btn_row = QHBoxLayout()

        self._btn_back = QPushButton(self._back_label)
        self._btn_back.clicked.connect(self._do_back)

        self._btn_rerun = QPushButton("Repetir verificações")
        self._btn_rerun.setEnabled(False)
        self._btn_rerun.clicked.connect(self._run_diagnostics)

        self._btn_start = QPushButton("✔  Iniciar Triagem")
        self._btn_start.setEnabled(False)
        self._btn_start.setStyleSheet(
            "QPushButton:enabled { background: #2e7d32; color: white; font-weight: bold; }"
        )
        self._btn_start.clicked.connect(self._do_start)

        btn_row.addWidget(self._btn_back)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_rerun)
        btn_row.addWidget(self._btn_start)
        root.addLayout(btn_row)

    # ------------------------------------------------------------- Actions --

    def _run_diagnostics(self):
        self._btn_rerun.setEnabled(False)
        self._btn_start.setEnabled(False)
        self._summary_lbl.setText("")

        # Limpar resultados anteriores
        while self._checks_layout.count():
            item = self._checks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._pending_lbl = QLabel("  Verificando pré-requisitos, aguarde...")
        self._checks_layout.addWidget(self._pending_lbl)

        self._worker = _TrabalhadorDiagnostico(self._session.user_id)
        # Conectar ao sinal correto (result_ready, não finished)
        self._worker.result_ready.connect(self._on_diag_done)
        self._worker.start()

        # Watchdog: se o worker travar (ex: câmera ou porta serial bloqueante),
        # força conclusão após 25 s para que a tela nunca fique presa indefinidamente.
        if self._watchdog is None:
            self._watchdog = QTimer(self)
            self._watchdog.setSingleShot(True)
            self._watchdog.timeout.connect(self._on_diag_timeout)
        self._watchdog.start(25_000)

    def _on_diag_timeout(self):
        """Acionado pelo watchdog quando o worker não termina em 25 s."""
        if self._worker:
            try:
                self._worker.result_ready.disconnect(self._on_diag_done)
            except Exception:
                pass
        report = PreOpReport()
        report.serial_check = CheckResult(
            "Diagnóstico",
            ok=False,
            detail="Tempo limite excedido — câmera ou porta serial sem resposta. Verifique o hardware e tente novamente.",
            elapsed_ms=25_000,
        )
        self._on_diag_done(report)

    def _on_diag_done(self, report: PreOpReport):
        if self._watchdog is not None:
            self._watchdog.stop()
        self._report = report

        # Reconstruir lista de resultados
        while self._checks_layout.count():
            item = self._checks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        checks = [
            report.db_check,
            report.config_check,
            report.model_check,
            report.camera_check,
            report.serial_check,
        ]
        for check in checks:
            if check is not None:
                self._checks_layout.addWidget(_LinhaDiagnostico(check))

        # Resumo — cores claras para legibilidade no tema escuro
        all_ok = report.all_critical_ok()
        color = "#81c784" if all_ok else "#ef9a9a"
        self._summary_lbl.setText(
            f'<span style="color:{color}"><b>{report.summary()}</b></span>'
        )

        self._btn_rerun.setEnabled(True)
        self._btn_start.setEnabled(all_ok)

    def _do_start(self):
        if self._on_start and self._report:
            self._on_start(self._session, self._report)

    def _do_back(self):
        if self._on_back:
            self._on_back()
