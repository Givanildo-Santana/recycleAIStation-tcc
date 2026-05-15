from PySide6.QtWidgets import QMainWindow, QStackedWidget
from PySide6.QtGui import QIcon
from core.auth.session import Session
from core.utils.paths import bundle_data_root
from gui.screens.login_screen import LoginScreen
from gui.screens.preop_screen import PreOpScreen
from gui.screens.operation_screen import OperationScreen
from gui.screens.maintenance_screen import MaintenanceScreen


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RecycleAI-Station")
        self.setMinimumSize(1024, 720)

        _icon_path = bundle_data_root() / "app" / "assets" / "icons" / "recycleai_icon.png"
        if _icon_path.exists():
            self.setWindowIcon(QIcon(str(_icon_path)))

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._show_login()

    # ---------------------------------------------------------------- Login --

    def _show_login(self):
        while self._stack.count():
            w = self._stack.widget(0)
            self._stack.removeWidget(w)
            w.deleteLater()

        login = LoginScreen(on_authenticated=self._on_authenticated)
        self._stack.addWidget(login)
        self._stack.setCurrentWidget(login)

    def _on_authenticated(self, session: Session):
        if session.is_maintenance():
            self._show_maintenance(session)
        else:
            self._show_preop(session)

    # ---------------------------------------------------- Manutenção --------

    def _show_maintenance(self, session: Session):
        # Limpa stack antes de (re)criar a tela de manutenção
        while self._stack.count():
            w = self._stack.widget(0)
            self._stack.removeWidget(w)
            w.deleteLater()
        maint = MaintenanceScreen(
            session=session,
            on_back=self._show_login,
            on_to_operation=self._on_maint_to_operation,
        )
        self._stack.addWidget(maint)
        self._stack.setCurrentWidget(maint)

    def _on_maint_to_operation(self, session: Session):
        """Permite que maintenance acesse o fluxo operacional (preop → operação)."""
        self._show_preop(session)

    def _return_to_maintenance(self, session: Session):
        """
        Retorna à tela de manutenção a partir do fluxo operacional.
        Limpa o stack inteiro e recria a MaintenanceScreen com a sessão existente,
        preservando o contexto de autenticação.
        """
        self._show_maintenance(session)

    # ----------------------------------------------------------- Pre-op -----

    def _show_preop(self, session: Session):
        # Perfil maintenance: botão «voltar» retorna à tela de manutenção.
        # Perfil operator: botão «voltar» retorna ao login.
        if session.is_maintenance():
            on_back      = lambda s=session: self._return_to_maintenance(s)
            back_label   = "← Administração"
        else:
            on_back      = self._show_login
            back_label   = "← Sair"

        preop = PreOpScreen(
            session=session,
            on_start=self._on_operation_start,
            on_back=on_back,
            back_label=back_label,
        )
        self._stack.addWidget(preop)
        self._stack.setCurrentWidget(preop)

    def _on_operation_start(self, session: Session, report):
        self._show_operation(session, report)

    # --------------------------------------------------------- Operação -----

    def _show_operation(self, session: Session, preop_report):
        # Perfil maintenance: botão «encerrar» retorna à tela de manutenção.
        # Perfil operator: botão «encerrar» retorna ao login.
        if session.is_maintenance():
            on_back      = lambda s=session: self._return_to_maintenance(s)
            back_label   = "← Administração"
        else:
            on_back      = self._show_login
            back_label   = "← Encerrar"

        op = OperationScreen(
            session=session,
            preop_report=preop_report,
            on_back=on_back,
            back_label=back_label,
        )
        self._stack.addWidget(op)
        self._stack.setCurrentWidget(op)
