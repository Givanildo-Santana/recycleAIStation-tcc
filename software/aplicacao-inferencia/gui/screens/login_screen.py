from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QMessageBox, QDialog,
)
from PySide6.QtCore import Qt
from core.auth.authenticator import (
    verify, change_password,
    InvalidCredentials, AccountLocked, MustChangePassword,
)


class ChangePasswordDialog(QDialog):
    def __init__(self, user_id: int, parent=None):
        super().__init__(parent)
        self.user_id = user_id
        self.setWindowTitle("Redefinição de senha")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Crie uma nova senha para prosseguir:"))

        self._new = QLineEdit(placeholderText="Nova senha")
        self._new.setEchoMode(QLineEdit.Password)
        self._confirm = QLineEdit(placeholderText="Confirme a nova senha")
        self._confirm.setEchoMode(QLineEdit.Password)

        btn = QPushButton("Confirmar")
        btn.clicked.connect(self._save)

        for w in (self._new, self._confirm, btn):
            layout.addWidget(w)

    def _save(self):
        new = self._new.text()
        if len(new) < 6:
            QMessageBox.warning(self, "Erro", "A senha deve ter ao menos 6 caracteres.")
            return
        if new != self._confirm.text():
            QMessageBox.warning(self, "Erro", "As senhas não coincidem.")
            return
        change_password(self.user_id, new)
        QMessageBox.information(self, "Senha redefinida", "Senha alterada com sucesso. Faça login novamente.")
        self.accept()


class LoginScreen(QWidget):
    def __init__(self, on_authenticated):
        super().__init__()
        self._on_authenticated = on_authenticated
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(12)

        layout.addWidget(
            QLabel("<h2>RecycleAI-Station</h2>", alignment=Qt.AlignCenter)
        )

        self._login_field = QLineEdit(placeholderText="Login")
        self._login_field.setFixedWidth(280)

        self._pass_field = QLineEdit(placeholderText="Senha")
        self._pass_field.setEchoMode(QLineEdit.Password)
        self._pass_field.setFixedWidth(280)
        self._pass_field.returnPressed.connect(self._attempt_login)

        self._btn = QPushButton("Entrar")
        self._btn.setFixedWidth(280)
        self._btn.clicked.connect(self._attempt_login)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)

        for w in (self._login_field, self._pass_field, self._btn,
                  self._status):
            layout.addWidget(w, alignment=Qt.AlignCenter)

    def _attempt_login(self):
        login = self._login_field.text().strip()
        password = self._pass_field.text()
        self._pass_field.clear()

        if not login:
            self._status.setText("Informe o login.")
            return

        try:
            session = verify(login, password)
            self._on_authenticated(session)

        except AccountLocked as exc:
            QMessageBox.warning(self, "Conta bloqueada", str(exc))

        except InvalidCredentials:
            self._status.setText("Login ou senha inválidos.")

        except MustChangePassword as exc:
            dlg = ChangePasswordDialog(exc.user_id, parent=self)
            dlg.exec()
            self._status.setText("Senha alterada. Faça login novamente.")
