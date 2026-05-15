import sys
from pathlib import Path

# Garante que o root do projeto está no sys.path independente de como main.py é chamado.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from db.database import initialize as db_init, close as db_close
from core.auth.authenticator import ensure_admin_user, ensure_operator_user
from core.settings.settings_manager import bootstrap_defaults
from core.detection.model_registry import seed_default as seed_model


def bootstrap():
    print("[BOOT] Inicializando banco de dados...")
    db_init()

    print("[BOOT] Verificando usuário admin inicial...")
    ensure_admin_user()

    print("[BOOT] Verificando usuário operador inicial...")
    ensure_operator_user()

    print("[BOOT] Carregando configurações padrão...")
    bootstrap_defaults()

    print("[BOOT] Registrando modelo operacional padrão...")
    seed_model()

    print("[BOOT] Bootstrap concluído.")


def main():
    bootstrap()

    try:
        from PySide6.QtWidgets import QApplication
        from gui.app_window import AppWindow

        from PySide6.QtGui import QIcon
        from core.utils.paths import bundle_data_root

        app = QApplication(sys.argv)
        app.setApplicationName("RecycleAI-Station")
        app.setOrganizationName("UNIP")

        # ── Tema escuro sistêmico (Fusion + QPalette) ────────────────────────
        # Usa o estilo Fusion para renderização consistente cross-platform,
        # depois substitui a paleta padrão por uma dark palette.
        # Widgets com setStyleSheet() próprio continuam com seus estilos;
        # widgets sem override automático ficam dark pelo palette.
        from PySide6.QtGui import QPalette, QColor
        app.setStyle("Fusion")
        _pal = QPalette()
        _C = QColor
        _pal.setColor(QPalette.ColorRole.Window,          _C(30,  30,  30))
        _pal.setColor(QPalette.ColorRole.WindowText,      _C(220, 220, 220))
        _pal.setColor(QPalette.ColorRole.Base,            _C(37,  37,  37))
        _pal.setColor(QPalette.ColorRole.AlternateBase,   _C(50,  50,  50))
        _pal.setColor(QPalette.ColorRole.ToolTipBase,     _C(37,  37,  37))
        _pal.setColor(QPalette.ColorRole.ToolTipText,     _C(220, 220, 220))
        _pal.setColor(QPalette.ColorRole.Text,            _C(220, 220, 220))
        _pal.setColor(QPalette.ColorRole.Button,          _C(50,  50,  50))
        _pal.setColor(QPalette.ColorRole.ButtonText,      _C(220, 220, 220))
        _pal.setColor(QPalette.ColorRole.BrightText,      _C(255,  80,  80))
        _pal.setColor(QPalette.ColorRole.Highlight,       _C(21, 101, 192))
        _pal.setColor(QPalette.ColorRole.HighlightedText, _C(255, 255, 255))
        _pal.setColor(QPalette.ColorRole.Link,            _C(100, 181, 246))
        _pal.setColor(QPalette.ColorRole.Mid,             _C(80,  80,  80))
        _pal.setColor(QPalette.ColorRole.Shadow,          _C(20,  20,  20))
        _pal.setColor(QPalette.ColorGroup.Disabled,
                      QPalette.ColorRole.Text,            _C(100, 100, 100))
        _pal.setColor(QPalette.ColorGroup.Disabled,
                      QPalette.ColorRole.ButtonText,      _C(100, 100, 100))
        _pal.setColor(QPalette.ColorGroup.Disabled,
                      QPalette.ColorRole.WindowText,      _C(100, 100, 100))
        app.setPalette(_pal)
        # QSS complementar: bordas de tabela/agrupamento e scrollbars
        app.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QTableWidget, QTableView {
                gridline-color: #3a3a3a;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                border: 1px solid #3a3a3a;
                padding: 4px;
            }
            QScrollBar:vertical {
                background: #252525;
                width: 10px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #555;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal {
                background: #252525;
                height: 10px;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background: #555;
                border-radius: 5px;
                min-width: 20px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
            QSplitter::handle { background: #3a3a3a; }
            QToolTip {
                background-color: #2d2d2d;
                color: #dcdcdc;
                border: 1px solid #555;
            }
        """)

        _icon_path = bundle_data_root() / "app" / "assets" / "icons" / "recycleai_icon.png"
        if _icon_path.exists():
            app.setWindowIcon(QIcon(str(_icon_path)))

        window = AppWindow()
        window.show()
        exit_code = app.exec()

    except Exception as exc:
        print(f"[BOOT] Erro fatal ao iniciar GUI: {exc}")
        raise
    finally:
        db_close()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
