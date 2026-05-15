"""
Diálogo de diagnóstico manual de hardware — 3 modos estruturados.

Modos:
  1. Chaves    — leitura de sensores fim de curso, sem acionamento de motor.
  2. Manual    — controle bruto (segurar para mover, sem proteção por chave).
  3. Assistido — controle guiado (para automaticamente no fim de curso correto).

Segurança:
  · DIAG_ENTER enviado ao conectar → Arduino suspende verificarAtuador(),
    eliminando o loop de briga entre DIAG_AxP e a lógica de retorno automático.
  · DIAG_EXIT enviado ao fechar → Arduino ressincroniza estados e retoma
    operação normal sem briga contra a posição atual dos atuadores.
  · PARAR TUDO sempre visível, suspende polling por 1500 ms para garantir
    entrega sem contaminação de comandos de status no canal serial.
  · Modo Manual: segurar para mover (pressed/released) + watchdog de 3 s.
  · Modo Assistido: para automaticamente no fim de curso; timeout de 10 s
    para evitar forçamento mecânico caso a chave não seja detectada.
  · closeEvent() garante DIAG_PARAR_TUDO → DIAG_EXIT antes de fechar.

Threading:
  · _ConexaoWorker(QThread) executa connect() + handshake() em background.
  · GerenciadorSerial.start_monitor(callback) → thread daemon de leitura.
  · _linha_serial = Signal(str) cruza monitor thread → UI thread com segurança.
  · QTimer de polling (chaves 300 ms, status 800 ms) dispara no UI thread.
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QScrollArea, QSplitter, QPlainTextEdit, QWidget, QFrame,
    QTabWidget,
)

from core.auth.session import Session
from core.diagnostics.diag_protocol import (
    ATUADOR_IDS, ATUADOR_CMDS,
    CMD_DIAG_ENTER, CMD_DIAG_EXIT,
    CMD_DIAG_ESTEIRA_ON, CMD_DIAG_ESTEIRA_OFF, CMD_DIAG_PARAR_TUDO,
    CMD_DIAG_CHAVES, CMD_STATUS,
    parse_chaves, parse_actuator_status, hint_posicao,
)

_LOG_MAX_LINES      = 400
_POLL_CHAVES_MS     = 300
_POLL_STATUS_MS     = 800
_STOP_RESUME_MS     = 1500   # pausa de polling após parada
_MODO2_WATCHDOG_MS  = 3000   # tempo máximo de acionamento contínuo (Modo Manual)
_MODO3_TIMEOUT_MS   = 10000  # tempo máximo sem fim de curso (Modo Assistido)

# Linhas de alta frequência suprimidas do log (ruído previsível)
_LOG_SUPPRESS = ("[Cmd] DIAG_CHAVES", "[Cmd] Status")


# ─────────────────────────────────────────────────────────────────────────────
# Worker: conexão serial em background
# ─────────────────────────────────────────────────────────────────────────────

class _ConexaoWorker(QThread):
    """
    Executa connect() + handshake() fora do UI thread.
    Emite (handler | None, detalhe_str) ao terminar.
    """
    result_ready = Signal(object, str)

    def run(self):
        try:
            from core.hardware.serial_handler import GerenciadorSerial
            handler = GerenciadorSerial.from_config()
            if not handler.connect():
                self.result_ready.emit(None, handler.connection_detail)
                return
            response = handler.handshake()
            if response is None:
                handler.stop()
                self.result_ready.emit(
                    None,
                    f"{handler.port}: sem resposta ao PING_RECYCLEAI "
                    "(firmware diferente ou porta serial incorreta)",
                )
                return
            # DECISÃO DE DESIGN: diagnóstico aceita SLAVE_ERROR com aviso e
            # mantém a conexão aberta. O objetivo do diálogo de diagnóstico é
            # justamente inspecionar hardware degradado — bloquear aqui
            # impediria a investigação da causa raiz.
            # Comportamento assimétrico ao pré-op (_eval_handshake em
            # pre_op_check.py), que rejeita SLAVE_ERROR porque triagem
            # de produção exige sistema I2C completo (master + slave).
            if "PONG_RECYCLEAI:SLAVE_ERROR" in response:
                self.result_ready.emit(
                    handler,
                    f"{handler.port}: CONECTADO com AVISO — slave I2C com SLAVE_ERROR. "
                    "Atuadores podem não responder.",
                )
                return
            self.result_ready.emit(handler, f"{handler.port}: Arduino RecycleAI OK")
        except Exception as exc:
            self.result_ready.emit(None, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Componentes visuais compartilhados
# ─────────────────────────────────────────────────────────────────────────────

class _LedIndicator(QLabel):
    _ON  = "color:#4caf50; font-size:22px;"
    _OFF = "color:#555;    font-size:22px;"

    def __init__(self, parent=None):
        super().__init__("●", parent)
        self.set_active(False)

    def set_active(self, active: bool):
        self.setStyleSheet(self._ON if active else self._OFF)
        self.setToolTip("Ativada (LOW)" if active else "Livre (HIGH)")


class _PainelChaves(QWidget):
    """Linha com LEDs de C1/C2 e hint de posição para um atuador."""

    def __init__(self, aid: str, parent=None):
        super().__init__(parent)
        self._build(aid)

    def _build(self, aid: str):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(10)

        lbl = QLabel(f"<b>{aid}</b>")
        lbl.setFixedWidth(32)
        lbl.setStyleSheet("font-size:13px;")
        lay.addWidget(lbl)

        self._led_c1 = _LedIndicator()
        lay.addWidget(self._led_c1)
        lay.addWidget(QLabel("C1 base"))

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color:#444;")
        lay.addWidget(sep)

        self._led_c2 = _LedIndicator()
        lay.addWidget(self._led_c2)
        lay.addWidget(QLabel("C2 topo"))

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color:#444;")
        lay.addWidget(sep2)

        self._hint = QLabel("—")
        self._hint.setMinimumWidth(100)
        self._hint.setStyleSheet("color:#aaa; font-size:11px;")
        lay.addWidget(self._hint)
        lay.addStretch()

    def update_chaves(self, c1: bool, c2: bool):
        self._led_c1.set_active(c1)
        self._led_c2.set_active(c2)
        hint = hint_posicao(c1, c2)
        self._hint.setText(hint)
        if "inconsistente" in hint:
            self._hint.setStyleSheet("color:#f44336; font-size:11px; font-weight:bold;")
        elif "trânsito" in hint:
            self._hint.setStyleSheet("color:#ffa726; font-size:11px;")
        else:
            self._hint.setStyleSheet("color:#81c784; font-size:11px;")


# ─────────────────────────────────────────────────────────────────────────────
# Modo 2 — painel de controle bruto (hold-to-move)
# ─────────────────────────────────────────────────────────────────────────────

class _PainelManual(QWidget):
    """
    Controle bruto de um atuador.
    pressed → envia comando de movimento; released → envia parar.
    """
    mover_iniciado   = Signal(str, str)   # (aid, "avançar"|"retornar")
    parar_solicitado = Signal(str)        # (aid)

    _ESTADO_COLOR = {
        "AVANCANDO": "#42a5f5", "RETORNANDO": "#ffa726",
        "PARADO": "#81c784", "?": "#888",
    }

    def __init__(self, aid: str, parent=None):
        super().__init__(parent)
        self._aid = aid
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(8)

        lbl = QLabel(f"<b>{self._aid}</b>")
        lbl.setFixedWidth(32)
        lbl.setStyleSheet("font-size:13px;")
        lay.addWidget(lbl)

        self._btn_av = QPushButton("↑ Avançar")
        self._btn_av.setFixedWidth(96)
        self._btn_av.setToolTip("Segure para mover — solte para parar")
        self._btn_av.pressed.connect(
            lambda: self.mover_iniciado.emit(self._aid, "avançar")
        )
        self._btn_av.released.connect(
            lambda: self.parar_solicitado.emit(self._aid)
        )
        lay.addWidget(self._btn_av)

        self._btn_ret = QPushButton("↓ Retornar")
        self._btn_ret.setFixedWidth(96)
        self._btn_ret.setToolTip("Segure para mover — solte para parar")
        self._btn_ret.pressed.connect(
            lambda: self.mover_iniciado.emit(self._aid, "retornar")
        )
        self._btn_ret.released.connect(
            lambda: self.parar_solicitado.emit(self._aid)
        )
        lay.addWidget(self._btn_ret)

        self._btn_par = QPushButton("■ Parar")
        self._btn_par.setFixedWidth(76)
        self._btn_par.setStyleSheet(
            "QPushButton{background:#e53935;color:white;border-radius:3px;font-weight:bold;}"
            "QPushButton:hover{background:#c62828;}"
            "QPushButton:disabled{background:#444;color:#777;}"
        )
        self._btn_par.clicked.connect(
            lambda: self.parar_solicitado.emit(self._aid)
        )
        lay.addWidget(self._btn_par)

        self._estado_lbl = QLabel("—")
        self._estado_lbl.setFixedWidth(90)
        self._estado_lbl.setAlignment(Qt.AlignCenter)
        self._estado_lbl.setStyleSheet(
            "border:1px solid #444;border-radius:3px;padding:1px 4px;"
            "font-size:11px;color:#888;"
        )
        lay.addWidget(self._estado_lbl)
        lay.addStretch()

    def set_enabled_controls(self, enabled: bool):
        self._btn_av.setEnabled(enabled)
        self._btn_ret.setEnabled(enabled)
        self._btn_par.setEnabled(enabled)

    def set_estado(self, estado: str):
        color = self._ESTADO_COLOR.get(estado, "#888")
        self._estado_lbl.setText(estado)
        self._estado_lbl.setStyleSheet(
            f"border:1px solid #444;border-radius:3px;padding:1px 4px;"
            f"font-size:11px;color:{color};"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Modo 3 — painel de controle assistido
# ─────────────────────────────────────────────────────────────────────────────

class _PainelAssistido(QWidget):
    """
    Controle guiado: avança/retorna até o fim de curso correto, para sozinho.
    """
    avancar_solicitado  = Signal(str)
    retornar_solicitado = Signal(str)
    parar_solicitado    = Signal(str)

    def __init__(self, aid: str, parent=None):
        super().__init__(parent)
        self._aid = aid
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(8)

        lbl = QLabel(f"<b>{self._aid}</b>")
        lbl.setFixedWidth(32)
        lbl.setStyleSheet("font-size:13px;")
        top.addWidget(lbl)

        self._btn_av = QPushButton("↑ Avançar")
        self._btn_av.setFixedWidth(88)
        self._btn_av.clicked.connect(
            lambda: self.avancar_solicitado.emit(self._aid)
        )
        top.addWidget(self._btn_av)

        self._btn_ret = QPushButton("↓ Retornar")
        self._btn_ret.setFixedWidth(88)
        self._btn_ret.clicked.connect(
            lambda: self.retornar_solicitado.emit(self._aid)
        )
        top.addWidget(self._btn_ret)

        self._btn_par = QPushButton("■ Parar")
        self._btn_par.setFixedWidth(76)
        self._btn_par.setStyleSheet(
            "QPushButton{background:#e53935;color:white;border-radius:3px;font-weight:bold;}"
            "QPushButton:hover{background:#c62828;}"
            "QPushButton:disabled{background:#444;color:#777;}"
        )
        self._btn_par.clicked.connect(
            lambda: self.parar_solicitado.emit(self._aid)
        )
        top.addWidget(self._btn_par)

        self._chave_lbl = QLabel("—")
        self._chave_lbl.setFixedWidth(100)
        self._chave_lbl.setAlignment(Qt.AlignCenter)
        self._chave_lbl.setStyleSheet("font-size:11px; color:#888;")
        top.addWidget(self._chave_lbl)
        top.addStretch()

        root.addLayout(top)

        self._prog_lbl = QLabel("")
        self._prog_lbl.setStyleSheet(
            "color:#90caf9; font-size:10px; padding-left:44px;"
        )
        root.addWidget(self._prog_lbl)

    def set_enabled_controls(self, enabled: bool):
        self._btn_av.setEnabled(enabled)
        self._btn_ret.setEnabled(enabled)
        self._btn_par.setEnabled(enabled)

    def update_chaves(self, c1: bool, c2: bool):
        hint = hint_posicao(c1, c2)
        self._chave_lbl.setText(hint)
        if "inconsistente" in hint:
            self._chave_lbl.setStyleSheet(
                "font-size:11px; color:#f44336; font-weight:bold;"
            )
        elif "trânsito" in hint:
            self._chave_lbl.setStyleSheet("font-size:11px; color:#ffa726;")
        elif hint in ("na base", "no topo"):
            self._chave_lbl.setStyleSheet("font-size:11px; color:#81c784;")
        else:
            self._chave_lbl.setStyleSheet("font-size:11px; color:#888;")

    def set_progresso(self, msg: str):
        self._prog_lbl.setText(msg)
        if "⚠" in msg:
            self._prog_lbl.setStyleSheet(
                "color:#ef9a9a; font-size:10px; padding-left:44px;"
            )
        elif "✓" in msg:
            self._prog_lbl.setStyleSheet(
                "color:#81c784; font-size:10px; padding-left:44px;"
            )
        else:
            self._prog_lbl.setStyleSheet(
                "color:#90caf9; font-size:10px; padding-left:44px;"
            )

    def clear_progresso(self):
        self._prog_lbl.setText("")
        self._prog_lbl.setStyleSheet(
            "color:#90caf9; font-size:10px; padding-left:44px;"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Diálogo principal
# ─────────────────────────────────────────────────────────────────────────────

class HardwareDiagDialog(QDialog):
    """
    Modal de diagnóstico manual — 3 modos: Chaves, Manual, Assistido.

    Abre já iniciando conexão serial em background.
    DIAG_ENTER é enviado ao conectar; DIAG_EXIT + DIAG_PARAR_TUDO são
    enviados ao fechar, garantindo retorno seguro ao fluxo operacional.
    """

    _linha_serial = Signal(str)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self._session  = session
        self._handler  = None
        self._conn_worker: _ConexaoWorker | None = None

        # Painéis dos 3 modos
        self._paineis_chaves:    dict[str, _PainelChaves]    = {}
        self._paineis_manual:    dict[str, _PainelManual]    = {}
        self._paineis_assistido: dict[str, _PainelAssistido] = {}

        # Timers de polling (parados por padrão)
        self._chaves_timer = QTimer(self)
        self._chaves_timer.timeout.connect(self._poll_chaves)
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._poll_status)

        # Watchdogs do Modo 2 — 1 por atuador, single-shot 3 s
        self._modo2_watchdog: dict[str, QTimer] = {}
        for aid in ATUADOR_IDS:
            t = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(lambda a=aid: self._modo2_timeout(a))
            self._modo2_watchdog[aid] = t

        # Timeouts do Modo 3 — 1 por atuador, single-shot 10 s
        self._modo3_timeout_timer: dict[str, QTimer] = {}
        for aid in ATUADOR_IDS:
            t = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(lambda a=aid: self._modo3_timeout(a))
            self._modo3_timeout_timer[aid] = t

        # Alvo de movimento no Modo 3: None | "base" | "topo"
        self._modo3_alvo: dict[str, str | None] = {a: None for a in ATUADOR_IDS}

        self.setWindowTitle("Diagnóstico Manual de Hardware — RecycleAI-Station")
        self.setModal(True)
        self.setMinimumSize(1060, 640)

        self._build_ui()
        self._linha_serial.connect(self._on_linha_serial)
        self._audit("DIAG_MANUAL_ENTER", "Diagnóstico manual iniciado")
        self._iniciar_conexao()

    # ── Construção da UI ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        root.addWidget(self._build_header())
        root.addWidget(self._build_conn_bar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setContentsMargins(8, 8, 8, 8)
        splitter.setHandleWidth(6)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_log_panel())
        splitter.setSizes([560, 460])
        root.addWidget(splitter, stretch=1)

        root.addWidget(self._build_footer())

    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background:#4a148c;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(20, 12, 20, 12)
        title = QLabel("🔧  Diagnóstico Manual de Hardware")
        title.setStyleSheet("color:white; font-size:15px; font-weight:bold;")
        lay.addWidget(title)
        lay.addStretch()
        note = QLabel("⚠  Modo diagnóstico — operações de triagem suspensas")
        note.setStyleSheet("color:#ce93d8; font-size:11px;")
        lay.addWidget(note)
        return w

    def _build_conn_bar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet("background:#1a1a2e; border-bottom:1px solid #333;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 8, 16, 8)
        lay.setSpacing(10)

        self._conn_status_lbl = QLabel("⏳  Conectando…")
        self._conn_status_lbl.setStyleSheet("color:#90caf9; font-size:12px;")
        lay.addWidget(self._conn_status_lbl)
        lay.addStretch()

        self._btn_reconectar = QPushButton("🔌  Reconectar")
        self._btn_reconectar.setEnabled(False)
        self._btn_reconectar.clicked.connect(self._iniciar_conexao)
        lay.addWidget(self._btn_reconectar)

        self._btn_desconectar = QPushButton("⏏  Desconectar")
        self._btn_desconectar.setEnabled(False)
        self._btn_desconectar.clicked.connect(self._desconectar)
        lay.addWidget(self._btn_desconectar)

        self._btn_parar_tudo = QPushButton("🛑  PARAR TUDO")
        self._btn_parar_tudo.setEnabled(False)
        self._btn_parar_tudo.setStyleSheet(
            "QPushButton{background:#c62828;color:white;font-weight:bold;"
            "  padding:6px 16px;border-radius:4px;font-size:13px;}"
            "QPushButton:hover{background:#b71c1c;}"
            "QPushButton:disabled{background:#444;color:#777;}"
        )
        self._btn_parar_tudo.clicked.connect(self._do_parar_tudo)
        lay.addWidget(self._btn_parar_tudo)
        return bar

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_tab_chaves(),    "① Chaves")
        self._tabs.addTab(self._build_tab_manual(),    "② Manual")
        self._tabs.addTab(self._build_tab_assistido(), "③ Assistido")
        lay.addWidget(self._tabs)
        return container

    # ── Tab 1: Chaves ─────────────────────────────────────────────────────────

    def _build_tab_chaves(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)
        lay.setContentsMargins(12, 12, 12, 12)

        intro = QLabel(
            "<b>Modo Leitura de Sensores</b><br>"
            "Pressione as chaves manualmente e observe a transição de estado em tempo real.<br>"
            "Nenhum motor é acionado neste modo."
        )
        intro.setStyleSheet(
            "background:#1b2838; color:#90caf9; padding:10px; "
            "border-radius:4px; font-size:11px;"
        )
        intro.setWordWrap(True)
        lay.addWidget(intro)

        gb = QGroupBox("Chaves Fim de Curso (polling 300 ms)")
        gb_lay = QVBoxLayout(gb)
        gb_lay.setSpacing(10)

        for aid in ATUADOR_IDS:
            painel = _PainelChaves(aid)
            self._paineis_chaves[aid] = painel
            gb_lay.addWidget(painel)

            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color:#333;")
            gb_lay.addWidget(sep)

        legenda = QLabel(
            "● verde = ativada (LOW / pressionada)   ● cinza = livre (HIGH)\n"
            "C1 = chave de base (repouso)   C2 = chave de topo (fim de curso avançado)"
        )
        legenda.setStyleSheet("color:#888; font-size:10px;")
        gb_lay.addWidget(legenda)

        lay.addWidget(gb)
        lay.addStretch()
        scroll.setWidget(w)
        return scroll

    # ── Tab 2: Manual ─────────────────────────────────────────────────────────

    def _build_tab_manual(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)
        lay.setContentsMargins(12, 12, 12, 12)

        aviso = QLabel(
            "⚠  MODO BRUTO — SEM PROTEÇÃO OBRIGATÓRIA POR FIM DE CURSO\n"
            "Segure o botão de direção para mover — solte para parar imediatamente.\n"
            "Parada automática de segurança após 3 s de acionamento contínuo."
        )
        aviso.setStyleSheet(
            "background:#b71c1c; color:white; padding:10px; "
            "border-radius:4px; font-size:11px; font-weight:bold;"
        )
        aviso.setWordWrap(True)
        lay.addWidget(aviso)

        esteira_gb = QGroupBox("Esteira")
        esteira_lay = QHBoxLayout(esteira_gb)

        self._btn_esteira_on = QPushButton("⚙  Ligar")
        self._btn_esteira_on.setEnabled(False)
        self._btn_esteira_on.clicked.connect(
            lambda: self._send_cmd(CMD_DIAG_ESTEIRA_ON)
        )
        esteira_lay.addWidget(self._btn_esteira_on)

        self._btn_esteira_off = QPushButton("⏹  Desligar")
        self._btn_esteira_off.setEnabled(False)
        self._btn_esteira_off.setStyleSheet(
            "QPushButton{background:#e53935;color:white;border-radius:3px;}"
            "QPushButton:hover{background:#c62828;}"
            "QPushButton:disabled{background:#444;color:#777;}"
        )
        self._btn_esteira_off.clicked.connect(
            lambda: self._send_cmd(CMD_DIAG_ESTEIRA_OFF)
        )
        esteira_lay.addWidget(self._btn_esteira_off)
        esteira_lay.addStretch()
        lay.addWidget(esteira_gb)

        atu_gb = QGroupBox("Atuadores — Controle Bruto  (segure para mover)")
        atu_lay = QVBoxLayout(atu_gb)
        atu_lay.setSpacing(6)

        for aid in ATUADOR_IDS:
            painel = _PainelManual(aid)
            painel.mover_iniciado.connect(self._modo2_iniciar)
            painel.parar_solicitado.connect(self._modo2_parar)
            self._paineis_manual[aid] = painel
            atu_lay.addWidget(painel)

        lay.addWidget(atu_gb)
        lay.addStretch()
        scroll.setWidget(w)
        return scroll

    # ── Tab 3: Assistido ──────────────────────────────────────────────────────

    def _build_tab_assistido(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)
        lay.setContentsMargins(12, 12, 12, 12)

        intro = QLabel(
            "<b>Modo Assistido</b><br>"
            "Avança até o fim de curso de topo (C2) ou retorna até a base (C1), "
            "parando automaticamente ao detectar a chave correta.<br>"
            "Timeout de segurança: 10 s por movimento — para se a chave não for detectada."
        )
        intro.setStyleSheet(
            "background:#1b2838; color:#90caf9; padding:10px; "
            "border-radius:4px; font-size:11px;"
        )
        intro.setWordWrap(True)
        lay.addWidget(intro)

        gb = QGroupBox("Atuadores — Controle Assistido")
        gb_lay = QVBoxLayout(gb)
        gb_lay.setSpacing(6)

        for aid in ATUADOR_IDS:
            painel = _PainelAssistido(aid)
            painel.avancar_solicitado.connect(self._modo3_avancar)
            painel.retornar_solicitado.connect(self._modo3_retornar)
            painel.parar_solicitado.connect(self._modo3_cancelar_e_parar)
            self._paineis_assistido[aid] = painel
            gb_lay.addWidget(painel)

            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color:#333;")
            gb_lay.addWidget(sep)

        lay.addWidget(gb)
        lay.addStretch()
        scroll.setWidget(w)
        return scroll

    # ── Log ───────────────────────────────────────────────────────────────────

    def _build_log_panel(self) -> QGroupBox:
        gb = QGroupBox("Log Serial")
        lay = QVBoxLayout(gb)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(_LOG_MAX_LINES)
        self._log.setStyleSheet(
            "background:#0d1117; color:#c9d1d9; "
            "border:1px solid #333; border-radius:4px;"
        )
        mono = QFont("Consolas", 10)
        mono.setStyleHint(QFont.Monospace)
        self._log.setFont(mono)
        lay.addWidget(self._log)
        return gb

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        footer.setStyleSheet("border-top:1px solid #333; background:#111;")
        lay = QHBoxLayout(footer)
        lay.setContentsMargins(16, 8, 16, 8)
        note = QLabel(
            "⚠  Parada segura sempre executada ao encerrar. "
            "Não feche durante movimento de atuadores."
        )
        note.setStyleSheet("color:#e57373; font-size:10px;")
        lay.addWidget(note)
        lay.addStretch()
        btn = QPushButton("✕  Encerrar")
        btn.setStyleSheet(
            "QPushButton{background:#37474f;color:white;padding:6px 18px;"
            "  border-radius:4px;}"
            "QPushButton:hover{background:#546e7a;}"
        )
        btn.clicked.connect(self.close)
        lay.addWidget(btn)
        return footer

    # ── Conexão serial ────────────────────────────────────────────────────────

    def _iniciar_conexao(self):
        if self._conn_worker and self._conn_worker.isRunning():
            return
        self._desconectar(silent=True)
        self._set_conn_ui(connecting=True)
        self._log_line(">>> Iniciando conexão serial…")
        self._conn_worker = _ConexaoWorker(self)
        self._conn_worker.result_ready.connect(self._on_conn_result)
        self._conn_worker.start()

    def _on_conn_result(self, handler, detail: str):
        if handler is None:
            self._set_conn_ui(connected=False, detail=f"❌  {detail}")
            self._log_line(f"[FALHA] {detail}")
            return

        self._handler = handler
        self._set_conn_ui(connected=True, detail=f"✅  {detail}")
        self._log_line(f"[OK] {detail}")

        self._handler.start_monitor(self._linha_serial.emit)

        # Entra em modo diagnóstico: Arduino suspende verificarAtuador()
        self._send_cmd(CMD_DIAG_ENTER)

        self._chaves_timer.start(_POLL_CHAVES_MS)
        self._status_timer.start(_POLL_STATUS_MS)

    def _desconectar(self, silent: bool = False):
        self._chaves_timer.stop()
        self._status_timer.stop()
        for aid in ATUADOR_IDS:
            self._modo2_watchdog[aid].stop()
            self._modo3_timeout_timer[aid].stop()
            self._modo3_alvo[aid] = None
        if self._handler:
            try:
                self._handler.stop()
            except Exception:
                pass
            self._handler = None
        if not silent:
            self._set_conn_ui(connected=False, detail="Desconectado.")
            self._log_line(">>> Desconectado.")

    # ── Polling ───────────────────────────────────────────────────────────────

    def _poll_chaves(self):
        if self._handler and self._handler.is_connected():
            try:
                self._handler.send(CMD_DIAG_CHAVES)
            except Exception:
                pass

    def _poll_status(self):
        if self._handler and self._handler.is_connected():
            try:
                self._handler.send(CMD_STATUS)
            except Exception:
                pass

    def _pausar_timers(self):
        self._chaves_timer.stop()
        self._status_timer.stop()

    def _retomar_timers(self):
        if self._handler and self._handler.is_connected():
            self._chaves_timer.start(_POLL_CHAVES_MS)
            self._status_timer.start(_POLL_STATUS_MS)

    # ── Recepção serial ───────────────────────────────────────────────────────

    def _on_linha_serial(self, line: str):
        # Chaves: processa silenciosamente, não loga
        chaves = parse_chaves(line)
        if chaves:
            self._on_chaves_update(chaves)
            return

        # Status de atuadores: atualiza painéis do Modo 2
        estados = parse_actuator_status(line)
        if estados:
            for aid, estado in estados.items():
                if aid in self._paineis_manual:
                    self._paineis_manual[aid].set_estado(estado)

        # Suprime linhas de alta frequência para não poluir o log
        if any(s in line for s in _LOG_SUPPRESS):
            return

        self._log_line(line)

    def _on_chaves_update(self, chaves: dict[str, bool]):
        for aid in ATUADOR_IDS:
            c1 = chaves.get(f"{aid}C1", False)
            c2 = chaves.get(f"{aid}C2", False)

            if aid in self._paineis_chaves:
                self._paineis_chaves[aid].update_chaves(c1, c2)
            if aid in self._paineis_assistido:
                self._paineis_assistido[aid].update_chaves(c1, c2)

            # Verifica chegada ao fim de curso no Modo 3
            alvo = self._modo3_alvo.get(aid)
            if alvo == "topo" and c2:
                self._modo3_chegou(aid, "topo")
            elif alvo == "base" and c1:
                self._modo3_chegou(aid, "base")

    # ── Envio genérico ────────────────────────────────────────────────────────

    def _send_cmd(self, cmd: str):
        if not self._handler or not self._handler.is_connected():
            return
        try:
            self._handler.send(cmd)
            self._log_line(f">>> {cmd}")
        except Exception as exc:
            self._log_line(f"[ERRO] {cmd}: {exc}")

    # ── Parada com prioridade ─────────────────────────────────────────────────

    def _parar_atuador(self, aid: str):
        """Para atuador individual: suspende polling por _STOP_RESUME_MS."""
        self._pausar_timers()
        self._send_cmd(ATUADOR_CMDS[aid]["parar"])
        self._audit("DIAG_MANUAL_CMD", f"Parar: {aid}")
        QTimer.singleShot(_STOP_RESUME_MS, self._retomar_timers)

    def _do_parar_tudo(self):
        """Para tudo com prioridade máxima — suspende polling por _STOP_RESUME_MS."""
        self._pausar_timers()
        for aid in ATUADOR_IDS:
            self._modo2_watchdog[aid].stop()
            self._modo3_timeout_timer[aid].stop()
            self._modo3_alvo[aid] = None
            if aid in self._paineis_assistido:
                self._paineis_assistido[aid].clear_progresso()
        self._send_cmd(CMD_DIAG_PARAR_TUDO)
        self._audit("DIAG_MANUAL_CMD", "PARAR TUDO executado")
        QTimer.singleShot(_STOP_RESUME_MS, self._retomar_timers)

    # ── Modo 2: Manual (hold-to-move) ─────────────────────────────────────────

    def _modo2_iniciar(self, aid: str, direcao: str):
        self._pausar_timers()
        self._send_cmd(ATUADOR_CMDS[aid][direcao])
        self._audit("DIAG_MANUAL_CMD", f"Manual: {aid} {direcao}")
        self._modo2_watchdog[aid].start(_MODO2_WATCHDOG_MS)

    def _modo2_parar(self, aid: str):
        self._modo2_watchdog[aid].stop()
        self._parar_atuador(aid)

    def _modo2_timeout(self, aid: str):
        self._log_line(
            f"[⚠ WATCHDOG] {aid}: parada automática após "
            f"{_MODO2_WATCHDOG_MS // 1000} s de acionamento contínuo"
        )
        self._parar_atuador(aid)

    # ── Modo 3: Assistido ─────────────────────────────────────────────────────

    def _modo3_avancar(self, aid: str):
        if self._modo3_alvo[aid] is not None:
            self._log_line(
                f"[Assistido] {aid}: movimento em andamento — cancele antes."
            )
            return
        self._send_cmd(ATUADOR_CMDS[aid]["avançar"])
        self._modo3_alvo[aid] = "topo"
        self._modo3_timeout_timer[aid].start(_MODO3_TIMEOUT_MS)
        self._paineis_assistido[aid].set_progresso(
            f"Avançando… aguardando C2 (topo)"
        )
        self._audit("DIAG_MANUAL_CMD", f"Assistido: {aid} avançar")

    def _modo3_retornar(self, aid: str):
        if self._modo3_alvo[aid] is not None:
            self._log_line(
                f"[Assistido] {aid}: movimento em andamento — cancele antes."
            )
            return
        self._send_cmd(ATUADOR_CMDS[aid]["retornar"])
        self._modo3_alvo[aid] = "base"
        self._modo3_timeout_timer[aid].start(_MODO3_TIMEOUT_MS)
        self._paineis_assistido[aid].set_progresso(
            f"Retornando… aguardando C1 (base)"
        )
        self._audit("DIAG_MANUAL_CMD", f"Assistido: {aid} retornar")

    def _modo3_chegou(self, aid: str, destino: str):
        self._modo3_timeout_timer[aid].stop()
        self._modo3_alvo[aid] = None   # limpa ANTES de parar (evita re-trigger)
        self._parar_atuador(aid)
        msg = f"✓ Chegou a: {destino} — parando"
        self._log_line(f"[Assistido] {aid}: {msg}")
        self._paineis_assistido[aid].set_progresso(msg)

    def _modo3_timeout(self, aid: str):
        self._modo3_alvo[aid] = None
        self._parar_atuador(aid)
        msg = (
            f"⚠ Timeout {_MODO3_TIMEOUT_MS // 1000} s — "
            "chave não detectada — parada de segurança"
        )
        self._log_line(f"[Assistido] {aid}: {msg}")
        self._paineis_assistido[aid].set_progresso(msg)

    def _modo3_cancelar_e_parar(self, aid: str):
        self._modo3_timeout_timer[aid].stop()
        self._modo3_alvo[aid] = None
        self._paineis_assistido[aid].clear_progresso()
        self._parar_atuador(aid)

    # ── Estado da UI ──────────────────────────────────────────────────────────

    def _set_conn_ui(self, *, connecting: bool = False,
                     connected: bool = False, detail: str = ""):
        if connecting:
            self._conn_status_lbl.setText("⏳  Conectando…")
            self._conn_status_lbl.setStyleSheet("color:#90caf9; font-size:12px;")
            self._btn_reconectar.setEnabled(False)
            self._btn_desconectar.setEnabled(False)
        else:
            self._conn_status_lbl.setText(
                detail or ("Conectado" if connected else "Não conectado")
            )
            color = "#a5d6a7" if connected else "#ef9a9a"
            self._conn_status_lbl.setStyleSheet(f"color:{color}; font-size:12px;")
            self._btn_reconectar.setEnabled(True)
            self._btn_desconectar.setEnabled(connected)

        ctrl_ok = connected and not connecting
        self._btn_parar_tudo.setEnabled(ctrl_ok)
        self._btn_esteira_on.setEnabled(ctrl_ok)
        self._btn_esteira_off.setEnabled(ctrl_ok)
        for p in self._paineis_manual.values():
            p.set_enabled_controls(ctrl_ok)
        for p in self._paineis_assistido.values():
            p.set_enabled_controls(ctrl_ok)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log_line(self, line: str):
        self._log.appendPlainText(line)

    # ── Parada segura ─────────────────────────────────────────────────────────

    def _parada_segura(self):
        self._chaves_timer.stop()
        self._status_timer.stop()
        for aid in ATUADOR_IDS:
            self._modo2_watchdog[aid].stop()
            self._modo3_timeout_timer[aid].stop()
        if self._handler and self._handler.is_connected():
            try:
                self._handler.send(CMD_DIAG_PARAR_TUDO)
                time.sleep(0.4)   # dá tempo ao Arduino processar PARAR_TUDO
                self._handler.send(CMD_DIAG_EXIT)
                time.sleep(0.1)   # dá tempo ao DIAG_EXIT ser lido antes de fechar
            except Exception:
                pass
            try:
                self._handler.stop()
            except Exception:
                pass
        self._handler = None

    # ── Auditoria ─────────────────────────────────────────────────────────────

    def _audit(self, event: str, description: str):
        try:
            from db.repositories import audit_repo
            audit_repo.record(
                event, description=description, user_id=self._session.user_id
            )
        except Exception:
            pass

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._conn_worker and self._conn_worker.isRunning():
            self._conn_worker.wait(4000)
        self._parada_segura()
        self._audit("DIAG_MANUAL_EXIT", "Diagnóstico manual encerrado")
        super().closeEvent(event)
