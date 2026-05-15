"""
Tela operacional de triagem de residuos.

Fluxo: PreOpScreen → OperationScreen → (encerrar) → LoginScreen

_TrabalhadorInferencia (QThread):
  - loop de câmera + inferência sem bloquear a GUI
  - consume exclusivamente a camada adaptadora (core/detection/legacy_adapter)
  - sem manipulação de sys.path ou imports diretos do pipeline legado
  - RoiTimer inline: confirma label após N segundos consecutivos sem mudança
  - Persiste detecção confirmada no banco e envia ao Arduino via serial
  - Encerramento seguro: libera câmera e fecha serial ao parar
  - Coleta métricas técnicas de inferência quando perfil=maintenance

OperationScreen (QWidget):
  - Exibe feed de câmera em tempo real
  - Painel de detecção: label atual, label confirmada, contagem por classe
  - Linha de status: modelo ativo, conexão serial
  - Botões: Iniciar / Parar / Encerrar
  - Auditoria de início, fim e erros operacionais
  - Sessão operacional formal (op_sessions) via session_manager
  - Métricas técnicas extras para perfil maintenance
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from core.auth.session import Session
from core.diagnostics.pre_op_check import PreOpReport


# ─────────────────────────────────────────────────────────────────────────────
# Worker de inferência — roda em QThread separado
# ─────────────────────────────────────────────────────────────────────────────

class _TrabalhadorInferencia(QThread):
    frame_ready           = Signal(QImage)
    detection_updated     = Signal(str, float)   # label atual, confiança
    rotulo_confirmado       = Signal(str, float)   # label confirmada, confiança
    error_occurred        = Signal(str)
    worker_stopped        = Signal()
    # Emitido ao encerrar com dict de métricas pré-computadas (apenas maintenance)
    metrics_ready         = Signal(object)
    # Estado da conexão serial (conectado: bool, detalhe: str)
    serial_status_changed = Signal(bool, str)
    # Linha de log para painel de observabilidade (apenas maintenance)
    entrada_log             = Signal(str)

    def __init__(self, session: Session, has_serial: bool, parent=None):
        super().__init__(parent)
        self._session    = session
        self._has_serial = has_serial
        self._stop_flag  = False

    def request_stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        import cv2

        # Adaptador centraliza todo o acesso ao pipeline legado
        from core.detection.legacy_adapter import run_frame, annotate_frame
        from core.detection import model_loader
        from core.detection.model_registry import get_active_classes
        from core.settings import settings_manager

        # ── Parâmetros do banco ──────────────────────────────────────────────
        device     = settings_manager.get("realtime.device")  or "cpu"
        source     = settings_manager.get("realtime.source")  or "0"
        conf_thres = float(settings_manager.get("realtime.conf_thres") or 0.50)
        iou_thres  = float(settings_manager.get("realtime.iou_thres")  or 0.45)
        img_size   = int(settings_manager.get("training.img_size")     or 640)
        roi_x1     = int(settings_manager.get("realtime.roi_x_start")  or 100)
        roi_x2     = int(settings_manager.get("realtime.roi_x_end")    or 480)
        roi_y1     = int(settings_manager.get("realtime.roi_y_start")  or 70)
        roi_y2     = int(settings_manager.get("realtime.roi_y_end")    or 450)
        timer_secs = float(settings_manager.get("roi_timer.seconds")   or 3.0)
        roi        = (roi_x1, roi_y1, roi_x2, roi_y2)

        # ── Classes do modelo ativo ──────────────────────────────────────────
        names = get_active_classes()
        if not names:
            self.error_occurred.emit("Nenhuma classe encontrada no modelo ativo. Verifique o painel de Administração.")
            self.worker_stopped.emit()
            return

        # ── Carregar modelo via model_registry ───────────────────────────────
        try:
            model = model_loader.load_active(device)
        except Exception as exc:
            self.error_occurred.emit(f"Falha ao carregar modelo: {exc}")
            self.worker_stopped.emit()
            return

        # ── Abrir câmera (com fallback de backend e timeout) ─────────────────
        # open_camera() tenta backend padrão (MSMF) primeiro. No Windows, se
        # o backend padrão travar ou falhar, tenta CAP_DSHOW automaticamente
        # — compatível com webcams USB cujo driver não responde ao MSMF.
        # Cada tentativa tem timeout de thread; total ≤ 10 s.
        src = int(source) if source.isdigit() else source
        from core.diagnostics.camera_open import open_camera as _open_camera
        cap, _cam_detail = _open_camera(src, timeout_s=10.0)
        if cap is None:
            self.error_occurred.emit(
                f"Câmera não disponível (fonte: {source}): {_cam_detail}"
            )
            self.worker_stopped.emit()
            return

        # ── Serial — opcional ─────────────────────────────────────────────────
        serial_handler = None
        if self._has_serial:
            try:
                from core.hardware.serial_handler import GerenciadorSerial
                serial_handler = GerenciadorSerial.from_config()
                if serial_handler.connect():
                    # Envia atrasos configurados ANTES de start_monitor() para
                    # evitar que a thread de monitoramento consuma o ACK (CONF_OK).
                    from core.operation.configuracao_esteira import (
                        enviar_atrasos_arduino as _enviar_atrasos,
                    )
                    _ok_cfg, _det_cfg = _enviar_atrasos(serial_handler)
                    self.entrada_log.emit(
                        f"[Config] Atrasos "
                        f"{'aplicados' if _ok_cfg else 'padrão (firmware)'}"
                        f": {_det_cfg}"
                    )
                    serial_handler.start_monitor(
                        lambda line: self.entrada_log.emit(f"[Arduino] {line}")
                    )
                    self.serial_status_changed.emit(
                        True, f"Serial conectada: {serial_handler.port}"
                    )
                    self.entrada_log.emit(f"[Serial] Conectado em {serial_handler.port}")
                else:
                    detail = serial_handler.connection_detail
                    serial_handler = None
                    self.serial_status_changed.emit(False, f"Serial indisponível: {detail}")
                    self.entrada_log.emit(f"[Serial] Falha na conexão: {detail}")
            except Exception as _e:
                serial_handler = None
                self.serial_status_changed.emit(False, f"Erro serial: {_e}")

        # ── Estado de reconexão serial ────────────────────────────────────────
        _RECONNECT_INTERVAL = 30.0
        _last_reconnect_t   = -_RECONNECT_INTERVAL  # permite tentativa imediata

        # ── Estado do RoiTimer ────────────────────────────────────────────────
        _cur_label:     str       = "NONE"
        _cur_start:     float     = time.monotonic()
        _confirmed_set: set[str]  = set()

        # ── Estado de ciclo de item (NAO_IDENTIFICADO) ────────────────────────
        # Detecta quando um item entra e sai do ROI sem gerar confirmação.
        _cicloAtivo:    bool  = False  # True enquanto há item em trânsito
        _cicloConfirmado: bool  = False  # True se o ciclo gerou confirmação
        _cicloInicioT: float = 0.0   # início do ciclo (monotonic)
        _semDeteccaoDesde:      float = 0.0   # quando roi_label voltou a NONE
        _DEBOUNCE_ROI_VAZIA_S        = 0.8   # NONE por ≥ 0.8 s encerra o ciclo
        _DURACAO_MIN_CICLO_S           = 1.0   # ciclo < 1.0 s → ruído → ignorar

        # ── Coleta de métricas (apenas perfil maintenance) ────────────────────
        _is_maintenance  = self._session.is_maintenance()
        _infer_times:    list[float] = []  # tempo de cada inferência (s)
        _conf_all:       list[float] = []  # confiança de cada detecção no ROI
        _frame_count:    int         = 0
        _session_t0:     float       = time.monotonic()

        try:
            while not self._stop_flag:
                ret, frame = cap.read()
                if not ret:
                    self.error_occurred.emit("A câmera interrompeu a transmissão. Verifique a conexão.")
                    break

                # Inferência + ROI filter via adaptador (sem sys.path aqui)
                _t_infer = time.monotonic()
                try:
                    result = run_frame(
                        model, frame,
                        device=device,
                        img_size=img_size,
                        conf_thres=conf_thres,
                        iou_thres=iou_thres,
                        names=names,
                        roi=roi,
                    )
                except Exception as _exc:
                    self.error_occurred.emit(f"Falha na inferência: {_exc}")
                    break
                _frame_count += 1

                # Coleta métricas de inferência (maintenance)
                if _is_maintenance:
                    _infer_times.append(time.monotonic() - _t_infer)
                    if result.roi_label != "NONE" and result.roi_conf > 0:
                        _conf_all.append(result.roi_conf)

                # Anotação visual (ROI rect + caixas) via adaptador
                annotate_frame(frame, result, roi)

                # Emitir frame como QImage (BGR → RGB, cópia thread-safe)
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(
                    rgb.data, w, h, ch * w, QImage.Format.Format_RGB888
                ).copy()
                self.frame_ready.emit(qimg)
                self.detection_updated.emit(result.roi_label, result.roi_conf)

                # ── RoiTimer + ciclo de item ──────────────────────────────────
                _now = time.monotonic()

                # Label timer: confirma quando o mesmo label persiste por timer_secs
                if result.roi_label != _cur_label:
                    _cur_label = result.roi_label
                    _cur_start = _now
                    _confirmed_set.clear()
                elif (
                    result.roi_label != "NONE"
                    and result.roi_label not in _confirmed_set
                    and _now - _cur_start >= timer_secs
                ):
                    _confirmed_set.add(result.roi_label)
                    _cicloConfirmado = True
                    self.rotulo_confirmado.emit(result.roi_label, result.roi_conf)
                    self.entrada_log.emit(
                        f"[ROI] Confirmado: {result.roi_label} "
                        f"({result.roi_conf:.1%})"
                    )

                    # Persistência
                    try:
                        from db.repositories import detection_repo
                        detection_repo.insert(
                            label=result.roi_label,
                            confidence=result.roi_conf,
                            session_id=self._session.session_id,
                            user_id=self._session.user_id,
                        )
                    except Exception:
                        pass

                    # Comando para Arduino
                    if serial_handler:
                        try:
                            serial_handler.send(result.roi_label)
                            self.entrada_log.emit(
                                f"[Serial] Enviado: {result.roi_label}"
                            )
                        except Exception as _exc:
                            self.entrada_log.emit(
                                f"[Serial] Erro ao enviar '{result.roi_label}': {_exc}"
                            )
                            serial_handler.stop()
                            serial_handler = None
                            self.serial_status_changed.emit(
                                False, "Conexão Arduino perdida durante operação"
                            )
                            self.error_occurred.emit(
                                "Arduino desconectado. Tentativa de reconexão em 30 s."
                            )
                            _last_reconnect_t = _now

                # ── Ciclo de item: detecta trânsito sem confirmação ───────────
                if result.roi_label != "NONE":
                    _semDeteccaoDesde = 0.0
                    if not _cicloAtivo:
                        _cicloAtivo    = True
                        _cicloConfirmado = False
                        _cicloInicioT = _now
                elif _cicloAtivo:
                    # Label voltou a NONE com ciclo aberto — iniciar/manter debounce
                    if _semDeteccaoDesde == 0.0:
                        _semDeteccaoDesde = _now
                    elif _now - _semDeteccaoDesde >= _DEBOUNCE_ROI_VAZIA_S:
                        # Debounce expirou: encerrar ciclo
                        cycle_duration = _semDeteccaoDesde - _cicloInicioT
                        if not _cicloConfirmado and cycle_duration >= _DURACAO_MIN_CICLO_S:
                            # Item passou pelo ROI sem nenhuma confirmação
                            self.rotulo_confirmado.emit("NAO_IDENTIFICADO", 0.0)
                            self.entrada_log.emit(
                                "[ROI] Não identificado — passagem livre"
                            )

                            try:
                                from db.repositories import detection_repo
                                detection_repo.insert(
                                    label="NAO_IDENTIFICADO",
                                    confidence=0.0,
                                    session_id=self._session.session_id,
                                    user_id=self._session.user_id,
                                )
                            except Exception:
                                pass

                            if serial_handler:
                                try:
                                    serial_handler.send("PASSAGEM_LIVRE")
                                    self.entrada_log.emit(
                                        "[Serial] Enviado: PASSAGEM_LIVRE"
                                    )
                                except Exception as _exc:
                                    self.entrada_log.emit(
                                        f"[Serial] Erro ao enviar 'PASSAGEM_LIVRE': {_exc}"
                                    )
                                    serial_handler.stop()
                                    serial_handler = None
                                    self.serial_status_changed.emit(
                                        False,
                                        "Conexão Arduino perdida durante operação",
                                    )
                                    self.error_occurred.emit(
                                        "Arduino desconectado. Tentativa de reconexão em 30 s."
                                    )
                                    _last_reconnect_t = _now

                        _cicloAtivo    = False
                        _cicloConfirmado = False
                        _semDeteccaoDesde      = 0.0

                # ── Tentativa de reconexão periódica ─────────────────────────
                if self._has_serial and serial_handler is None:
                    _now = time.monotonic()
                    if _now - _last_reconnect_t >= _RECONNECT_INTERVAL:
                        _last_reconnect_t = _now
                        self.entrada_log.emit("[Serial] Tentando reconexão...")
                        try:
                            from core.hardware.serial_handler import GerenciadorSerial
                            _found = GerenciadorSerial.scan_and_connect()
                        except Exception:
                            _found = None
                        if _found:
                            serial_handler = _found
                            # Reenvia atrasos após reconexão — antes do monitor.
                            from core.operation.configuracao_esteira import (
                                enviar_atrasos_arduino as _enviar_atrasos_r,
                            )
                            _ok_r, _det_r = _enviar_atrasos_r(serial_handler)
                            self.entrada_log.emit(
                                f"[Config] Atrasos (reconexão) "
                                f"{'aplicados' if _ok_r else 'padrão (firmware)'}"
                                f": {_det_r}"
                            )
                            serial_handler.start_monitor(
                                lambda line: self.entrada_log.emit(
                                    f"[Arduino] {line}"
                                )
                            )
                            self.serial_status_changed.emit(
                                True,
                                f"Arduino reconectado em {serial_handler.port}",
                            )
                            self.entrada_log.emit(
                                f"[Serial] Reconectado em {serial_handler.port}"
                            )
                        else:
                            self.entrada_log.emit(
                                "[Serial] Reconexão falhou — nenhum Arduino encontrado"
                            )

        finally:
            # ── Liberação segura de recursos ──────────────────────────────────
            # Cada passo em try/except próprio: qualquer exceção individual
            # não impede os passos seguintes nem bloqueia worker_stopped.emit().
            try:
                cap.release()
            except Exception:
                pass

            try:
                if serial_handler:
                    serial_handler.stop()
            except Exception:
                pass

            # ── Emitir métricas ao encerrar (maintenance) ─────────────────────
            try:
                if _is_maintenance and _infer_times:
                    elapsed = max(time.monotonic() - _session_t0, 0.001)
                    fps_avg = round(_frame_count / elapsed, 2)

                    def _safe(fn, lst):
                        return round(fn(lst), 4) if lst else None

                    metrics = {
                        "infer_count":     len(_infer_times),
                        "infer_time_min":  _safe(min, _infer_times),
                        "infer_time_avg":  _safe(lambda l: sum(l)/len(l), _infer_times),
                        "infer_time_max":  _safe(max, _infer_times),
                        "conf_min":        _safe(min, _conf_all),
                        "conf_avg":        _safe(lambda l: sum(l)/len(l), _conf_all),
                        "conf_max":        _safe(max, _conf_all),
                        "fps_avg":         fps_avg,
                        "serial_port":     (
                            serial_handler.port if serial_handler else None
                        ),
                        "model_path":      str(
                            __import__("core.detection.model_registry",
                                       fromlist=["get_active_path"]).get_active_path()
                            or ""
                        ),
                        "extra": {
                            "conf_thres": conf_thres,
                            "iou_thres":  iou_thres,
                            "img_size":   img_size,
                            "roi":        list(roi),
                            "timer_secs": timer_secs,
                        },
                    }
                    self.metrics_ready.emit(metrics)
            except Exception:
                pass

            # ── Sinal de término — SEMPRE emitido ────────────────────────────
            self.worker_stopped.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Tela de operação
# ─────────────────────────────────────────────────────────────────────────────

class OperationScreen(QWidget):
    """
    Tela principal de operação para o perfil operador.

    Parâmetros:
      session       — sessão autenticada
      preop_report  — resultado do diagnóstico pré-op (serial_check.ok)
      on_back       — callback para voltar ao login (após encerramento seguro)
    """

    def __init__(
        self,
        session: Session,
        preop_report: PreOpReport,
        on_back=None,
        back_label: str = "← Encerrar",
        parent=None,
    ):
        super().__init__(parent)
        self._session       = session
        self._preop_report  = preop_report
        self._on_back       = on_back
        self._back_label    = back_label
        self._worker: _TrabalhadorInferencia | None = None
        self._is_running    = False
        self._pending_back  = False
        # Timer de segurança: se worker_stopped não chegar em _STOP_TIMEOUT_MS,
        # a UI é recuperada e o thread é terminado forçadamente.
        self._stop_timer: QTimer | None = None
        self._confirmed_counts: dict[str, int] = {}
        # Sessão operacional formal (op_sessions)
        self._op_session_id:  int | None      = None
        self._op_started_at:  datetime | None = None

        self._build_ui()
        self._populate_status()
        self._audit("OPERATION_ENTER",
                    f"Operador {session.login} entrou na tela de operação")

    # ─────────────────────────────────────────────────────────── UI ──

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        # Cabeçalho
        header = QHBoxLayout()
        title = QLabel("<h2>Triagem em Operação</h2>")
        self._lbl_user = QLabel()
        self._lbl_user.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self._lbl_user)
        root.addLayout(header)

        # Linha de status do hardware
        status_row = QHBoxLayout()
        self._lbl_model_status  = QLabel("Modelo: —")
        self._lbl_serial_status = QLabel("Serial: —")
        status_row.addWidget(self._lbl_model_status)
        status_row.addStretch()
        status_row.addWidget(self._lbl_serial_status)
        root.addLayout(status_row)

        # Corpo: câmera à esquerda, painel de detecção à direita
        body = QHBoxLayout()
        body.setSpacing(16)

        # Câmera
        cam_group  = QGroupBox("Feed de Câmera")
        cam_layout = QVBoxLayout(cam_group)
        self._lbl_camera = QLabel("Pressione  ▶ Iniciar Triagem  para ativar a câmera")
        self._lbl_camera.setAlignment(Qt.AlignCenter)
        self._lbl_camera.setMinimumSize(640, 480)
        self._lbl_camera.setStyleSheet("background: #111; color: #666; font-size: 13px;")
        cam_layout.addWidget(self._lbl_camera)
        body.addWidget(cam_group, stretch=3)

        # Painel de detecção
        det_group  = QGroupBox("Detecção")
        det_layout = QVBoxLayout(det_group)
        det_layout.setSpacing(14)

        det_layout.addWidget(QLabel("<b>Detectando agora:</b>"))

        self._lbl_current = QLabel("AGUARDANDO")
        self._lbl_current.setAlignment(Qt.AlignCenter)
        f_big = QFont(); f_big.setPointSize(20); f_big.setBold(True)
        self._lbl_current.setFont(f_big)
        self._lbl_current.setStyleSheet(
            "border: 2px solid #888; border-radius: 4px; padding: 8px;"
        )
        det_layout.addWidget(self._lbl_current)

        self._lbl_conf = QLabel("")
        self._lbl_conf.setAlignment(Qt.AlignCenter)
        self._lbl_conf.setStyleSheet("font-size: 12px;")
        det_layout.addWidget(self._lbl_conf)

        det_layout.addWidget(QLabel("<b>Último confirmado:</b>"))

        self._lbl_confirmed = QLabel("—")
        self._lbl_confirmed.setAlignment(Qt.AlignCenter)
        f_huge = QFont(); f_huge.setPointSize(28); f_huge.setBold(True)
        self._lbl_confirmed.setFont(f_huge)
        self._lbl_confirmed.setStyleSheet("color: #90caf9; padding: 12px;")  # azul claro (tema escuro)
        det_layout.addWidget(self._lbl_confirmed)

        det_layout.addWidget(QLabel("<b>Contagem da sessão:</b>"))
        self._lbl_counts = QLabel("—")
        self._lbl_counts.setStyleSheet("font-size: 12px;")
        self._lbl_counts.setWordWrap(True)
        det_layout.addWidget(self._lbl_counts)

        det_layout.addStretch()

        self._lbl_op_status = QLabel("⚪  Parado")
        self._lbl_op_status.setAlignment(Qt.AlignCenter)
        self._lbl_op_status.setStyleSheet("font-size: 13px;")
        det_layout.addWidget(self._lbl_op_status)

        body.addWidget(det_group, stretch=1)
        root.addLayout(body, stretch=1)

        # ── Painel de log (visível apenas para admin) ─────────────────────────
        # Condição: session.is_admin() — login=="admin", NÃO apenas role=="maintenance".
        # Outros usuários maintenance não são admin e não veem este painel.
        self._log_panel: QGroupBox | None = None
        self._log_view:  QPlainTextEdit | None = None
        if self._session.is_admin():
            log_group  = QGroupBox("Log de Operação (Admin)")
            log_layout = QVBoxLayout(log_group)
            log_layout.setContentsMargins(6, 6, 6, 6)
            self._log_view = QPlainTextEdit()
            self._log_view.setReadOnly(True)
            self._log_view.setMaximumBlockCount(500)
            self._log_view.setFixedHeight(120)
            self._log_view.setStyleSheet(
                "font-family: monospace; font-size: 11px;"
                " background: #1e1e1e; color: #d4d4d4;"
            )
            log_layout.addWidget(self._log_view)
            self._log_panel = log_group
            root.addWidget(log_group)

        # Mensagem de erro
        self._lbl_error = QLabel("")
        self._lbl_error.setStyleSheet("color: #ef9a9a; font-size: 12px;")  # vermelho claro (tema escuro)
        self._lbl_error.setAlignment(Qt.AlignCenter)
        root.addWidget(self._lbl_error)

        # ── Linha de exportacao (visivel apos sessao encerrada) ───────────────
        export_row = QHBoxLayout()
        self._lbl_export_status = QLabel("")
        self._lbl_export_status.setStyleSheet("font-size: 11px;")
        self._btn_export_pdf = QPushButton("Exportar PDF")
        self._btn_export_csv = QPushButton("Exportar CSV")
        _exp_style = (
            "QPushButton { background:#1565c0; color:white; font-size:11px;"
            " padding:4px 14px; border-radius:3px; }"
            "QPushButton:disabled { background:#bbb; color:#eee; }"
        )
        self._btn_export_pdf.setStyleSheet(_exp_style)
        self._btn_export_csv.setStyleSheet(_exp_style)
        self._btn_export_pdf.setEnabled(False)
        self._btn_export_csv.setEnabled(False)
        self._btn_export_pdf.setToolTip(
            "Exportar relatório da última sessão em PDF"
        )
        self._btn_export_csv.setToolTip(
            "Exportar relatório da última sessão em CSV"
        )
        self._btn_export_pdf.clicked.connect(self._do_export_pdf)
        self._btn_export_csv.clicked.connect(self._do_export_csv)
        export_row.addWidget(self._lbl_export_status)
        export_row.addStretch()
        export_row.addWidget(self._btn_export_pdf)
        export_row.addWidget(self._btn_export_csv)
        root.addLayout(export_row)

        # ── Botoes principais ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self._btn_back = QPushButton(self._back_label)
        self._btn_back.clicked.connect(self._do_back)

        self._btn_start = QPushButton(">  Iniciar Triagem")
        self._btn_start.setStyleSheet(
            "QPushButton { background:#2e7d32; color:white; font-weight:bold;"
            " padding:6px 20px; border-radius:4px; }"
            "QPushButton:disabled { background:#aaa; }"
        )
        self._btn_start.clicked.connect(self._do_start)

        self._btn_stop = QPushButton("Parar")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(
            "QPushButton { background:#b71c1c; color:white; font-weight:bold;"
            " padding:6px 20px; border-radius:4px; }"
            "QPushButton:disabled { background:#aaa; }"
        )
        self._btn_stop.clicked.connect(self._do_stop)

        btn_row.addWidget(self._btn_back)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        root.addLayout(btn_row)

        # ID da ultima sessao encerrada (para exportacao)
        self._last_closed_op_session_id: int | None = None

    # ─────────────────────────────────────────────────── Inicialização ──

    def _populate_status(self) -> None:
        from core.detection.model_loader import get_active_info
        info = get_active_info()
        if info:
            name = info.get("name", "—")
            nc   = info.get("nc") or len(info.get("class_names", []))
            self._lbl_model_status.setText(f"Modelo: <b>{name}</b> ({nc} classes)")
        else:
            self._lbl_model_status.setText(
                "Modelo: <span style='color:#c62828'><b>NENHUM ATIVO</b></span>"
            )

        has_serial = (
            self._preop_report.serial_check is not None
            and self._preop_report.serial_check.ok
        )
        if has_serial:
            self._lbl_serial_status.setText(
                "Serial: <b style='color:#2e7d32'>Conectado</b>"
            )
        else:
            self._lbl_serial_status.setText(
                "Serial: <b style='color:#888'>Offline (operação sem Arduino)</b>"
            )

        self._lbl_user.setText(
            f"Operador: <b>{self._session.login}</b>"
            f" &nbsp;|&nbsp; {self._session.display_role}"
        )

    # ────────────────────────────────────────────────────── Ações ──

    def _do_start(self) -> None:
        if self._is_running:
            return
        self._lbl_error.setText("")
        self._confirmed_counts.clear()
        self._lbl_counts.setText("—")
        self._lbl_confirmed.setText("—")
        self._is_running = True
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._lbl_op_status.setText("Triagem ativa")
        self._lbl_op_status.setStyleSheet(
            "font-size:13px; color:#a5d6a7; font-weight:bold;"  # verde claro (tema escuro)
        )

        # ── Abrir sessão operacional formal ──────────────────────────────────
        try:
            from core.operation import session_manager
            self._op_session_id, self._op_started_at = session_manager.open_session(
                self._session
            )
        except Exception:
            self._op_session_id  = None
            self._op_started_at  = None

        has_serial = (
            self._preop_report.serial_check is not None
            and self._preop_report.serial_check.ok
        )
        self._worker = _TrabalhadorInferencia(
            session=self._session,
            has_serial=has_serial,
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.detection_updated.connect(self._on_detection)
        self._worker.rotulo_confirmado.connect(self._on_confirmed)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.worker_stopped.connect(self._on_worker_stopped)
        self._worker.metrics_ready.connect(self._on_metrics)
        self._worker.serial_status_changed.connect(self._on_serial_status)
        self._worker.entrada_log.connect(self._append_log)
        self._worker.start()

        self._audit("OPERATION_START", f"Triagem iniciada — {self._session.login}")

    # Tempo máximo (ms) que a UI espera worker_stopped antes do fallback forçado
    _STOP_TIMEOUT_MS = 5_000

    def _do_stop(self) -> None:
        if self._worker and self._is_running:
            self._btn_stop.setEnabled(False)
            self._lbl_op_status.setText("🟡  Encerrando...")
            self._lbl_op_status.setStyleSheet("font-size:13px; color:#f57c00;")
            self._worker.request_stop()
            # Arma timer de segurança: se worker_stopped não chegar a tempo,
            # _on_stop_timeout força recuperação da UI e termina o thread.
            if self._stop_timer is None:
                self._stop_timer = QTimer(self)
                self._stop_timer.setSingleShot(True)
                self._stop_timer.timeout.connect(self._on_stop_timeout)
            self._stop_timer.start(self._STOP_TIMEOUT_MS)

    def _do_back(self) -> None:
        if self._is_running:
            self._pending_back = True
            self._do_stop()
        else:
            self._audit_stop()
            if self._on_back:
                self._on_back()

    # ────────────────────────────────── Fallback de encerramento seguro ──

    def _on_stop_timeout(self) -> None:
        """
        Fallback de segurança: worker_stopped não chegou em _STOP_TIMEOUT_MS.
        Causas típicas: cap.release() bloqueando no driver, exceção no finally
        do worker, ou thread presa em I/O de câmera/serial.
        Ação: força QThread.terminate(), recupera a UI e honra navegação pendente.
        """
        if not self._is_running:
            return  # worker parou normalmente antes do timer disparar — ignorar

        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2_000)   # aguarda até 2 s para o SO liberar o thread

        self._is_running = False
        self._lbl_error.setText(
            "⚠  Encerramento forçado — câmera ou porta serial sem resposta."
        )
        self._recover_stopped_ui()
        self._audit_stop()  # encerra sessão mesmo em timeout forçado

        if self._pending_back:
            self._pending_back = False
            if self._on_back:
                self._on_back()

    def _recover_stopped_ui(self) -> None:
        """Restaura widgets para o estado ⚪ Parado (normal e por timeout)."""
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._lbl_op_status.setText("⚪  Parado")
        self._lbl_op_status.setStyleSheet("font-size:13px;")
        self._lbl_current.setText("AGUARDANDO")
        self._lbl_current.setStyleSheet(
            "border:2px solid #888; border-radius:4px; padding:8px;"
        )
        self._lbl_conf.setText("")
        self._lbl_camera.setText(
            "Pressione  ▶ Iniciar Triagem  para ativar a câmera"
        )

    # ────────────────────────────────────────────────────── Slots ──

    def _on_frame(self, img: QImage) -> None:
        pix    = QPixmap.fromImage(img)
        scaled = pix.scaled(
            self._lbl_camera.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._lbl_camera.setPixmap(scaled)

    def _on_detection(self, label: str, conf: float) -> None:
        if label == "NONE":
            self._lbl_current.setText("AGUARDANDO")
            self._lbl_current.setStyleSheet(
                "border:2px solid #888; border-radius:4px; padding:8px;"
            )
            self._lbl_conf.setText("")
        else:
            self._lbl_current.setText(label)
            self._lbl_current.setStyleSheet(
                # Azul claro para legibilidade no tema escuro
                "color:#90caf9; border:2px solid #90caf9; border-radius:4px; padding:8px;"
            )
            self._lbl_conf.setText(f"Confiança: {conf:.1%}")

    def _on_confirmed(self, label: str, _conf: float) -> None:
        if label == "NAO_IDENTIFICADO":
            self._lbl_confirmed.setText("NÃO IDENTIFICADO")
            self._lbl_confirmed.setStyleSheet("color: #ffb74d; padding: 12px;")
        else:
            self._lbl_confirmed.setText(label)
            self._lbl_confirmed.setStyleSheet("color: #90caf9; padding: 12px;")
        self._confirmed_counts[label] = self._confirmed_counts.get(label, 0) + 1
        counts_text = "   ".join(
            f"<b>{lbl.replace('NAO_IDENTIFICADO', 'NÃO IDENT.')}</b>: {n}"
            for lbl, n in sorted(self._confirmed_counts.items())
        )
        self._lbl_counts.setText(counts_text)

    def _on_error(self, msg: str) -> None:
        self._lbl_error.setText(f"! {msg}")
        self._audit("OPERATION_ERROR", msg)
        # Registrar erro na sessão operacional formal
        if self._op_session_id is not None:
            try:
                from core.operation import session_manager
                session_manager.record_event(self._op_session_id, "ERROR", msg)
            except Exception:
                pass

    def _on_serial_status(self, connected: bool, detail: str) -> None:
        if connected:
            self._lbl_serial_status.setText(
                f"Serial: <b style='color:#2e7d32'>{detail}</b>"
            )
        else:
            self._lbl_serial_status.setText(
                f"Serial: <b style='color:#c62828'>{detail}</b>"
            )

    def _append_log(self, entry: str) -> None:
        if self._log_view is None:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_view.appendPlainText(f"{ts}  {entry}")
        sb = self._log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_metrics(self, metrics: dict) -> None:
        """Recebe métricas pré-computadas do worker (apenas maintenance)."""
        if self._op_session_id is None:
            return
        try:
            from core.operation import session_manager
            session_manager.record_metrics(self._op_session_id, metrics)
        except Exception:
            pass

    def _on_worker_stopped(self) -> None:
        # Cancelar timer de segurança — encerramento ocorreu dentro do prazo.
        if self._stop_timer is not None:
            self._stop_timer.stop()

        self._is_running = False
        self._recover_stopped_ui()
        self._audit_stop()  # encerra sessão em TODOS os caminhos de parada

        if self._pending_back:
            self._pending_back = False
            if self._on_back:
                self._on_back()

    # ────────────────────────────────────────────────────── Auditoria ──

    def _audit(self, event: str, description: str) -> None:
        try:
            from db.repositories import audit_repo
            audit_repo.record(event, description=description,
                              user_id=self._session.user_id)
        except Exception:
            pass

    def _audit_stop(self) -> None:
        nao_id   = self._confirmed_counts.get("NAO_IDENTIFICADO", 0)
        total_id = sum(v for k, v in self._confirmed_counts.items()
                       if k != "NAO_IDENTIFICADO")
        total    = total_id + nao_id
        nao_id_str = f", {nao_id} não identificado(s)" if nao_id else ""
        self._audit(
            "OPERATION_STOP",
            f"Operacao encerrada — {total_id} identificado(s){nao_id_str}"
            f" — {total} total — {self._session.login}",
        )
        # Fechar sessao operacional formal
        if self._op_session_id is not None and self._op_started_at is not None:
            closed_id = self._op_session_id
            try:
                from core.operation import session_manager
                session_manager.close_session(
                    op_session_id = self._op_session_id,
                    started_at    = self._op_started_at,
                    counts        = dict(self._confirmed_counts),
                    auth_session  = self._session,
                )
                # Habilitar exportacao com a sessao recentemente encerrada
                self._last_closed_op_session_id = closed_id
                self._btn_export_pdf.setEnabled(True)
                self._btn_export_csv.setEnabled(True)
                self._lbl_export_status.setText(
                    f"Sessão #{closed_id} encerrada — {total} item(s). "
                    f"Pronto para exportar."
                )
            except Exception:
                pass
            finally:
                self._op_session_id = None
                self._op_started_at = None

    # ────────────────────────────────────────────────────── Exportacao ──

    def _do_export_pdf(self) -> None:
        """Exporta relatorio da ultima sessao encerrada em PDF."""
        self._run_export(fmt="pdf")

    def _do_export_csv(self) -> None:
        """Exporta relatorio da ultima sessao encerrada em CSV."""
        self._run_export(fmt="csv")

    def _run_export(self, fmt: str) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from core.utils.paths import project_root

        op_id = self._last_closed_op_session_id
        if op_id is None:
            QMessageBox.warning(self, "Exportação", "Nenhuma sessão encerrada disponível.")
            return

        # Pasta padrao: project_root/reports/
        default_dir = str(project_root() / "reports")
        import os; os.makedirs(default_dir, exist_ok=True)

        suffix     = ".pdf" if fmt == "pdf" else ".csv"
        profile    = self._session.role
        tipo       = "tecnico" if profile == "maintenance" else "operacional"
        default_fn = f"relatorio_{tipo}_sessao{op_id}{suffix}"

        if fmt == "pdf":
            filters = "PDF (*.pdf)"
        else:
            filters = "CSV (*.csv)"

        dest, _ = QFileDialog.getSaveFileName(
            self, f"Exportar Relatório {fmt.upper()}",
            str(Path(default_dir) / default_fn),
            filters,
        )
        if not dest:
            return  # usuario cancelou

        dest_path = Path(dest)

        try:
            from core.operation import report_service

            report_data = report_service.build_report(op_id, profile)

            if fmt == "pdf":
                out = report_service.export_pdf(report_data, dest_path)
            else:
                out = report_service.export_csv(report_data, dest_path)

            self._audit(
                f"REPORT_EXPORT_{fmt.upper()}",
                f"Relatorio {fmt.upper()} exportado: {out.name} — sessao #{op_id}",
            )
            self._lbl_export_status.setText(
                f"Exportado: {out.name}"
            )
            QMessageBox.information(
                self,
                "Exportação concluída",
                f"Relatório {fmt.upper()} salvo em:\n{out}",
            )
        except Exception as exc:
            self._audit("REPORT_EXPORT_ERROR", str(exc))
            QMessageBox.critical(
                self,
                "Erro na exportação",
                f"Não foi possível exportar o relatório.\n\n{exc}",
            )

    # ─────────────────────────────────────────────── Fechamento seguro ──

    def closeEvent(self, event) -> None:
        if self._is_running:
            self._do_stop()
            if self._worker:
                self._worker.wait(4000)
        self._audit_stop()
        super().closeEvent(event)
