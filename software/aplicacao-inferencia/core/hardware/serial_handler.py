"""
Handler serial para comunicação com o Arduino Master.

Modos de uso:
  1. Diagnóstico (pre_op_check / hardware_probe):
       h = GerenciadorSerial.from_config()
       ok = h.connect()
       h.send("Status")
       line = h.read_line(timeout=2.0)
       h.stop()

  2. Operação contínua (operation_screen):
       h = GerenciadorSerial.from_config()
       h.connect()
       h.start_monitor(callback=my_fn)   # thread daemon
       h.send("vidro")
       ...
       h.stop()

Protocolo Arduino (master.ino):
  Python → "LABEL\\n"               ex.: "vidro\\n", "metal\\n"  (classes lowercase)
  Python → "Status\\n"              solicita status dos atuadores
  Python → "CONFIG_ATRASOS:...\\n"  configura atrasos da esteira em runtime
  Arduino → "Status: A2:2,A3:2,A4:2,A5:2\\n"
  Arduino → "CONF_OK\\n" / "CONF_ERRO:<motivo>\\n"
  Arduino → "[Esteira] ...\\n" / "[Cmd] ...\\n"   logs de debug
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import serial
import serial.tools.list_ports


class SerialError(Exception):
    pass


class GerenciadorSerial:
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self._conn: serial.Serial | None = None
        self._lock = threading.Lock()

        self._monitor_thread: threading.Thread | None = None
        self._monitor_running = False
        self._monitor_callback: Callable[[str], None] | None = None

        # Guardado após connect() para diagnóstico
        self.connection_detail: str = ""
        # Última resposta recebida de handshake() — None se ainda não chamado
        self.last_handshake_response: str | None = None

    # ------------------------------------------------------------------
    # Construção a partir de configuração (settings_manager)
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls) -> "GerenciadorSerial":
        from core.settings import settings_manager
        port = settings_manager.get("arduino.port")
        baudrate = int(settings_manager.get("arduino.baudrate"))
        return cls(port, baudrate)

    # ------------------------------------------------------------------
    # Conexão
    # ------------------------------------------------------------------

    # Timeout máximo para serial.Serial() open() — portas Bluetooth no Windows
    # bloqueiam indefinidamente quando não há dispositivo BT pareado/conectado.
    _CONNECT_TIMEOUT = 3.0

    def connect(self) -> bool:
        """
        Tenta abrir a porta serial com timeout de thread.
        Retorna True se bem-sucedido.

        Motivação: serial.Serial.open() bloqueia indefinidamente no Windows
        para portas Bluetooth (COM virtuais via BTHENUM) sem dispositivo
        conectado. O wrapper de thread garante que connect() sempre retorna
        dentro de _CONNECT_TIMEOUT segundos.
        """
        _result: list = [None]  # [serial.Serial | Exception]

        def _do_open():
            try:
                _result[0] = serial.Serial(
                    self.port, self.baudrate, timeout=self.timeout
                )
            except Exception as exc:
                _result[0] = exc

        t = threading.Thread(target=_do_open, daemon=True)
        t.start()
        t.join(timeout=self._CONNECT_TIMEOUT)

        if t.is_alive():
            self.connection_detail = (
                f"{self.port}: timeout ({self._CONNECT_TIMEOUT:.0f}s) — "
                "porta bloqueada (Bluetooth sem dispositivo ou porta virtual)"
            )
            return False

        if isinstance(_result[0], Exception):
            self.connection_detail = f"{self.port}: {_result[0]}"
            return False

        conn = _result[0]
        if conn is None:
            self.connection_detail = f"{self.port}: falha desconhecida ao abrir"
            return False

        # Aguarda o bootloader do Arduino Uno (optiboot) finalizar.
        # O bootloader ocupa ~1,8 s após a abertura da porta. Com 0,5 s o
        # PING_RECYCLEAI chegava durante o bootloader e era descartado;
        # 2,2 s garante margem segura sem ser excessivo.
        time.sleep(2.2)
        self._flush_input()   # descarta bytes residuais do bootloader
        with self._lock:
            self._conn = conn
        self.connection_detail = f"Porta {self.port} aberta ({self.baudrate} baud)"
        return True

    def _flush_input(self):
        """
        Descarta todos os bytes pendentes no buffer de recepção.

        Usado antes de cada PING_RECYCLEAI para evitar que mensagens de boot
        ou lixo residual do bootloader do Arduino sejam confundidos com PONG.
        """
        try:
            with self._lock:
                if self._conn and self._conn.is_open:
                    self._conn.reset_input_buffer()
        except Exception:
            pass

    def is_connected(self) -> bool:
        with self._lock:
            return self._conn is not None and self._conn.is_open

    # ------------------------------------------------------------------
    # Envio
    # ------------------------------------------------------------------

    def send(self, command: str, retry: int = 2) -> bool:
        """
        Envia 'command\\n' via serial.
        Tenta até `retry` vezes antes de levantar SerialError.
        """
        payload = (command.strip() + "\n").encode()
        last_exc = None
        for attempt in range(retry):
            try:
                with self._lock:
                    if self._conn is None or not self._conn.is_open:
                        raise SerialError("Serial não conectada")
                    self._conn.write(payload)
                    self._conn.flush()
                return True
            except serial.SerialException as exc:
                last_exc = exc
                if attempt < retry - 1:
                    time.sleep(0.1)
        raise SerialError(f"Falha ao enviar '{command}' após {retry} tentativas: {last_exc}")

    # ------------------------------------------------------------------
    # Leitura
    # ------------------------------------------------------------------

    def read_line(self, timeout: float = 1.0) -> str | None:
        """
        Lê uma linha da serial com timeout.
        Retorna a string decodificada (sem '\\n') ou None se timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with self._lock:
                    if self._conn is None or not self._conn.is_open:
                        return None
                    waiting = self._conn.in_waiting
                if waiting:
                    with self._lock:
                        raw = self._conn.readline()
                    return raw.decode(errors="ignore").strip()
                time.sleep(0.02)
            except serial.SerialException:
                return None
        return None

    def read_until(
        self, marker: str, timeout: float = 5.0
    ) -> str | None:
        """Lê linhas até encontrar uma que contenha `marker` ou timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.read_line(timeout=min(0.5, deadline - time.monotonic()))
            if line is not None and marker in line:
                return line
        return None

    # ------------------------------------------------------------------
    # Monitor em background (modo operação)
    # ------------------------------------------------------------------

    def start_monitor(self, callback: Callable[[str], None]):
        """
        Inicia thread daemon que repassa cada linha recebida do Arduino
        para `callback(line: str)`.
        Usar no modo de operação contínua.
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._monitor_running = True
        self._monitor_callback = callback
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="serial-monitor"
        )
        self._monitor_thread.start()

    def _monitor_loop(self):
        while self._monitor_running:
            line = self.read_line(timeout=0.3)
            if line and self._monitor_callback:
                try:
                    self._monitor_callback(line)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Encerramento
    # ------------------------------------------------------------------

    def stop(self):
        self._monitor_running = False
        with self._lock:
            if self._conn and self._conn.is_open:
                try:
                    self._conn.close()
                except Exception:
                    pass
            self._conn = None

    # ------------------------------------------------------------------
    # Handshake e auto-detecção
    # ------------------------------------------------------------------

    def handshake(self, timeout: float = 2.0, attempts: int = 3) -> str | None:
        """
        Envia PING_RECYCLEAI e aguarda PONG_RECYCLEAI do Arduino.

        Retorna a linha de resposta completa (ex.: "PONG_RECYCLEAI:OK" ou
        "PONG_RECYCLEAI:SLAVE_ERROR"), ou None se não houve resposta.

        O chamador é responsável por interpretar o resultado:
          · qualquer PONG_RECYCLEAI → dispositivo RecycleAI identificado
          · "PONG_RECYCLEAI:OK"     → master + slave ambos saudáveis
          · "PONG_RECYCLEAI:SLAVE_ERROR" → master OK, slave com falha
          · None                   → sem resposta (firmware diferente ou
                                     dispositivo serial incorreto)

        Parâmetros:
          timeout  — tempo máximo (s) de espera por PONG em cada tentativa.
          attempts — número de tentativas (padrão 3). Entre cada tentativa
                     o buffer de entrada é limpo para evitar resposta de
                     ciclo anterior. Use attempts=1 para comportamento legado.
        """
        for attempt in range(attempts):
            # Limpa lixo/respostas anteriores antes de cada PING
            self._flush_input()
            try:
                self.send("PING_RECYCLEAI")
            except SerialError:
                self.last_handshake_response = None
                return None
            response = self.read_until("PONG_RECYCLEAI", timeout=timeout)
            self.last_handshake_response = response
            if response is not None:
                return response
            # Sem PONG — aguarda um tick antes de tentar novamente
            if attempt < attempts - 1:
                time.sleep(0.1)
        return None

    @classmethod
    def scan_and_connect(
        cls,
        handshake_timeout: float = 2.0,
        attempts: int = 2,
    ) -> "GerenciadorSerial | None":
        """
        Varre todas as portas COM disponíveis procurando o Arduino RecycleAI.

        Critério de identificação: qualquer resposta PONG_RECYCLEAI (OK ou
        SLAVE_ERROR). Isso localiza o dispositivo correto sem depender do
        estado do slave I2C. A validação completa de saúde (slave OK)
        é responsabilidade da camada acima (pre_op_check._check_serial).

        Parâmetros:
          handshake_timeout — tempo máximo (s) de espera por PONG por tentativa.
          attempts          — tentativas de handshake por porta (padrão 2).

        Retorna o primeiro GerenciadorSerial cujo handshake identificou o
        dispositivo, ou None se nenhuma porta respondeu.
        """
        from core.settings import settings_manager
        baudrate = int(settings_manager.get("arduino.baudrate"))
        for info in cls.list_ports():
            # Portas Bluetooth (BTHENUM) bloqueiam serial.Serial.open() no Windows
            # quando não há dispositivo pareado/ativo — pular preventivamente.
            if "BTHENUM" in info.get("hwid", "").upper():
                continue
            handler = cls(info["device"], baudrate)
            if not handler.connect():
                continue
            response = handler.handshake(timeout=handshake_timeout, attempts=attempts)
            if response is not None:      # dispositivo RecycleAI identificado
                return handler
            handler.stop()
        return None

    # ------------------------------------------------------------------
    # Utilitários estáticos
    # ------------------------------------------------------------------

    @staticmethod
    def list_ports() -> list[dict]:
        """
        Lista todas as portas seriais disponíveis no sistema.
        Retorna lista de dicts: {device, description, hwid}
        """
        return [
            {
                "device": p.device,
                "description": p.description,
                "hwid": p.hwid,
            }
            for p in serial.tools.list_ports.comports()
        ]
