"""
Configuração dinâmica dos atrasos da esteira transportadora.

Responsabilidade: ler os atrasos de conveyor.delay_* do banco e enviá-los
ao Arduino master via comando CONFIG_ATRASOS antes de iniciar o monitor serial.

─────────────────────────────────────────────────────────────────────────────
Contrato serial (novo — criado neste módulo):

  Python  → "CONFIG_ATRASOS:<vidro>:<papel>:<plastico>:<metal>:<livre>\\n"
  Arduino → "CONF_OK"              (valores aceitos e aplicados em memória)
  Arduino → "CONF_ERRO:<motivo>"   (valor inválido ou formato incorreto)

  Exemplo de comando enviado:
    "CONFIG_ATRASOS:4700:6260:7900:9000:10000"

  Ordem fixa dos valores (ms):
    1. delay_vidro_ms        → atrasoVidro    no firmware
    2. delay_papel_ms        → atrasoPapel    no firmware
    3. delay_plastico_ms     → atrasoPlastico no firmware
    4. delay_metal_ms        → atrasoMetal    no firmware
    5. delay_nao_identi_ms   → atrasoPassagemLivre no firmware

─────────────────────────────────────────────────────────────────────────────
Integração com operation_screen.py:

  enviar_atrasos_arduino() DEVE ser chamado ANTES de start_monitor(),
  para evitar conflito de leitura com a thread de monitoramento.

  Fluxo na tela de operação:
    1. GerenciadorSerial.connect()            ← abre porta + handshake
    2. enviar_atrasos_arduino(serial_handler) ← THIS MODULE (antes do monitor)
    3. GerenciadorSerial.start_monitor(cb)    ← inicia thread de leitura

─────────────────────────────────────────────────────────────────────────────
Nomenclatura:
  Nomes internos deste módulo seguem a regra do projeto: português para
  tudo que for autoral. Exceções: CONF_OK/CONF_ERRO são tokens do protocolo
  serial (contrato Arduino↔Python) — mantidos como definidos no firmware.
"""
from __future__ import annotations

import time

from core.settings import settings_manager

# Comando enviado ao Arduino para configurar os atrasos.
# Mantido em inglês estrutural ("CONFIG") + português semântico ("ATRASOS")
# para consistência com o padrão de comandos do protocolo (PING_RECYCLEAI,
# PASSAGEM_LIVRE, DIAG_*) já estabelecido no firmware.
COMANDO_CONFIG_ATRASOS = "CONFIG_ATRASOS"

# Chaves de configuração lidas do banco, em ordem fixa de envio.
_CHAVES_ATRASO = [
    "conveyor.delay_vidro_ms",
    "conveyor.delay_papel_ms",
    "conveyor.delay_plastico_ms",
    "conveyor.delay_metal_ms",
    "conveyor.delay_nao_identificado_ms",  # → atrasoPassagemLivre no firmware
]

# Tempo máximo (segundos) para aguardar CONF_OK do Arduino após envio.
_TEMPO_ESPERA_ACK_S = 2.0


def ler_atrasos() -> dict[str, int]:
    """
    Retorna os atrasos configurados (ms) como dicionário chave → valor.

    Lê do banco via settings_manager; fallback para padrão do _DEFAULTS
    se a chave não estiver presente (idem ao comportamento do settings_manager).
    """
    return {chave: settings_manager.get_int(chave) for chave in _CHAVES_ATRASO}


def montar_comando(atrasos: dict[str, int]) -> str:
    """
    Monta a string do comando CONFIG_ATRASOS a partir do dicionário de atrasos.

    A ordem das chaves é fixa (_CHAVES_ATRASO) — deve coincidir com a ordem
    de parse em processarConfigAtrasos() no firmware master.ino.

    Exemplo de saída: 'CONFIG_ATRASOS:4700:6260:7900:9000:10000'
    """
    valores = ":".join(str(atrasos[c]) for c in _CHAVES_ATRASO)
    return f"{COMANDO_CONFIG_ATRASOS}:{valores}"


def enviar_atrasos_arduino(gerenciador_serial) -> tuple[bool, str]:
    """
    Lê os atrasos configurados no banco e os envia ao Arduino master.

    IMPORTANTE: chamar ANTES de GerenciadorSerial.start_monitor() para evitar
    que a thread de monitoramento consuma o ACK (CONF_OK) antes desta função.

    Parâmetro:
        gerenciador_serial — instância de GerenciadorSerial ou None.

    Retorna (sucesso: bool, detalhe: str).

    Nunca levanta exceção — retorna (False, motivo) em qualquer falha.
    A operação continua com os delays padrão hardcoded no firmware (fallback seguro).
    """
    if gerenciador_serial is None:
        return False, "sem serial — atrasos padrão do firmware serão usados"

    if not gerenciador_serial.is_connected():
        return False, "serial não conectada — atrasos padrão do firmware serão usados"

    atrasos = ler_atrasos()

    # Firmware rejeita valores fora de [0, 30000] ms com CONF_ERRO.
    # Barrar aqui evita envio silencioso de configuração inválida.
    invalidos = {c: v for c, v in atrasos.items() if not (0 <= v <= 30_000)}
    if invalidos:
        desc = ", ".join(
            f"{c.split('.')[-1]}={v}" for c, v in invalidos.items()
        )
        return False, f"atraso(s) fora do intervalo [0, 30000] ms — {desc}"

    cmd = montar_comando(atrasos)

    try:
        gerenciador_serial.send(cmd)
    except Exception as exc:
        return False, f"erro ao enviar CONFIG_ATRASOS: {exc}"

    # Aguarda ACK do Arduino.
    # Linhas de debug como "[Cmd] CONFIG_ATRASOS:..." são ignoradas (continue).
    # Retorna ao primeiro CONF_OK ou CONF_ERRO recebido.
    t0 = time.monotonic()
    while time.monotonic() - t0 < _TEMPO_ESPERA_ACK_S:
        try:
            linha = gerenciador_serial.read_line(timeout=0.3)
        except Exception:
            break
        if linha is None:
            continue
        linha = linha.strip()
        if linha == "CONF_OK":
            return True, f"atrasos configurados: {atrasos}"
        if linha.startswith("CONF_ERRO"):
            return False, f"Arduino rejeitou configuração: {linha}"
        # Ignora linhas intermediárias (ex.: "[Cmd] CONFIG_ATRASOS:...") e continua

    return False, "timeout aguardando CONF_OK — atrasos padrão do firmware serão usados"
