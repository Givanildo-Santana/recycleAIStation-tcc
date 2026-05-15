"""
Protocolo de diagnóstico manual de hardware — constantes e parsers.

Todos os comandos DIAG_* são exclusivos para diagnóstico e trafegam
de forma independente das operações de triagem normais.

Protocolo serial (Python → Arduino master.ino):
  DIAG_ESTEIRA_ON    → ligarEsteira() sem atuador e sem delay
  DIAG_ESTEIRA_OFF   → desligarEsteira()
  DIAG_A2A / A2R / A2P → enviarComandoI2C direto (sem esteira, sem delay)
  DIAG_A3A / A3R / A3P → idem para A3
  DIAG_A4A / A4R / A4P → idem para A4
  DIAG_A5A / A5R / A5P → idem para A5
  DIAG_PARAR_TUDO    → desligar esteira + parar todos os atuadores
  DIAG_CHAVES        → ler 8 chaves fim de curso; responde DIAG:CHAVES:...

Protocolo serial (Arduino → Python):
  DIAG:CHAVES:A2C1=1,A2C2=0,...   1 = ativada (LOW), 0 = livre (HIGH)
  Status: A2:2,A3:2,...            resposta ao CMD_STATUS (operacional)
  [Diag] ...                       logs de confirmação internos
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Constantes de comando — diagnóstico
# ---------------------------------------------------------------------------

CMD_DIAG_ENTER       = "DIAG_ENTER"    # suspende verificarAtuador() no firmware
CMD_DIAG_EXIT        = "DIAG_EXIT"     # retoma operação normal + CMD_STATUS de sync
CMD_DIAG_ESTEIRA_ON  = "DIAG_ESTEIRA_ON"
CMD_DIAG_ESTEIRA_OFF = "DIAG_ESTEIRA_OFF"
CMD_DIAG_PARAR_TUDO  = "DIAG_PARAR_TUDO"
CMD_DIAG_CHAVES      = "DIAG_CHAVES"

# Respostas do firmware ao DIAG_ENTER / DIAG_EXIT
RESP_MODO_DIAG_ON  = "MODO_DIAG:ON"
RESP_MODO_DIAG_OFF = "MODO_DIAG:OFF"

CMD_DIAG_A2A = "DIAG_A2A"
CMD_DIAG_A2R = "DIAG_A2R"
CMD_DIAG_A2P = "DIAG_A2P"

CMD_DIAG_A3A = "DIAG_A3A"
CMD_DIAG_A3R = "DIAG_A3R"
CMD_DIAG_A3P = "DIAG_A3P"

CMD_DIAG_A4A = "DIAG_A4A"
CMD_DIAG_A4R = "DIAG_A4R"
CMD_DIAG_A4P = "DIAG_A4P"

CMD_DIAG_A5A = "DIAG_A5A"
CMD_DIAG_A5R = "DIAG_A5R"
CMD_DIAG_A5P = "DIAG_A5P"

# Reutiliza o comando operacional existente — Status retorna atuador states
CMD_STATUS = "Status"

# ---------------------------------------------------------------------------
# Identificadores
# ---------------------------------------------------------------------------

ATUADOR_IDS = ["A2", "A3", "A4", "A5"]

# Chaves fim de curso: C1 = base (início), C2 = topo (fim de curso)
CHAVE_IDS = ["A2C1", "A2C2", "A3C1", "A3C2", "A4C1", "A4C2", "A5C1", "A5C2"]

# ---------------------------------------------------------------------------
# Mapeamento atuador → comandos DIAG
# ---------------------------------------------------------------------------

ATUADOR_CMDS: dict[str, dict[str, str]] = {
    "A2": {"avançar": CMD_DIAG_A2A, "retornar": CMD_DIAG_A2R, "parar": CMD_DIAG_A2P},
    "A3": {"avançar": CMD_DIAG_A3A, "retornar": CMD_DIAG_A3R, "parar": CMD_DIAG_A3P},
    "A4": {"avançar": CMD_DIAG_A4A, "retornar": CMD_DIAG_A4R, "parar": CMD_DIAG_A4P},
    "A5": {"avançar": CMD_DIAG_A5A, "retornar": CMD_DIAG_A5R, "parar": CMD_DIAG_A5P},
}

_ESTADO_MAP = {"0": "AVANCANDO", "1": "RETORNANDO", "2": "PARADO"}

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_chaves(line: str) -> dict[str, bool] | None:
    """
    Parseia a linha DIAG:CHAVES enviada pelo firmware.

    Formato: "DIAG:CHAVES:A2C1=1,A2C2=0,A3C1=1,A3C2=0,A4C1=1,A4C2=0,A5C1=1,A5C2=0"

    Retorna:
      dict chave → bool:  True = ativada (LOW/pressionada)  /  False = livre (HIGH)
      None se a linha não é uma resposta DIAG:CHAVES válida.
    """
    if "DIAG:CHAVES:" not in line:
        return None
    _, _, payload = line.partition("DIAG:CHAVES:")
    result: dict[str, bool] = {}
    for part in payload.split(","):
        part = part.strip()
        if "=" in part:
            key, _, val = part.partition("=")
            key = key.strip()
            if key in CHAVE_IDS:
                result[key] = val.strip() == "1"
    return result if result else None


def parse_actuator_status(line: str) -> dict[str, str] | None:
    """
    Parseia a linha "Status: A2:2,A3:0,A4:2,A5:2" enviada pelo firmware master.ino.

    Formato real (master.ino → Python):
      "Status: A2:2,A3:2,A4:2,A5:2"
      Valores: 0=AVANCANDO, 1=RETORNANDO, 2=PARADO

    Retorna:
      dict atuador → estado_str (AVANCANDO / RETORNANDO / PARADO / ?)
      None se a linha não contém status de atuadores (sem "A2:").
    """
    if "A2:" not in line:
        return None
    # Remove prefixo "Status: " para isolar o payload de atuadores.
    # Fallback para a linha inteira caso o prefixo seja omitido (defensivo).
    payload = line.partition("Status:")[2].lstrip() if "Status:" in line else line
    result: dict[str, str] = {}
    for aid in ATUADOR_IDS:
        idx = payload.find(f"{aid}:")
        if idx >= 0:
            raw = payload[idx + 3: idx + 4].strip()
            result[aid] = _ESTADO_MAP.get(raw, "?")
    return result if result else None


def hint_posicao(c1: bool, c2: bool) -> str:
    """
    Interpreta as duas chaves de um atuador e devolve string descritiva.

      c1=True, c2=False  → na base (posição de repouso)
      c1=False, c2=True  → no topo (fim de curso avançado)
      c1=False, c2=False → em trânsito
      c1=True,  c2=True  → sinal inconsistente (erro de hardware)
    """
    if c1 and not c2:
        return "na base"
    if not c1 and c2:
        return "no topo"
    if c1 and c2:
        return "⚠ inconsistente"
    return "em trânsito"
