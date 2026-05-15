#include <Wire.h>
#include <SoftPWM.h>

// ===== PINOS DE CHAVES FIM DE CURSO =====
#define A2C1 2
#define A2C2 3
#define A3C1 4
#define A3C2 5
#define A4C1 6
#define A4C2 7
#define A5C1 8
#define A5C2 9

// ===== PINOS DA ESTEIRA =====
#define RPWM 10
#define LPWM 11
#define R_EN 12
#define L_EN 13

// ===== COMANDOS I2C =====
// Strings curtas: ficam em flash como literais de #define.
// Passadas como const char* — sem copia para heap.
#define CMD_A2A    "A2A"
#define CMD_A2R    "A2R"
#define CMD_A2P    "A2P"
#define CMD_A3A    "A3A"
#define CMD_A3R    "A3R"
#define CMD_A3P    "A3P"
#define CMD_A4A    "A4A"
#define CMD_A4R    "A4R"
#define CMD_A4P    "A4P"
#define CMD_A5A    "A5A"
#define CMD_A5R    "A5R"
#define CMD_A5P    "A5P"
#define CMD_STATUS "STATUS_REQ"

// ===== ESTADOS DE ATUADOR =====
#define AVANCANDO  0
#define RETORNANDO 1
#define PARADO     2

// ===== PINOS DAS CHAVES (array de leitura no setup) =====
const uint8_t pinosChaveFim[] = {A2C1, A2C2, A3C1, A3C2, A4C1, A4C2, A5C1, A5C2};

// ===== BUFFERS ESTATICOS GLOBAIS =====
// Substituem toda alocacao dinamica da classe String.
// bufferSerial: maior comando e "CONFIG_ATRASOS:10000:10000:10000:10000:10000"
//               (45 chars) + null. Alocado com margem em 64.
// respostaI2C : slave envia "A2:0,A3:0,A4:0,A5:0" (19 chars) + null.
char bufferSerial[64];
char respostaI2C[33];

// ===== ESTADOS DOS ATUADORES =====
// uint8_t suficiente: valores sao apenas AVANCANDO(0), RETORNANDO(1), PARADO(2).
uint8_t estadoA2 = PARADO;
uint8_t estadoA3 = PARADO;
uint8_t estadoA4 = PARADO;
uint8_t estadoA5 = PARADO;

// Suspende verificarAtuador() durante diagnóstico manual.
// Sem este flag, verificarAtuador() reenvia CMD_AxR se o atuador parar
// no meio do stroke, brigando contra DIAG_AxP e tornando a parada irreal.
bool em_modo_diagnostico = false;

// ===== WATCHDOG DO MODO DIAGNÓSTICO =====
// Saida automatica se nenhum comando serial for recebido por TIMEOUT_DIAG_MS ms
// enquanto em_modo_diagnostico=true. Protege contra perda de conexao com a GUI
// (ex.: crash do processo, queda de energia no PC) sem envio de DIAG_EXIT.
// Valor 8000 ms e conservador: superior ao intervalo de polling DIAG_CHAVES (~500 ms).
#define TIMEOUT_DIAG_MS 8000UL
unsigned long ultimoComandoSerial = 0;

// ===== ATRASOS CONFIGURÁVEIS DA ESTEIRA (em ms) =====
// Valores padrão iguais aos hardcoded anteriores.
// Sobrescritos em runtime pelo comando serial CONFIG_ATRASOS (sem reflash).
// unsigned int: suficiente para 0–30000 ms (< 65535).
unsigned int atrasoVidro         = 4700;
unsigned int atrasoPapel         = 6260;
unsigned int atrasoPlastico      = 7900;
unsigned int atrasoMetal         = 9000;
unsigned int atrasoPassagemLivre = 10000;

uint8_t pwmInicial    = 150;
uint8_t pwmFinal      = 150;
uint8_t incrementoPWM = 5;
uint8_t intervaloRampa = 30;

uint8_t ultimoPWM_Reverso = 0;
uint8_t ultimoPWM_Normal  = 0;


// ======================================================
// Liga a esteira no sentido normal (rampa de aceleracao)
// ======================================================
void ligarEsteira() {
  Serial.println(F("[Esteira] Acelerando..."));

  digitalWrite(R_EN, HIGH);
  digitalWrite(L_EN, HIGH);

  SoftPWMSet(RPWM, 0);
  ultimoPWM_Reverso = 0;

  for (uint8_t pwm = pwmInicial; pwm <= pwmFinal; pwm += incrementoPWM) {
    SoftPWMSet(LPWM, pwm);
    ultimoPWM_Normal = pwm;
    delay(intervaloRampa);
  }

  Serial.println(F("[Esteira] OK."));
}


// ======================================================
// Desliga com rampa suave
// ======================================================
void desligarEsteira() {
  Serial.println(F("[Esteira] Desacelerando..."));

  // Loop usa int para que o decremento possa cruzar zero sem wrap-around de uint8_t.
  uint8_t pwmMax = (ultimoPWM_Reverso > ultimoPWM_Normal)
                   ? ultimoPWM_Reverso : ultimoPWM_Normal;

  for (int pwm = (int)pwmMax; pwm >= 0; pwm -= (int)incrementoPWM) {
    if ((uint8_t)pwm <= ultimoPWM_Reverso) {
      SoftPWMSet(RPWM, (uint8_t)pwm);
      ultimoPWM_Reverso = (uint8_t)pwm;
    }
    if ((uint8_t)pwm <= ultimoPWM_Normal) {
      SoftPWMSet(LPWM, (uint8_t)pwm);
      ultimoPWM_Normal = (uint8_t)pwm;
    }
    delay(intervaloRampa);
  }

  SoftPWMSet(RPWM, 0);
  SoftPWMSet(LPWM, 0);
  ultimoPWM_Reverso = 0;
  ultimoPWM_Normal  = 0;

  Serial.println(F("[Esteira] Parada."));
}


// ======================================================
// Envia comando I2C ao Slave e le resposta em respostaI2C[].
// Chama atualizarEstados() automaticamente se cmd == CMD_STATUS.
// ======================================================
void enviarComandoI2C(const char* cmd) {
  Wire.beginTransmission(8);
  Wire.write(cmd);
  Wire.endTransmission();
  delay(10);

  Wire.requestFrom(8, 32);
  uint8_t i = 0;
  while (Wire.available() && i < 32) {
    respostaI2C[i++] = (char)Wire.read();
  }
  respostaI2C[i] = '\0';

  if (i > 0) {
    Serial.print(F("Status: "));
    Serial.println(respostaI2C);
  }

  if (strcmp(cmd, CMD_STATUS) == 0 && i > 0) {
    atualizarEstados(respostaI2C);
  }
}


// ======================================================
// Atualiza estados globais a partir da resposta do Slave.
// Formato esperado: "A2:0,A3:2,A4:2,A5:2"
// Cada valor e um digito unico (0, 1 ou 2).
// ======================================================
void atualizarEstados(const char* s) {
  const char* p;
  p = strstr(s, "A2:"); if (p) estadoA2 = (uint8_t)(p[3] - '0');
  p = strstr(s, "A3:"); if (p) estadoA3 = (uint8_t)(p[3] - '0');
  p = strstr(s, "A4:"); if (p) estadoA4 = (uint8_t)(p[3] - '0');
  p = strstr(s, "A5:"); if (p) estadoA5 = (uint8_t)(p[3] - '0');
}


// ======================================================
// Monitora e corrige posicao de um atuador a cada ciclo.
// Parametros const char* eliminam copias de String na chamada.
// ======================================================
void verificarAtuador(const char* nome, uint8_t status,
                      uint8_t pinoC1, uint8_t pinoC2,
                      const char* cmdRetornar, const char* cmdParar,
                      uint8_t& statusRef) {
  delay(100);
  enviarComandoI2C(CMD_STATUS);
  delay(50);

  bool c1 = (digitalRead(pinoC1) == LOW);
  bool c2 = (digitalRead(pinoC2) == LOW);

  if (status == PARADO && !c1 && !c2) {
    Serial.print(nome); Serial.println(F(": parado no meio - retornando..."));
    enviarComandoI2C(cmdRetornar);
    delay(50);
    return;
  }

  if (status == PARADO && c2) {
    Serial.print(nome); Serial.println(F(": parado no topo - retornando..."));
    enviarComandoI2C(cmdRetornar);
    delay(50);
    return;
  }

  if (status == RETORNANDO) {
    if (c1) {
      Serial.print(nome); Serial.println(F(": na base - parando..."));
      delay(120);
      enviarComandoI2C(cmdParar);
      delay(120);
    } else {
      Serial.print(nome); Serial.println(F(": retornando..."));
    }
    return;
  }

  if (status == PARADO && c1) {
    // Em repouso na base — sem print para reduzir carga serial no loop.
    return;
  }

  if (status == AVANCANDO && c2) {
    Serial.print(nome); Serial.println(F(": no topo - retornando..."));
    delay(100);
    enviarComandoI2C(cmdRetornar);
    delay(100);
    return;
  }

  if (status == AVANCANDO || status == RETORNANDO) return;
}


// ======================================================
// Verifica se o Slave I2C esta respondendo.
// Sem alocacao de String: conta bytes recebidos e descarta.
// ======================================================
bool verificarEscravoConectado() {
  Wire.beginTransmission(8);
  Wire.write(CMD_STATUS);
  Wire.endTransmission();
  delay(15);
  Wire.requestFrom(8, 32);
  uint8_t n = 0;
  while (Wire.available()) { Wire.read(); n++; }
  return n > 0;
}


// ======================================================
// Le uma linha da serial para bufferSerial[].
// Retorna true se leu pelo menos 1 caractere util.
// Usa readBytesUntil (sem String) e trata \r de finais Windows.
// ======================================================
bool lerLinhaSerial() {
  if (!Serial.available()) return false;
  uint8_t len = (uint8_t)Serial.readBytesUntil('\n', bufferSerial,
                                                sizeof(bufferSerial) - 1);
  bufferSerial[len] = '\0';
  // Remove \r de finais de linha Windows (\r\n)
  if (len > 0 && bufferSerial[len - 1] == '\r') bufferSerial[--len] = '\0';
  return len > 0;
}


// ======================================================
// Processa comando CONFIG_ATRASOS enviado pelo software Python.
//
// Formato do payload (tudo após "CONFIG_ATRASOS:"):
//   "<vidro>:<papel>:<plastico>:<metal>:<passagem_livre>"
//   ex.: "4700:6260:7900:9000:10000"
//
// Validação: cada valor deve ser inteiro sem sinal no intervalo [0, 30000].
// Resposta serial:
//   "CONF_OK"              → valores aceitos e aplicados em memória (sem reflash)
//   "CONF_ERRO:<motivo>"   → rejeitado; atrasos anteriores permanecem intactos
// ======================================================
void processarConfigAtrasos(const char* payload) {
  char* ptr = (char*)payload;
  char* fim;
  unsigned long vals[5];

  for (uint8_t i = 0; i < 5; i++) {
    vals[i] = strtoul(ptr, &fim, 10);

    // strtoul retorna ptr original quando não há dígito válido
    if (fim == ptr) {
      Serial.println(F("CONF_ERRO:formato_invalido"));
      return;
    }
    // Rejeita valores acima do máximo permitido pela UI (30 000 ms)
    if (vals[i] > 30000UL) {
      Serial.println(F("CONF_ERRO:valor_invalido"));
      return;
    }
    // Após o último valor não deve haver separador — apenas fim de string
    if (i < 4) {
      if (*fim != ':') {
        Serial.println(F("CONF_ERRO:separador_ausente"));
        return;
      }
      ptr = fim + 1;  // avança para o próximo valor
    }
  }

  // Todos os valores validados — aplica em memória
  atrasoVidro         = (unsigned int)vals[0];
  atrasoPapel         = (unsigned int)vals[1];
  atrasoPlastico      = (unsigned int)vals[2];
  atrasoMetal         = (unsigned int)vals[3];
  atrasoPassagemLivre = (unsigned int)vals[4];

  Serial.println(F("CONF_OK"));
}


// ======================================================
void setup() {
  Wire.begin();
  Serial.begin(9600);

  for (uint8_t i = 0; i < sizeof(pinosChaveFim); i++)
    pinMode(pinosChaveFim[i], INPUT_PULLUP);

  pinMode(R_EN, OUTPUT);
  pinMode(L_EN, OUTPUT);
  pinMode(RPWM, OUTPUT);
  pinMode(LPWM, OUTPUT);

  digitalWrite(R_EN, HIGH);
  digitalWrite(L_EN, HIGH);

  SoftPWMBegin();
  SoftPWMSet(RPWM, 0);
  SoftPWMSet(LPWM, 0);

  desligarEsteira();
}


// ======================================================
void loop() {

  // Limpa buffer a cada ciclo para evitar reprocessamento
  bufferSerial[0] = '\0';

  // ── Leitura serial PRIORITARIA ────────────────────────────────────────────
  // PING_RECYCLEAI e tratado ANTES de qualquer verificarAtuador.
  // Motivacao: verificarAtuador x4 + delays acrescentava ~950 ms de latencia
  // ao PONG, causando timeout no handshake do pre-operacao.
  // Com leitura prioritaria, o PONG chega em < 50 ms.
  bool temCmdPrecoce = lerLinhaSerial();

  if (temCmdPrecoce && strcasecmp(bufferSerial, "PING_RECYCLEAI") == 0) {
    if (verificarEscravoConectado()) {
      Serial.println(F("PONG_RECYCLEAI:OK"));
    } else {
      Serial.println(F("PONG_RECYCLEAI:SLAVE_ERROR"));
    }
    return;  // nao executa verificarAtuador nem delay neste ciclo
  }

  // ── Watchdog do modo diagnóstico ─────────────────────────────────────────
  // Sai automaticamente se nenhum comando serial chegar por TIMEOUT_DIAG_MS ms.
  // Protege contra abandono de sessao sem DIAG_EXIT (crash, kill, powercut).
  // Apos a saida, verificarAtuador() retoma no mesmo ciclo — resincronizando
  // os atuadores sem intervencao manual.
  if (em_modo_diagnostico &&
      (millis() - ultimoComandoSerial) >= TIMEOUT_DIAG_MS) {
    Serial.println(F("[Diag] WDT: timeout — saindo do modo diagnostico."));
    em_modo_diagnostico = false;
    enviarComandoI2C(CMD_STATUS);
  }

  // ── Verificacao continua dos atuadores ────────────────────────────────────
  // Em modo diagnostico, verificarAtuador() e suspensa: ela reenviaria CMD_AxR
  // se o atuador parasse no meio do stroke, brigando contra DIAG_AxP do operador.
  if (!em_modo_diagnostico) {
    verificarAtuador("A2", estadoA2, A2C1, A2C2, CMD_A2R, CMD_A2P, estadoA2);
    verificarAtuador("A3", estadoA3, A3C1, A3C2, CMD_A3R, CMD_A3P, estadoA3);
    verificarAtuador("A4", estadoA4, A4C1, A4C2, CMD_A4R, CMD_A4P, estadoA4);
    verificarAtuador("A5", estadoA5, A5C1, A5C2, CMD_A5R, CMD_A5P, estadoA5);
  }

  // Reutiliza cmd precoce (nao-PING) ou le novo apos os verificarAtuador
  if (!temCmdPrecoce) lerLinhaSerial();

  if (bufferSerial[0] != '\0') {

    Serial.print(F("[Cmd] "));
    Serial.println(bufferSerial);

    // Atualiza timestamp do ultimo comando serial — reseta o watchdog de diagnostico
    ultimoComandoSerial = millis();

    // =========================================================================
    // ===== BLOCO OPERACIONAL =================================================
    // =========================================================================

    if (strcasecmp(bufferSerial, "Vidro") == 0) {
      Serial.println(F("[Cmd] Vidro -> A2"));
      enviarComandoI2C(CMD_STATUS);
      delay(30);

      if (estadoA2 == PARADO && digitalRead(A2C1) == LOW && digitalRead(A2C2) == HIGH) {
        ligarEsteira();
        delay(atrasoVidro);
        Serial.println(F("[A2] Avancando..."));
        enviarComandoI2C(CMD_A2A);
        enviarComandoI2C(CMD_STATUS);
        delay(30);
        desligarEsteira();
      }
    }

    else if (strcasecmp(bufferSerial, "Papel") == 0) {
      Serial.println(F("[Cmd] Papel -> A3"));
      enviarComandoI2C(CMD_STATUS);
      delay(30);

      if (estadoA3 == PARADO && digitalRead(A3C1) == LOW && digitalRead(A3C2) == HIGH) {
        ligarEsteira();
        delay(atrasoPapel);
        Serial.println(F("[A3] Avancando..."));
        enviarComandoI2C(CMD_A3A);
        enviarComandoI2C(CMD_STATUS);
        delay(30);
        desligarEsteira();
      }
    }

    else if (strcasecmp(bufferSerial, "Plastico") == 0) {
      Serial.println(F("[Cmd] Plastico -> A4"));
      enviarComandoI2C(CMD_STATUS);
      delay(30);

      if (estadoA4 == PARADO && digitalRead(A4C1) == LOW && digitalRead(A4C2) == HIGH) {
        ligarEsteira();
        delay(atrasoPlastico);
        Serial.println(F("[A4] Avancando..."));
        enviarComandoI2C(CMD_A4A);
        enviarComandoI2C(CMD_STATUS);
        delay(30);
        desligarEsteira();
      }
    }

    else if (strcasecmp(bufferSerial, "Metal") == 0) {
      Serial.println(F("[Cmd] Metal -> A5"));
      enviarComandoI2C(CMD_STATUS);
      delay(30);

      if (estadoA5 == PARADO && digitalRead(A5C1) == LOW && digitalRead(A5C2) == HIGH) {
        ligarEsteira();
        delay(atrasoMetal);
        Serial.println(F("[A5] Avancando..."));
        enviarComandoI2C(CMD_A5A);
        enviarComandoI2C(CMD_STATUS);
        delay(30);
        desligarEsteira();
      }
    }

    else if (strcasecmp(bufferSerial, "PASSAGEM_LIVRE") == 0) {
      Serial.println(F("[Cmd] Passagem livre"));
      ligarEsteira();
      delay(atrasoPassagemLivre);
      desligarEsteira();
    }

    // Configuração dinâmica de atrasos — sem necessidade de reflash.
    // Prefixo "CONFIG_ATRASOS:" identificado por strncasecmp (15 chars).
    // Payload após o ":" é passado para processarConfigAtrasos().
    else if (strncasecmp(bufferSerial, "CONFIG_ATRASOS:", 15) == 0) {
      processarConfigAtrasos(bufferSerial + 15);
    }

    else if (strcasecmp(bufferSerial, "Status") == 0) {
      enviarComandoI2C(CMD_STATUS);
    }

    else if (strcasecmp(bufferSerial, "Parar") == 0) {
      enviarComandoI2C(CMD_A2P);
    }

    else if (strcasecmp(bufferSerial, "Retornar") == 0) {
      enviarComandoI2C(CMD_A2R);
    }

    // PING_RECYCLEAI fallback: cobre chegada entre verificarAtuador e lerLinhaSerial
    else if (strcasecmp(bufferSerial, "PING_RECYCLEAI") == 0) {
      if (verificarEscravoConectado()) {
        Serial.println(F("PONG_RECYCLEAI:OK"));
      } else {
        Serial.println(F("PONG_RECYCLEAI:SLAVE_ERROR"));
      }
    }

    // =========================================================================
    // ===== BLOCO DE DIAGNOSTICO MANUAL =======================================
    // =========================================================================
    // Comandos DIAG_* sao exclusivos para diagnostico.
    // Nao combinam esteira + atuador + delay.
    // Isolamento total: nenhum interfere na triagem operacional.

    else if (strcasecmp(bufferSerial, "DIAG_ENTER") == 0) {
      em_modo_diagnostico = true;
      Serial.println(F("MODO_DIAG:ON"));
    }

    else if (strcasecmp(bufferSerial, "DIAG_EXIT") == 0) {
      em_modo_diagnostico = false;
      enviarComandoI2C(CMD_STATUS); // ressincroniza estados antes de retomar verificarAtuador
      Serial.println(F("MODO_DIAG:OFF"));
    }

    else if (strcasecmp(bufferSerial, "DIAG_ESTEIRA_ON") == 0) {
      Serial.println(F("[Diag] Esteira ON..."));
      ligarEsteira();
    }

    else if (strcasecmp(bufferSerial, "DIAG_ESTEIRA_OFF") == 0) {
      Serial.println(F("[Diag] Esteira OFF..."));
      desligarEsteira();
    }

    // Estado local atualizado imediatamente apos cada DIAG_Ax* para evitar
    // que verificarAtuador() (ao retomar) aja sobre estado desatualizado.
    else if (strcasecmp(bufferSerial, "DIAG_A2A") == 0) { Serial.println(F("[Diag] A2->AVA")); enviarComandoI2C(CMD_A2A); estadoA2 = AVANCANDO; }
    else if (strcasecmp(bufferSerial, "DIAG_A2R") == 0) { Serial.println(F("[Diag] A2->RET")); enviarComandoI2C(CMD_A2R); estadoA2 = RETORNANDO; }
    else if (strcasecmp(bufferSerial, "DIAG_A2P") == 0) { Serial.println(F("[Diag] A2->PAR")); enviarComandoI2C(CMD_A2P); estadoA2 = PARADO; }

    else if (strcasecmp(bufferSerial, "DIAG_A3A") == 0) { Serial.println(F("[Diag] A3->AVA")); enviarComandoI2C(CMD_A3A); estadoA3 = AVANCANDO; }
    else if (strcasecmp(bufferSerial, "DIAG_A3R") == 0) { Serial.println(F("[Diag] A3->RET")); enviarComandoI2C(CMD_A3R); estadoA3 = RETORNANDO; }
    else if (strcasecmp(bufferSerial, "DIAG_A3P") == 0) { Serial.println(F("[Diag] A3->PAR")); enviarComandoI2C(CMD_A3P); estadoA3 = PARADO; }

    else if (strcasecmp(bufferSerial, "DIAG_A4A") == 0) { Serial.println(F("[Diag] A4->AVA")); enviarComandoI2C(CMD_A4A); estadoA4 = AVANCANDO; }
    else if (strcasecmp(bufferSerial, "DIAG_A4R") == 0) { Serial.println(F("[Diag] A4->RET")); enviarComandoI2C(CMD_A4R); estadoA4 = RETORNANDO; }
    else if (strcasecmp(bufferSerial, "DIAG_A4P") == 0) { Serial.println(F("[Diag] A4->PAR")); enviarComandoI2C(CMD_A4P); estadoA4 = PARADO; }

    else if (strcasecmp(bufferSerial, "DIAG_A5A") == 0) { Serial.println(F("[Diag] A5->AVA")); enviarComandoI2C(CMD_A5A); estadoA5 = AVANCANDO; }
    else if (strcasecmp(bufferSerial, "DIAG_A5R") == 0) { Serial.println(F("[Diag] A5->RET")); enviarComandoI2C(CMD_A5R); estadoA5 = RETORNANDO; }
    else if (strcasecmp(bufferSerial, "DIAG_A5P") == 0) { Serial.println(F("[Diag] A5->PAR")); enviarComandoI2C(CMD_A5P); estadoA5 = PARADO; }

    else if (strcasecmp(bufferSerial, "DIAG_PARAR_TUDO") == 0) {
      Serial.println(F("[Diag] PARAR TUDO..."));
      desligarEsteira();
      enviarComandoI2C(CMD_A2P); delay(50);
      enviarComandoI2C(CMD_A3P); delay(50);
      enviarComandoI2C(CMD_A4P); delay(50);
      enviarComandoI2C(CMD_A5P); delay(50);
      // Atualiza estado local para garantir que verificarAtuador() nao
      // reenicie movimento ao retomar apos DIAG_EXIT.
      estadoA2 = PARADO; estadoA3 = PARADO;
      estadoA4 = PARADO; estadoA5 = PARADO;
      enviarComandoI2C(CMD_STATUS);
      Serial.println(F("[Diag] Parada segura OK."));
    }

    else if (strcasecmp(bufferSerial, "DIAG_CHAVES") == 0) {
      // Convencao: 1=ativada (LOW/INPUT_PULLUP pressionado), 0=livre (HIGH).
      // Saida por Serial.print() sequencial: evita alocar String na heap.
      // Formato preservado: "DIAG:CHAVES:A2C1=X,A2C2=X,...,A5C2=X\r\n"
      Serial.print(F("DIAG:CHAVES:"));
      Serial.print(F("A2C1=")); Serial.print(digitalRead(A2C1) == LOW ? 1 : 0); Serial.print(F(","));
      Serial.print(F("A2C2=")); Serial.print(digitalRead(A2C2) == LOW ? 1 : 0); Serial.print(F(","));
      Serial.print(F("A3C1=")); Serial.print(digitalRead(A3C1) == LOW ? 1 : 0); Serial.print(F(","));
      Serial.print(F("A3C2=")); Serial.print(digitalRead(A3C2) == LOW ? 1 : 0); Serial.print(F(","));
      Serial.print(F("A4C1=")); Serial.print(digitalRead(A4C1) == LOW ? 1 : 0); Serial.print(F(","));
      Serial.print(F("A4C2=")); Serial.print(digitalRead(A4C2) == LOW ? 1 : 0); Serial.print(F(","));
      Serial.print(F("A5C1=")); Serial.print(digitalRead(A5C1) == LOW ? 1 : 0); Serial.print(F(","));
      Serial.print(F("A5C2=")); Serial.println(digitalRead(A5C2) == LOW ? 1 : 0);
    }

    else {
      Serial.println(F("Cmd desconhecido."));
    }

  }

  delay(100);
}
