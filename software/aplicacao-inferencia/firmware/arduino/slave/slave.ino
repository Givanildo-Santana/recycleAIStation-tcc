
#include <Wire.h>
#include <SoftPWM.h>

// ===== COMANDOS =====
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

const uint8_t L_EN[] = {2, 6, 10, A0};
const uint8_t R_EN[] = {3, 7, 11, A1};
const uint8_t LPWM[] = {5, 8, 12, A2};
const uint8_t RPWM[] = {4, 9, 13, A3};

char bufferCmd[12];
char bufferStatus[20];

// ===== LOG DIFERIDO (ISR-SAFE) =====
// Serial.print() e inseguro dentro de Wire.onReceive() (ISR do TWI):
// interrupcoes internas do UART podem nao funcionar em contexto de ISR,
// causando bloqueio se o buffer de transmissao estiver cheio.
// Solucao: setamos _logPendente em recebeDados() e imprimimos em loop().
volatile bool _logPendente = false;

class Atuador {
  public:
    uint8_t L_EN, R_EN, LPWM, RPWM, pwm;

    enum Status : uint8_t { AVANCANDO = 0, RETORNANDO = 1, PARADO = 2 };
    Status estado;

    Atuador(uint8_t l_en, uint8_t r_en, uint8_t lpwm, uint8_t rpwm,
            uint8_t pwmDefault) {
      L_EN  = l_en;
      R_EN  = r_en;
      LPWM  = lpwm;
      RPWM  = rpwm;
      pwm   = pwmDefault;
      estado = PARADO;

      pinMode(L_EN, OUTPUT);
      pinMode(R_EN, OUTPUT);
      SoftPWMSet(LPWM, 0);
      SoftPWMSet(RPWM, 0);
    }

    void avancar() {
      digitalWrite(L_EN, HIGH);
      digitalWrite(R_EN, HIGH);
      SoftPWMSet(LPWM, pwm);
      SoftPWMSet(RPWM, 0);
      estado = AVANCANDO;
    }

    void retornar() {
      digitalWrite(L_EN, HIGH);
      digitalWrite(R_EN, HIGH);
      SoftPWMSet(LPWM, 0);
      SoftPWMSet(RPWM, pwm);
      estado = RETORNANDO;
    }

    void parar() {
      SoftPWMSet(LPWM, 0);
      SoftPWMSet(RPWM, 0);
      estado = PARADO;
    }
};

Atuador atuador2(L_EN[0], R_EN[0], LPWM[0], RPWM[0], 150);
Atuador atuador3(L_EN[1], R_EN[1], LPWM[1], RPWM[1], 150);
Atuador atuador4(L_EN[2], R_EN[2], LPWM[2], RPWM[2], 250); // A4: pwm=250 (carga mec. diferente)
Atuador atuador5(L_EN[3], R_EN[3], LPWM[3], RPWM[3], 150);


void atualizaStatus() {
  uint8_t p = 0;

  // A2
  bufferStatus[p++] = 'A'; bufferStatus[p++] = '2'; bufferStatus[p++] = ':';
  bufferStatus[p++] = '0' + (uint8_t)atuador2.estado;
  bufferStatus[p++] = ',';
  // A3
  bufferStatus[p++] = 'A'; bufferStatus[p++] = '3'; bufferStatus[p++] = ':';
  bufferStatus[p++] = '0' + (uint8_t)atuador3.estado;
  bufferStatus[p++] = ',';
  // A4
  bufferStatus[p++] = 'A'; bufferStatus[p++] = '4'; bufferStatus[p++] = ':';
  bufferStatus[p++] = '0' + (uint8_t)atuador4.estado;
  bufferStatus[p++] = ',';
  // A5
  bufferStatus[p++] = 'A'; bufferStatus[p++] = '5'; bufferStatus[p++] = ':';
  bufferStatus[p++] = '0' + (uint8_t)atuador5.estado;
  bufferStatus[p]   = '\0'; // p = 19
  // Impressao via Serial feita em loop() (ISR-safe) — ver _logPendente.
}


void recebeDados(int quantidade) {
  uint8_t i = 0;
  while (Wire.available() && i < (uint8_t)(sizeof(bufferCmd) - 1)) {
    bufferCmd[i++] = (char)Wire.read();
  }
  while (Wire.available()) Wire.read(); 
  bufferCmd[i] = '\0';
  _logPendente = true;  // sinaliza loop() para imprimir cmd + status

  // === Executa acao ===
  if      (strcmp(bufferCmd, CMD_A2A) == 0) atuador2.avancar();
  else if (strcmp(bufferCmd, CMD_A2R) == 0) atuador2.retornar();
  else if (strcmp(bufferCmd, CMD_A2P) == 0) atuador2.parar();
  else if (strcmp(bufferCmd, CMD_A3A) == 0) atuador3.avancar();
  else if (strcmp(bufferCmd, CMD_A3R) == 0) atuador3.retornar();
  else if (strcmp(bufferCmd, CMD_A3P) == 0) atuador3.parar();
  else if (strcmp(bufferCmd, CMD_A4A) == 0) atuador4.avancar();
  else if (strcmp(bufferCmd, CMD_A4R) == 0) atuador4.retornar();
  else if (strcmp(bufferCmd, CMD_A4P) == 0) atuador4.parar();
  else if (strcmp(bufferCmd, CMD_A5A) == 0) atuador5.avancar();
  else if (strcmp(bufferCmd, CMD_A5R) == 0) atuador5.retornar();
  else if (strcmp(bufferCmd, CMD_A5P) == 0) atuador5.parar();

  atualizaStatus();
}

void enviaDados() {
  Wire.write(bufferStatus);
}


void setup() {
  Wire.begin(8); 
  Serial.begin(9600);
  SoftPWMBegin();

  Wire.onReceive(recebeDados);
  Wire.onRequest(enviaDados);

  atualizaStatus();
  Serial.println(F("[Slave] Pronto. Aguardando comandos via I2C."));
}


void loop() {
  // Impressao diferida: evita Serial.print() dentro da ISR Wire.
  // _logPendente e setado por recebeDados(); bufferCmd e bufferStatus
  // sao validos apos retorno da ISR, antes do proximo evento Wire.
  if (_logPendente) {
    _logPendente = false;
    Serial.print(F("[Slave] Cmd: "));
    Serial.println(bufferCmd);
    Serial.print(F("[Slave] Status: "));
    Serial.println(bufferStatus);
  }
}
