; Inno Setup 6 — RecycleAI-Station Installer
;
; Pré-requisito: bundle já gerado em dist\RecycleAI-Station\
; Executar de dentro da pasta triagem-residuos\:
;   "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" install\recycleai_setup.iss
;
; Gera: dist\RecycleAI-Station-Setup-1.0.0.exe
;
; Pré-requisito verificado pelo wizard:
;   Visual C++ Redistributable 2015-2022 (x64)
;   Detectado via: HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64 → Installed=1
;   Se ausente: aviso claro + opção de abortar ou continuar por conta e risco.

#define AppName      "RecycleAI-Station"
#define AppVersion   "1.0.0"
#define AppPublisher "UNIP — TCC Ciência da Computação"
#define AppURL       "https://www.unip.br"
#define AppExeName   "RecycleAI-Station.exe"
#define BundleDir    "..\dist\RecycleAI-Station"

[Setup]
AppId={{A7B3C2D1-E4F5-4A6B-8C9D-0E1F2A3B4C5D}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
AppComments=Equipe: CLAYTON MACIEL TANCREDO, GIVANILDO SANTANA RIBEIRO, LUIS FELIPE NASSIF. Orientador: Prof. Marco Gomes. UNIP — Universidade Paulista — Ciência da Computação.
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=..\dist
OutputBaseFilename=RecycleAI-Station-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\app\assets\icons\recycleai_icon.ico
UninstallDisplayIcon={app}\{#AppExeName}
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
DisableProgramGroupPage=yes
; Solicitar privilégios de admin para instalar em Program Files e
; garantir permissões nas pastas de dados graváveis
PrivilegesRequired=admin
; Mostrar EULA mínima
LicenseFile=license.txt
; Minimizar splash pages — instalador direto ao ponto
DisableWelcomePage=no
DisableReadyPage=no

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon";  Description: "Criar atalho na Área de Trabalho"; GroupDescription: "Atalhos:"; Flags: unchecked

[Files]
; ── Executável principal ──────────────────────────────────────────────────────
Source: "{#BundleDir}\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion

; ── Ícone oficial (referenciado explicitamente nos atalhos) ───────────────────
Source: "..\app\assets\icons\recycleai_icon.ico"; DestDir: "{app}"; Flags: ignoreversion

; ── Runtime Python bundled (_internal/) — somente-leitura ────────────────────
; excludefiles exclui artefatos de teste que não devem ir ao prod
Source: "{#BundleDir}\_internal\*"; DestDir: "{app}\_internal"; \
    Flags: ignoreversion recursesubdirs createallsubdirs; \
    Excludes: "*.pyc,__pycache__"

; NOTA: data\ e runtime_inferencia\modelos_importados\ NÃO são copiados do bundle de dev —
;       são criados vazios abaixo e preenchidos pelo bootstrap no primeiro boot.

[Dirs]
; Diretório gravável para o banco SQLite (criado no 1.º boot)
Name: "{app}\data";                                       Permissions: everyone-full
; Diretório gravável para modelos importados pelo operador
Name: "{app}\runtime_inferencia\modelos_importados";      Permissions: everyone-full

[Icons]
Name: "{group}\{#AppName}";             Filename: "{app}\{#AppExeName}"; \
    IconFilename: "{app}\recycleai_icon.ico"
Name: "{group}\Desinstalar {#AppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#AppName}";     Filename: "{app}\{#AppExeName}"; \
    IconFilename: "{app}\recycleai_icon.ico"; \
    Tasks: desktopicon

[Run]
; Refresh do cache de ícones do Windows — envolvido em cmd.exe para ser silencioso
; cmd.exe sempre existe; se ie4uinit.exe estiver ausente, cmd sai com erro mas
; o Inno Setup não exibe diálogo para falha de saída do processo filho.
Filename: "{sys}\cmd.exe"; Parameters: "/c ie4uinit.exe -show"; \
    Flags: runhidden
; Nenhum post-install externo: o bootstrap roda dentro do próprio exe no 1.º boot.
; A linha abaixo abre o app ao final do wizard (opcional, unchecked por padrão).
Filename: "{app}\{#AppExeName}"; \
    Description: "Iniciar {#AppName} agora"; \
    Flags: nowait postinstall skipifsilent unchecked

[UninstallDelete]
; Remove dados criados em runtime (DB, modelos importados) na desinstalação
Type: filesandordirs; Name: "{app}\data"
Type: filesandordirs; Name: "{app}\runtime_inferencia"

; ─────────────────────────────────────────────────────────────────────────────
; [Code] — Verificação de pré-requisito: VC++ Redistributable 2015-2022 (x64)
;
; Critério de detecção (dois caminhos, fallback para WOW6432Node):
;   HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64 → Installed (DWORD) = 1
;   HKLM\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\X64 → idem
;
; Comportamento:
;   Presente  → instalação prossegue normalmente, sem interrupção.
;   Ausente   → MsgBox com URL de download e opção Sim/Não:
;               Sim = continua (risco do operador)
;               Não = aborta o setup antes de copiar qualquer arquivo.
; ─────────────────────────────────────────────────────────────────────────────
[Code]

{ Retorna True se o VC++ Redist 2015-2022 (x64) estiver instalado. }
function VCRedist2022X64Installed(): Boolean;
var
  Installed: Cardinal;
begin
  { Caminho primário — sistemas 64-bit modernos }
  Result := RegQueryDWordValue(
    HKLM,
    'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64',
    'Installed',
    Installed
  ) and (Installed = 1);

  { Fallback — alguns sistemas gravam apenas em WOW6432Node }
  if not Result then
    Result := RegQueryDWordValue(
      HKLM,
      'SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\X64',
      'Installed',
      Installed
    ) and (Installed = 1);
end;

{ Executado antes do primeiro wizard page. }
function InitializeSetup(): Boolean;
var
  Resposta: Integer;
begin
  Result := True;  { padrão: prosseguir }

  if not VCRedist2022X64Installed() then
  begin
    Resposta := MsgBox(
      'PRE-REQUISITO NAO ENCONTRADO' + #13#10 +
      '--------------------------------------------' + #13#10 +
      'Visual C++ Redistributable 2015-2022 (x64)' + #13#10 +
      'nao foi detectado neste computador.' + #13#10 +
      'O RecycleAI-Station pode falhar ao iniciar.' + #13#10 +
      '' + #13#10 +
      'Baixe e instale antes de continuar:' + #13#10 +
      'https://aka.ms/vs/17/release/vc_redist.x64.exe' + #13#10 +
      '' + #13#10 +
      'Deseja continuar mesmo assim?' + #13#10 +
      '(NAO = abortar e instalar o pre-requisito primeiro)',
      mbConfirmation,
      MB_YESNO or MB_DEFBUTTON2
    );

    if Resposta = IDNO then
      Result := False;  { aborta setup }
  end;
end;
