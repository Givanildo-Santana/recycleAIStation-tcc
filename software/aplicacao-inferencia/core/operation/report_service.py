"""
Servico de relatorios operacionais.

Constroi dicionarios estruturados prontos para exportacao em CSV ou PDF.

Dois tipos:
  - Operacional padrao: dados de negocio (totais, operador, modelo, etc.)
  - Tecnico ampliado:   superset do operacional + metricas de inferencia
                        (destinado ao perfil maintenance)

Exportacao disponivel:
  - CSV: via stdlib csv (sem dependencia externa)
  - PDF: via fpdf2 (instalado em runtime_inferencia/venv)

Contrato publico:
  get_operational_report(op_session_id) -> dict
  get_technical_report(op_session_id)   -> dict   # superset
  export_csv(report_data, path)         -> Path
  export_pdf(report_data, path)         -> Path
  build_report(op_session_id, profile)  -> dict   # escolhe tipo pelo perfil
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from db.repositories import session_repo

# Larguras de coluna (mm) para tabelas do PDF
_COL_KEY = 75
_COL_VAL = 110


# ─────────────────────────────────────────── Construtores de relatorio ────────

def get_operational_report(op_session_id: int) -> dict[str, Any]:
    """
    Relatorio operacional padrao.

    Campos:
      report_type, op_session_id, session_uuid
      operator, profile
      started_at, ended_at, duration_s, duration_fmt
      model_name, model_id
      total_items, counts_by_class
      status, notes, errors
      generated_at
    """
    row = session_repo.get_by_id(op_session_id)
    if row is None:
        raise ValueError(f"op_session_id={op_session_id} nao encontrada")

    events = session_repo.get_events(op_session_id)
    errors = [
        {"event_type": e["event_type"], "detail": e["detail"], "ts": e["ts"]}
        for e in events
        if e["event_type"] in ("ERROR", "WARNING")
    ]
    counts: dict[str, int] = (
        json.loads(row["counts_json"]) if row["counts_json"] else {}
    )

    return {
        "report_type":     "operational",
        "op_session_id":   row["id"],
        "session_uuid":    row["session_id"],
        "operator":        row["user_login"],
        "profile":         row["profile"],
        "started_at":      row["started_at"],
        "ended_at":        row["ended_at"],
        "duration_s":      row["duration_s"],
        "duration_fmt":    _fmt_duration(row["duration_s"]),
        "model_name":      row["model_name"],
        "model_id":        row["model_id"],
        "total_items":     row["total_items"],
        "counts_by_class": counts,
        "status":          row["status"],
        "notes":           row["notes"],
        "errors":          errors,
        "generated_at":    datetime.now().isoformat(timespec="seconds"),
    }


def get_technical_report(op_session_id: int) -> dict[str, Any]:
    """
    Relatorio tecnico ampliado (superset do operacional).
    Destina-se ao perfil maintenance.

    Adiciona:
      config_snapshot — configuracoes do sistema no inicio da sessao
      metrics         — metricas de inferencia (None se nao coletadas)
    """
    report = get_operational_report(op_session_id)
    report["report_type"] = "technical"

    row = session_repo.get_by_id(op_session_id)
    snapshot = json.loads(row["config_snapshot"]) if row["config_snapshot"] else {}
    report["config_snapshot"] = snapshot

    m = session_repo.get_metrics(op_session_id)
    if m:
        report["metrics"] = {
            "infer_count":    m["infer_count"],
            "infer_time_min": m["infer_time_min"],
            "infer_time_avg": m["infer_time_avg"],
            "infer_time_max": m["infer_time_max"],
            "conf_min":       m["conf_min"],
            "conf_avg":       m["conf_avg"],
            "conf_max":       m["conf_max"],
            "fps_avg":        m["fps_avg"],
            "serial_port":    m["serial_port"],
            "model_path":     m["model_path"],
            "extra":          json.loads(m["extra_json"]) if m["extra_json"] else {},
        }
    else:
        report["metrics"] = None

    return report


def build_report(op_session_id: int, profile: str) -> dict[str, Any]:
    """
    Atalho: retorna o tipo correto de relatorio pelo perfil.
      operator    -> get_operational_report()
      maintenance -> get_technical_report()
    """
    if profile == "maintenance":
        return get_technical_report(op_session_id)
    return get_operational_report(op_session_id)


# ──────────────────────────────────────────────────────── Exportacao CSV ──────

def export_csv(report_data: dict, output_path: str | Path) -> Path:
    """
    Exporta relatorio em CSV.

    Estrutura (4 secoes):
      1. Dados gerais da sessao
      2. Totais por classe
      3. Erros/alertas (se houver)
      4. Metricas tecnicas + snapshot de config (somente technical)

    Encoding: UTF-8 com BOM (utf-8-sig) — compativel com Excel.

    Returns: Path do arquivo gerado.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)

        w.writerow(["# RecycleAI-Station - Relatorio Operacional"])
        w.writerow(["# Tipo", report_data.get("report_type", "?")])
        w.writerow(["# Gerado em", report_data.get("generated_at", "")])
        w.writerow([])

        # Secao 1: dados gerais
        w.writerow(["=== DADOS DA SESSAO ==="])
        w.writerow(["Campo", "Valor"])
        for key, label in [
            ("op_session_id", "ID da Sessao"),
            ("session_uuid",  "UUID da Sessao"),
            ("operator",      "Operador"),
            ("profile",       "Perfil"),
            ("started_at",    "Inicio"),
            ("ended_at",      "Termino"),
            ("duration_fmt",  "Duracao"),
            ("model_name",    "Modelo Utilizado"),
            ("total_items",   "Total de Itens"),
            ("status",        "Status"),
            ("notes",         "Observacoes"),
        ]:
            w.writerow([label, report_data.get(key, "")])
        w.writerow([])

        # Secao 2: totais por classe
        w.writerow(["=== TOTAIS POR CLASSE ==="])
        w.writerow(["Classe", "Quantidade"])
        for cls, qty in sorted(report_data.get("counts_by_class", {}).items()):
            w.writerow([cls, qty])
        w.writerow([])

        # Secao 3: erros/alertas
        errors = report_data.get("errors", [])
        if errors:
            w.writerow(["=== ERROS / ALERTAS DA SESSAO ==="])
            w.writerow(["Tipo", "Detalhe", "Timestamp"])
            for e in errors:
                w.writerow([e.get("event_type"), e.get("detail", ""), e.get("ts", "")])
            w.writerow([])

        # Secao 4: metricas tecnicas (somente technical)
        metrics = report_data.get("metrics")
        if metrics:
            w.writerow(["=== METRICAS TECNICAS DE INFERENCIA ==="])
            w.writerow(["Metrica", "Valor"])
            for key, label in [
                ("infer_count",    "Total de Inferencias"),
                ("infer_time_min", "Tempo Inf. Min (s)"),
                ("infer_time_avg", "Tempo Inf. Medio (s)"),
                ("infer_time_max", "Tempo Inf. Max (s)"),
                ("conf_min",       "Confianca Min"),
                ("conf_avg",       "Confianca Media"),
                ("conf_max",       "Confianca Max"),
                ("fps_avg",        "FPS Medio"),
                ("serial_port",    "Porta Serial"),
                ("model_path",     "Caminho do Modelo"),
            ]:
                w.writerow([label, metrics.get(key, "")])
            w.writerow([])

            snapshot = report_data.get("config_snapshot", {})
            if snapshot:
                w.writerow(["=== SNAPSHOT DE CONFIGURACAO DA SESSAO ==="])
                w.writerow(["Parametro", "Valor na Sessao"])
                for k, v in sorted(snapshot.items()):
                    w.writerow([k, v])
                w.writerow([])

    return path


# ──────────────────────────────────────────────────────────── PDF ─────────────

def export_pdf(report_data: dict, output_path: str | Path) -> Path:
    """
    Exporta relatorio em PDF usando fpdf2.

    Estrutura:
      Relatorio operacional:
        Cabecalho, dados da sessao, totais por classe,
        observacoes/erros (se houver)
      Relatorio tecnico (maintenance):
        Idem + metricas de inferencia + snapshot de configuracao

    Returns: Path do arquivo .pdf gerado.
    Raises:  ImportError se fpdf2 nao estiver instalado.
             RuntimeError em caso de erro de geracao.
    """
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ImportError(
            "fpdf2 nao esta instalado. Execute: pip install fpdf2"
        ) from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    is_technical = report_data.get("report_type") == "technical"
    title = "Relatorio Tecnico Ampliado" if is_technical else "Relatorio Operacional"

    pdf = _RecycleAIPDF(title=title, orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Dados da sessao ───────────────────────────────────────────────────────
    _section(pdf, "DADOS DA SESSAO")
    rows_sessao = [
        ("Operador",        report_data.get("operator", "-")),
        ("Perfil",          report_data.get("profile",  "-")),
        ("Inicio",          report_data.get("started_at", "-")),
        ("Termino",         report_data.get("ended_at",   "-") or "-"),
        ("Duracao",         report_data.get("duration_fmt", "-")),
        ("Modelo utilizado",report_data.get("model_name",   "-") or "-"),
        ("Total de itens",  str(report_data.get("total_items", 0))),
        ("Status",          report_data.get("status", "-")),
    ]
    if report_data.get("notes"):
        rows_sessao.append(("Observacoes", report_data["notes"]))
    _table(pdf, rows_sessao)

    # ── Totais por classe ─────────────────────────────────────────────────────
    counts = report_data.get("counts_by_class", {})
    if counts:
        _section(pdf, "TOTAIS POR CLASSE DE RESIDUO")
        _table(pdf, [(cls, str(qty)) for cls, qty in sorted(counts.items())])
    else:
        _section(pdf, "TOTAIS POR CLASSE DE RESIDUO")
        _info(pdf, "Nenhum item triado nesta sessao.")

    # ── Erros/alertas ─────────────────────────────────────────────────────────
    errors = report_data.get("errors", [])
    if errors:
        _section(pdf, "EVENTOS DA SESSAO (ERROS / ALERTAS)")
        _table(pdf, [
            (e.get("event_type", ""), f"{e.get('detail', '')} [{e.get('ts', '')}]")
            for e in errors
        ])

    # ── Metricas (somente technical) ──────────────────────────────────────────
    if is_technical:
        metrics = report_data.get("metrics")
        if metrics:
            _section(pdf, "METRICAS DE INFERENCIA")
            rows_metrics = [
                ("Total de inferencias",      str(metrics.get("infer_count", "-"))),
                ("Tempo inf. min (s)",         _fmt_float(metrics.get("infer_time_min"))),
                ("Tempo inf. medio (s)",       _fmt_float(metrics.get("infer_time_avg"))),
                ("Tempo inf. max (s)",         _fmt_float(metrics.get("infer_time_max"))),
                ("Confianca min",              _fmt_float(metrics.get("conf_min"))),
                ("Confianca media",            _fmt_float(metrics.get("conf_avg"))),
                ("Confianca max",              _fmt_float(metrics.get("conf_max"))),
                ("FPS medio",                 _fmt_float(metrics.get("fps_avg"))),
                ("Porta serial",              str(metrics.get("serial_port") or "-")),
                ("Caminho do modelo",         str(metrics.get("model_path")  or "-")),
            ]
            # Extras (conf_thres, iou_thres, etc.)
            extra = metrics.get("extra", {})
            if extra:
                for k, v in extra.items():
                    rows_metrics.append((f"  {k}", str(v)))
            _table(pdf, rows_metrics)
        else:
            _section(pdf, "METRICAS DE INFERENCIA")
            _info(pdf, "Metricas nao coletadas nesta sessao.")

        # ── Snapshot de configuracao ──────────────────────────────────────────
        snapshot = report_data.get("config_snapshot", {})
        if snapshot:
            _section(pdf, "SNAPSHOT DE CONFIGURACAO DA SESSAO")
            _table(pdf, [(k, str(v)) for k, v in sorted(snapshot.items())])

    # ── Rodape final ──────────────────────────────────────────────────────────
    pdf.set_y(-25)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(
        0, 6,
        f"Gerado em: {report_data.get('generated_at', '')}  |  "
        f"Sessao #{report_data.get('op_session_id', '')}  |  "
        f"RecycleAI-Station - UNIP TCC",
        align="C",
    )
    pdf.set_text_color(0, 0, 0)

    pdf.output(str(path))
    return path


# ──────────────────────────────────────────── Helpers internos de PDF ─────────

class _RecycleAIPDF:
    """
    Wrapper fino sobre FPDF que gera o PDF e expoe apenas o necessario.
    Nao herda de FPDF para evitar conflitos com o metodo header/footer.
    Usa composicao.
    """

    def __init__(self, title: str, **kwargs):
        from fpdf import FPDF
        self._pdf = FPDF(**kwargs)
        self._pdf.compress = False   # streams nao comprimidos → texto pesquisavel no binario
        self._title = title
        self._pdf.set_margins(15, 20, 15)
        self._pdf.set_author("RecycleAI-Station - UNIP TCC")
        self._pdf.set_title(title)
        self._pdf.set_creator("fpdf2")

    def add_page(self):
        self._pdf.add_page()
        self._draw_header()

    def set_auto_page_break(self, auto: bool, margin: float):
        self._pdf.set_auto_page_break(auto=auto, margin=margin)

    def set_font(self, *a, **kw):
        self._pdf.set_font(*a, **kw)

    def set_y(self, y: float):
        self._pdf.set_y(y)

    def set_text_color(self, r: int, g: int, b: int):
        self._pdf.set_text_color(r, g, b)

    def cell(self, *a, **kw):
        self._pdf.cell(*a, **kw)

    def output(self, path: str):
        self._pdf.output(path)

    def _draw_header(self):
        p = self._pdf
        # Barra azul de topo
        p.set_fill_color(21, 101, 192)   # #1565c0
        p.rect(0, 0, 210, 18, style="F")
        p.set_text_color(255, 255, 255)
        p.set_font("Helvetica", "B", 12)
        p.set_xy(15, 4)
        p.cell(100, 10, "RecycleAI-Station", ln=0)
        p.set_font("Helvetica", "", 9)
        p.set_xy(115, 4)
        p.cell(80, 10, self._title, align="R", ln=0)
        p.set_text_color(0, 0, 0)
        p.set_xy(15, 22)
        p.set_font("Helvetica", "", 8)
        p.set_text_color(100, 100, 100)
        p.cell(0, 5, "UNIP - Ciencia da Computacao - TCC", ln=1)
        p.set_text_color(0, 0, 0)
        p.ln(2)


def _section(pdf: _RecycleAIPDF, title: str) -> None:
    """Imprime titulo de secao com fundo cinza."""
    p = pdf._pdf
    p.ln(3)
    p.set_fill_color(224, 224, 224)  # #e0e0e0
    p.set_font("Helvetica", "B", 9)
    p.cell(0, 7, f"  {title}", ln=1, fill=True)
    p.ln(1)


def _table(pdf: _RecycleAIPDF, rows: list[tuple[str, str]]) -> None:
    """Imprime tabela de duas colunas (chave, valor)."""
    p = pdf._pdf
    p.set_font("Helvetica", "", 8)
    fill = False
    for key, val in rows:
        p.set_fill_color(245, 245, 245)
        p.cell(_COL_KEY, 6, f"  {key}", border=0, fill=fill)
        p.cell(_COL_VAL, 6, f"  {_safe_str(val)}", border=0, fill=fill, ln=1)
        fill = not fill
    p.ln(1)


def _info(pdf: _RecycleAIPDF, text: str) -> None:
    """Imprime linha informativa em italico."""
    p = pdf._pdf
    p.set_font("Helvetica", "I", 8)
    p.set_text_color(120, 120, 120)
    p.cell(0, 6, f"  {text}", ln=1)
    p.set_text_color(0, 0, 0)
    p.ln(1)


# ──────────────────────────────────────────────────────────── Helpers ─────────

def _fmt_duration(duration_s: float | None) -> str:
    if duration_s is None:
        return "-"
    total = int(duration_s)
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _fmt_float(v) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _safe_str(v) -> str:
    """Converte valor para string sem caracteres fora do Latin-1."""
    s = str(v) if v is not None else "-"
    # Substituir caracteres fora do Latin-1 para evitar erro do fpdf2 core fonts
    return s.encode("latin-1", errors="replace").decode("latin-1")
