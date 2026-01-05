from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, List, Literal, Optional

import docx
import pypdf
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .config import InvoiceSchema, QuoteSchema
from .supabase_client import get_client

TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(enabled_extensions=("html", "xml")),
)


def extract_pdf_text(path: str) -> tuple[str, int]:
    reader = pypdf.PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text, len(reader.pages)


def extract_docx_text(path: str) -> tuple[str, int]:
    document = docx.Document(path)
    text = "\n".join(p.text for p in document.paragraphs)
    return text, len(document.paragraphs)


def extract_image_text(path: str) -> tuple[str, int]:
    """OCR basique pour PNG/JPG si pytesseract + Tesseract sont installes."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return "", 0

    try:
        image = Image.open(path)
        return pytesseract.image_to_string(image), 1
    except Exception:
        return "", 0


def _normalize_decimal(value: Any, default: Decimal = Decimal(0)) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        return default


class ExtractInput(BaseModel):
    file_path: str
    doc_type: Literal["quote", "invoice", "auto"] = "auto"


@tool(args_schema=ExtractInput)
def extract_pdf_tool(file_path: str, doc_type: str = "auto"):
    """Extraction texte/structure depuis PDF/DOCX/PNG/JPG."""
    path = Path(file_path)
    if not path.exists():
        return {"error": "file_not_found", "file_path": file_path, "doc_type": doc_type}

    suffix = path.suffix.lower()
    text = ""
    page_count = 0
    if suffix == ".pdf":
        text, page_count = extract_pdf_text(file_path)
    elif suffix == ".docx":
        text, page_count = extract_docx_text(file_path)
    elif suffix in (".png", ".jpg", ".jpeg"):
        text, page_count = extract_image_text(file_path)
        if not text:
            return {"error": "ocr_unavailable", "file_path": file_path, "doc_type": doc_type}
    else:
        return {"error": "unsupported_format", "file_path": file_path, "doc_type": doc_type}

    detected = doc_type if doc_type != "auto" else "quote"
    return {
        "doc_type": detected,
        "parsed_text": text,
        "page_count": page_count,
        "file_path": str(path),
    }


class ParseInput(BaseModel):
    file_path: str
    doc_type: Literal["quote", "invoice", "auto"] = "auto"


@tool(args_schema=ParseInput)
def parse_document(file_path: str, doc_type: str = "auto"):
    """Alias retrocompatible vers extract_pdf_tool."""
    return extract_pdf_tool.func(file_path=file_path, doc_type=doc_type)


class CleanLinesInput(BaseModel):
    lines: List[dict] = Field(default_factory=list)
    default_vat_rate: float | None = 20.0


@tool(args_schema=CleanLinesInput)
def clean_lines_tool(lines: List[dict], default_vat_rate: float | None = 20.0):
    """Nettoie les lignes (quantites, unites, prix) et deduplication basique."""
    cleaned = []
    warnings = []
    seen_desc = set()

    for idx, line in enumerate(lines):
        desc = str(line.get("description") or "").strip()
        if not desc:
            warnings.append({"index": idx, "issue": "description_vide"})
            continue
        key = desc.lower()
        if key in seen_desc:
            warnings.append({"index": idx, "issue": "duplicate_description", "description": desc})
        seen_desc.add(key)

        qty = _normalize_decimal(line.get("quantity") or line.get("qty") or 0)
        price = _normalize_decimal(line.get("unit_price_ht") or line.get("unit_price") or 0)
        vat = _normalize_decimal(line.get("vat_rate", default_vat_rate if default_vat_rate is not None else 20))
        discount = _normalize_decimal(line.get("discount_rate") or 0)
        unit = (line.get("unit") or "").strip() or None

        if qty < 0:
            warnings.append({"index": idx, "issue": "negative_quantity", "quantity": float(qty)})
            qty = abs(qty)
        if price < 0:
            warnings.append({"index": idx, "issue": "negative_price", "unit_price": float(price)})
            price = abs(price)
        if vat < 0:
            warnings.append({"index": idx, "issue": "negative_vat_rate", "vat_rate": float(vat)})
            vat = abs(vat)

        cleaned.append(
            {
                "description": desc,
                "quantity": float(qty),
                "unit": unit,
                "unit_price_ht": float(price),
                "vat_rate": float(vat),
                "discount_rate": float(discount),
            }
        )

    return {"lines": cleaned, "warnings": warnings}


class CalculateTotalsInput(BaseModel):
    lines: List[dict] = Field(default_factory=list)
    default_vat_rate: float | None = 20.0
    doc_type: Literal["quote", "invoice"] | None = None


@tool(args_schema=CalculateTotalsInput)
def calculate_totals_tool(lines: List[dict], default_vat_rate: float | None = 20.0, doc_type: str | None = None):
    """Calcule HT/TVA/TTC et signale les incoherences numeriques."""
    total_ht = Decimal("0")
    total_tva = Decimal("0")
    issues = []
    normalized_lines = []

    for idx, line in enumerate(lines):
        qty = _normalize_decimal(line.get("quantity") or line.get("qty") or 0)
        price = _normalize_decimal(line.get("unit_price_ht") or line.get("unit_price") or 0)
        vat_rate = _normalize_decimal(line.get("vat_rate", default_vat_rate if default_vat_rate is not None else 20))
        discount = _normalize_decimal(line.get("discount_rate") or 0)

        line_total_ht = qty * price * (Decimal(1) - discount / Decimal(100))
        line_total_tva = line_total_ht * vat_rate / Decimal(100)

        if qty == 0 or price == 0:
            issues.append({"index": idx, "issue": "zero_value_line", "severity": "medium"})
        if vat_rate == 0:
            issues.append({"index": idx, "issue": "vat_missing_or_zero", "severity": "low"})

        total_ht += line_total_ht
        total_tva += line_total_tva

        normalized_lines.append(
            {
                "description": line.get("description"),
                "quantity": float(qty),
                "unit": line.get("unit"),
                "unit_price_ht": float(price),
                "vat_rate": float(vat_rate),
                "discount_rate": float(discount),
            }
        )

    totals = {
        "total_ht": float(total_ht),
        "total_tva": float(total_tva),
        "total_ttc": float(total_ht + total_tva),
    }
    return {"totals": totals, "issues": issues, "lines": normalized_lines, "doc_type": doc_type or "quote"}


def render_pdf_from_html(html: str, output_path: Path) -> tuple[Optional[Path], Optional[str]]:
    """Convertit HTML en PDF si weasyprint est disponible."""
    try:
        import weasyprint
    except Exception as exc:
        return None, f"WeasyPrint indisponible: {exc}"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weasyprint.HTML(string=html).write_pdf(output_path)
        return output_path, None
    except Exception as exc:
        return None, f"WeasyPrint a echoue: {exc}"


def render_pdf_playwright(html: str, output_path: Path) -> tuple[Optional[Path], Optional[str]]:
    """Fallback PDF via Playwright (Chromium). Necessite playwright + playwright install chromium."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        return None, f"Playwright indisponible: {exc}"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.set_content(html, wait_until="networkidle")
            page.pdf(path=str(output_path), format="A4", print_background=True)
            browser.close()
            return output_path, None
    except Exception as exc:
        return None, f"Playwright a echoue: {exc}"


@tool
def render_document(data: QuoteSchema | InvoiceSchema):
    """Rend un PDF depuis les donnees structurees."""
    context = data.model_dump()
    totals = data.totals()
    context.update(
        {
            "total_ht": float(totals["total_ht"]),
            "total_tva": float(totals["total_tva"]),
            "total_ttc": float(totals["total_ttc"]),
        }
    )
    context["totals"] = {
        "total_ht": float(totals["total_ht"]),
        "total_tva": float(totals["total_tva"]),
        "total_ttc": float(totals["total_ttc"]),
    }
    if context.get("dtu_references") is None:
        context["dtu_references"] = []

    doc_type = context.get("doc_type", "quote")
    template_name = "quote.docx.j2" if doc_type == "quote" else "invoice.docx.j2"

    OUTPUT_DIR.mkdir(exist_ok=True)
    html_template = env.get_template(template_name)
    html_content = html_template.render(**context)
    html_path = OUTPUT_DIR / f"{doc_type}_rendered.html"
    html_path.write_text(html_content, encoding="utf-8")

    pdf_path = OUTPUT_DIR / f"{doc_type}_rendered.pdf"
    pdf_result, pdf_err = render_pdf_from_html(html_content, pdf_path)

    if not pdf_result:
        pdf_result, pdf_err = render_pdf_playwright(html_content, pdf_path)

    status = "rendered" if pdf_result else "html_only"
    error = None if pdf_result else pdf_err

    return {
        "status": status,
        "error": error,
        "pdf_path": str(pdf_result) if pdf_result else None,
        "html_path": str(html_path),
        "artifacts": [str(p) for p in ([html_path] + ([pdf_result] if pdf_result else []))],
        "data": context,
    }


class ValidateInput(BaseModel):
    payload: dict


@tool(args_schema=ValidateInput)
def validate_devis_tool(payload: dict):
    """Controle TVA, mentions obligatoires et coherence des totaux."""
    doc_type = payload.get("doc_type") or "quote"
    try:
        document = InvoiceSchema(**payload) if doc_type == "invoice" else QuoteSchema(**payload)
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)], "issues": [], "totals": {}}

    totals = document.totals()
    issues = []

    for key, computed_value in totals.items():
        declared = payload.get(key)
        if declared is not None:
            diff = abs(_normalize_decimal(declared) - computed_value)
            if diff > Decimal("0.01"):
                issues.append(
                    {
                        "field": key,
                        "issue": f"ecart_total_{key}",
                        "details": {"declared": float(_normalize_decimal(declared)), "computed": float(computed_value)},
                        "severity": "medium",
                    }
                )

    if not document.payment_terms:
        issues.append({"field": "payment_terms", "issue": "conditions_paiement_manquantes", "severity": "high"})
    if doc_type == "invoice":
        if not document.penalties_late_payment:
            issues.append({"field": "penalties_late_payment", "issue": "penalites_retard_manquantes", "severity": "high"})
        if not document.professional_liability:
            issues.append({"field": "professional_liability", "issue": "mention_rc_pro_manquante", "severity": "medium"})
        if not document.due_date:
            issues.append({"field": "due_date", "issue": "date_echeance_manquante", "severity": "high"})

    for idx, line in enumerate(document.line_items):
        if line.vat_rate < 0:
            issues.append({"index": idx, "field": "vat_rate", "issue": "tva_negative", "severity": "medium"})
        if line.vat_rate == 0:
            issues.append({"index": idx, "field": "vat_rate", "issue": "tva_absente", "severity": "low"})
        if line.quantity == 0 or line.unit_price_ht == 0:
            issues.append({"index": idx, "field": "line_items", "issue": "ligne_zero", "severity": "medium"})

    suggested = []
    if issues:
        suggested.append("Verifier TVA, penalites et mentions obligatoires.")
        suggested.append("Recalculer les totaux HT/TVA/TTC a partir des lignes nettoyees.")

    return {
        "valid": len([i for i in issues if i.get("severity") == "high"]) == 0,
        "issues": issues,
        "totals": {k: float(v) for k, v in totals.items()},
        "suggested_corrections": suggested,
    }


@tool(args_schema=ValidateInput)
def validate_and_inconsistencies(payload: dict):
    """Alias retrocompatible vers validate_devis_tool."""
    return validate_devis_tool.func(payload=payload)


class SupabaseLookupInput(BaseModel):
    query: Optional[str] = None
    mode: Literal["clients", "materials", "history", "prefill", "auto"] = "auto"
    limit: int = 10


@tool(args_schema=SupabaseLookupInput)
def supabase_lookup_tool(query: Optional[str] = None, mode: str = "auto", limit: int = 10):
    """Recherche clients/materiaux/historiques dans Supabase."""
    sb = get_client()
    if not sb:
        return {"results": {}, "error": "supabase_not_configured"}

    results: dict[str, Any] = {}
    try:
        if mode in ("clients", "auto", "prefill"):
            q = sb.table("clients").select("id,name,address,contact").order("created_at", desc=True).limit(limit)
            if query:
                q = q.ilike("name", f"%{query}%")
            results["clients"] = q.execute().data or []

        if mode in ("materials", "auto", "prefill"):
            q_mat = sb.table("devis_items").select("description,unit_price,qty").limit(limit)
            mats = q_mat.execute().data or []
            dedup = []
            seen = set()
            for item in mats:
                desc = item.get("description")
                if desc and desc not in seen:
                    seen.add(desc)
                    dedup.append({"description": desc, "unit_price": item.get("unit_price"), "qty": item.get("qty")})
            results["materials"] = dedup

        if mode in ("history", "auto", "prefill"):
            q_hist = sb.table("devis").select("id,client_id,total,status,metadata").order("created_at", desc=True).limit(limit)
            results["history"] = q_hist.execute().data or []
    except Exception as exc:
        return {"results": results, "error": str(exc)}

    return {"results": results}
