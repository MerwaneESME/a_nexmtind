"""API FastAPI V2 - Gestion correcte des donnÃ©es structurÃ©es."""
import json
import os
import re
import tempfile
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, validator

from api.chat import ChatInput as OptimizedChatInput
from api.chat import handle_chat_non_stream, router as optimized_chat_router
from .runtime import invoke_agent
from .config import get_fast_llm, get_llm
from .supabase_client import get_client, upsert_document
from .tools import calculate_totals_tool, clean_lines_tool, supabase_lookup_tool, validate_devis_tool
from .logging_config import logger
from .utils.pdf_generator import ChecklistPDFGenerator, extract_checklist_info_with_llm

# Cache par thread_id pour garder le dernier formulaire
SESSION_PAYLOADS: dict[str, dict] = {}

# Cache contexte projet (évite de refaire 8 appels Supabase par requête)
_PROJECT_CONTEXT_CACHE: dict[str, tuple[float, dict]] = {}
_PROJECT_CONTEXT_TTL = 45.0  # secondes


_EURO_AMOUNT_RE = re.compile(
    r"(?P<num>(?:\d{1,3}(?:[ \u202f.,]\d{3})+|\d+)(?:[.,]\d+)?)\s*(?:EUR|€)\b",
    re.IGNORECASE,
)


def format_eur(value: int | float) -> str:
    """Formate un montant en euros au format français (ex: 1 234,00 €)."""
    amount = float(value)
    raw = f"{amount:,.2f}"  # ex: 1,234.56
    raw = raw.replace(",", "X").replace(".", ",").replace("X", "\u202f")
    return f"{raw}\u00a0€"


def normalize_eur_in_text(text: str) -> str:
    """Normalise les montants 'XXX EUR' ou 'XXX€' en format français 'XXX,00 €'."""

    def _parse_num(num: str) -> float | None:
        cleaned = (num or "").replace("\u202f", "").replace(" ", "")
        if not cleaned:
            return None

        has_comma = "," in cleaned
        has_dot = "." in cleaned
        if has_comma and has_dot:
            dec_is_comma = cleaned.rfind(",") > cleaned.rfind(".")
            if dec_is_comma:
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif has_comma and not has_dot:
            cleaned = cleaned.replace(",", ".")

        try:
            return float(cleaned)
        except Exception:
            return None

    def _repl(match: re.Match[str]) -> str:
        raw_num = match.group("num")
        parsed = _parse_num(raw_num)
        if parsed is None:
            return match.group(0)
        return format_eur(parsed)

    return _EURO_AMOUNT_RE.sub(_repl, text or "")


def _get_project_context_cache(key: str) -> dict | None:
    entry = _PROJECT_CONTEXT_CACHE.get(key)
    if entry and (time.monotonic() - entry[0]) < _PROJECT_CONTEXT_TTL:
        return entry[1]
    return None


def _set_project_context_cache(key: str, ctx: dict) -> None:
    _PROJECT_CONTEXT_CACHE[key] = (time.monotonic(), ctx)


# ── Fast-path project-chat ─────────────────────────────────────────────────
_PROJECT_GREETING_RE = re.compile(
    r"^\s*(bonjour|salut|hello|hey|bonsoir|coucou|bonne\s*(journée|nuit|soirée))\b[!?.,\s]*$",
    re.IGNORECASE,
)
_PROJECT_THANKS_RE = re.compile(
    r"^\s*(merci|thanks|super|parfait|top|c'?est\s+bon|ok|d'?accord|très\s+bien|bonne?\s+idée)\b[!?.,\s]*$",
    re.IGNORECASE,
)
_PROJECT_DATA_KEYWORDS = re.compile(
    r"\b(devis|planning|tâche|tache|budget|coût|cout|avancement|membre|message|projet|"
    r"interven|phase|lot|document|facture|délai|delai|optimis|analys|conformité|marge|"
    r"rentabilité|risque|prochaine?s?\s+étape|étape)\b",
    re.IGNORECASE,
)


def _project_trivial_reply(message: str) -> str | None:
    """Réponse instantanée pour messages ne nécessitant pas le contexte projet."""
    msg = (message or "").strip()
    if not msg:
        return None
    if _PROJECT_GREETING_RE.match(msg):
        return "Bonjour ! Posez-moi vos questions sur votre projet : devis, planning, budget, avancement…"
    if _PROJECT_THANKS_RE.match(msg):
        return "Avec plaisir ! Avez-vous d'autres questions sur votre projet ?"
    return None


def _project_needs_context(message: str) -> bool:
    """Retourne True si le message nécessite le contexte projet (Supabase)."""
    return bool(_PROJECT_DATA_KEYWORDS.search(message or ""))


def _project_quick_actions(query: str, is_client: bool) -> list[dict]:
    """Génère des actions rapides contextuelles pour le chat projet."""
    q = (query or "").lower()
    actions: list[dict] = []

    if any(k in q for k in ("coût", "cout", "budget", "prix", "marge", "optimis", "rentab")):
        actions.append({"id": "analyze_costs", "label": "Analyser les marges", "prompt": "Analyse les marges et la rentabilité de chaque poste du devis.", "icon": "💰"})
        if not is_client:
            actions.append({"id": "optimize_costs", "label": "Optimiser les coûts", "prompt": "Quels postes peut-on réduire sans impacter la qualité ?", "icon": "✂️"})

    if any(k in q for k in ("planning", "délai", "delai", "étape", "etape", "tâche", "tache", "calendrier")):
        actions.append({"id": "propose_plan", "label": "Proposer un planning", "prompt": "Génère un planning détaillé pour ce projet.", "icon": "📅"})
        actions.append({"id": "next_steps", "label": "Prochaines étapes", "prompt": "Quelles sont les prochaines étapes prioritaires du projet ?", "icon": "➡️"})

    if any(k in q for k in ("membre", "equipe", "équipe", "rôle", "role", "permission")):
        actions.append({"id": "team_summary", "label": "Point sur l'équipe", "prompt": "Peux-tu me faire un point sur les rôles dans le projet avec les permissions ?", "icon": "👥"})

    if any(k in q for k in ("devis", "conformité", "mention", "tva", "vérif", "verif")):
        actions.append({"id": "check_compliance", "label": "Vérifier la conformité", "prompt": "Vérifie la conformité réglementaire du devis (TVA, mentions obligatoires, DTU).", "icon": "✅"})

    if any(k in q for k in ("risque", "attention", "problème", "probleme", "danger")):
        actions.append({"id": "identify_risks", "label": "Identifier les risques", "prompt": "Quels sont les risques principaux sur ce projet et comment les mitiger ?", "icon": "⚠️"})

    # Fallback selon rôle
    if not actions:
        if is_client:
            actions = [
                {"id": "explain_devis", "label": "Expliquer le devis", "prompt": "Explique-moi les postes principaux du devis en termes simples.", "icon": "📄"},
                {"id": "project_steps", "label": "Étapes du projet", "prompt": "Quelles sont les grandes étapes de mon projet ?", "icon": "📋"},
                {"id": "budget_summary", "label": "Résumé du budget", "prompt": "Quel est le budget total et quels sont les postes les plus importants ?", "icon": "💶"},
            ]
        else:
            actions = [
                {"id": "analyze_devis", "label": "Analyser le devis", "prompt": "Analyse le devis : rentabilité, conformité et points d'amélioration.", "icon": "📊"},
                {"id": "propose_plan", "label": "Proposer un planning", "prompt": "Génère un planning détaillé pour ce projet.", "icon": "📅"},
                {"id": "next_steps", "label": "Prochaines étapes", "prompt": "Quelles sont les prochaines étapes prioritaires du projet ?", "icon": "➡️"},
            ]

    return actions[:3]


ALLOWED_ORIGINS = os.getenv("AI_CORS_ALLOW_ORIGINS", "*")
origins = [origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()]
STATIC_DIR = Path(__file__).parent.parent / "static"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

app = FastAPI(title="Agent IA BTP V2 - Devis & Factures")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
if OUTPUT_DIR.exists():
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Optimized /chat endpoint (JSON + SSE streaming)
app.include_router(optimized_chat_router)


def save_upload(file: UploadFile) -> str:
    fd, path = tempfile.mkstemp(suffix=f"_{file.filename}")
    with os.fdopen(fd, "wb") as fh:
        fh.write(file.file.read())
    return path


def _extract_file_text(file: UploadFile) -> tuple[str, str]:
    """Extrait le texte d'un fichier uploadé (PDF, DOCX, PNG, JPG).
    Retourne (texte_extrait, nom_fichier).
    """
    filename = file.filename or "fichier"
    suffix = Path(filename).suffix.lower()
    path = save_upload(file)
    text = ""
    try:
        if suffix == ".pdf":
            import pypdf
            reader = pypdf.PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif suffix == ".docx":
            import docx
            document = docx.Document(path)
            text = "\n".join(p.text for p in document.paragraphs)
        elif suffix in (".png", ".jpg", ".jpeg"):
            try:
                import pytesseract
                from PIL import Image
                image = Image.open(path)
                text = pytesseract.image_to_string(image)
            except ImportError:
                text = "[OCR non disponible pour les images]"
        elif suffix in (".txt", ".md", ".csv"):
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        else:
            text = f"[Format non supporté : {suffix}]"
    except Exception as exc:
        logger.warning("_extract_file_text error: %s", exc)
        text = f"[Erreur lors de la lecture du fichier : {exc}]"
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    return text, filename


def _download_devis_pdf_text(sb, devis_row: dict) -> str | None:
    """Télécharge et extrait le texte du PDF d'un devis stocké dans Supabase Storage.
    Retourne None si aucun PDF ou si l'extraction échoue.
    """
    metadata = devis_row.get("metadata") or {}
    if not isinstance(metadata, dict):
        return None

    bucket = metadata.get("pdf_bucket")
    path = metadata.get("pdf_path")
    pdf_url = metadata.get("pdf_url")

    pdf_bytes: bytes | None = None

    # 1. Supabase Storage (prioritaire)
    if bucket and path:
        try:
            pdf_bytes = sb.storage.from_(bucket).download(path)
        except Exception as exc:
            logger.warning("_download_devis_pdf_text: storage download failed: %s", exc)

    # 2. URL publique en fallback
    if not pdf_bytes and pdf_url:
        try:
            import urllib.request
            req = urllib.request.Request(pdf_url, headers={"User-Agent": "NextMind/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                pdf_bytes = resp.read()
        except Exception as exc:
            logger.warning("_download_devis_pdf_text: URL download failed: %s", exc)

    if not pdf_bytes:
        return None

    try:
        import io
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()[:6000] if text.strip() else None
    except Exception as exc:
        logger.warning("_download_devis_pdf_text: pdf extract failed: %s", exc)
        return None


def _extract_total_from_pdf_text(text: str) -> float | None:
    """Tente d'extraire le montant total d'un PDF (recherche Total TTC / HT)."""
    if not text:
        return None
    # Patterns: "Total TTC : 12 345,67 €" ou "TOTAL 12345.67" etc.
    patterns = [
        r"[Tt]otal\s+[Tt][Tt][Cc]\s*[:\s]+(\d[\d\s]*[.,]\d{2})",
        r"[Tt]otal\s+[Hh][Tt]\s*[:\s]+(\d[\d\s]*[.,]\d{2})",
        r"TOTAL\s*[:\s]+(\d[\d\s]*[.,]\d{2})",
        r"[Mm]ontant\s+[Tt][Tt][Cc]\s*[:\s]+(\d[\d\s]*[.,]\d{2})",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            raw = match.group(1).replace(" ", "").replace(",", ".")
            try:
                return float(raw)
            except Exception:
                continue
    return None


def _structured_from_metadata(metadata: Dict[str, Any] | None) -> dict | None:
    """Reconstruit structured_payload depuis les mÃ©tadonnÃ©es du chat."""
    if not isinstance(metadata, dict):
        return None

    # Cas direct: payload dÃ©jÃ  structurÃ©
    if isinstance(metadata.get("structured_payload"), dict):
        structured = dict(metadata["structured_payload"])
        structured.setdefault("doc_type", structured.get("doc_type", "quote"))
        return structured

    # Cas: customer/supplier/line_items dÃ©jÃ  prÃ©sents
    if any(k in metadata for k in ["customer", "supplier", "line_items"]):
        structured = dict(metadata)
        structured.setdefault("doc_type", structured.get("doc_type", "quote"))
        return structured

    # âœ… Cas: champs plats du formulaire - TOUJOURS crÃ©er structured_payload mÃªme si vide
    customer = {
        "name": metadata.get("client_name") or metadata.get("customer_name") or "",
        "address": metadata.get("client_address") or "",
        "contact": metadata.get("client_contact") or "",
        "siret": metadata.get("client_siret"),
        "tva_number": metadata.get("client_tva"),
    }
    supplier = {
        "name": metadata.get("supplier_name") or "",
        "address": metadata.get("supplier_address") or "",
        "contact": metadata.get("supplier_contact") or "",
        "siret": metadata.get("supplier_siret"),
        "tva_number": metadata.get("supplier_tva"),
    }
    line_items = metadata.get("line_items") or metadata.get("items") or []
    project_label = metadata.get("project_label") or metadata.get("project") or metadata.get("project_name") or ""
    doc_type = metadata.get("doc_type") or metadata.get("docType") or "quote"
    notes = metadata.get("notes")

    # âœ… VÃ©rifier si on a des champs dans metadata (mÃªme vides)
    has_metadata_fields = any(k in metadata for k in [
        "client_name", "customer_name", "client_address", "client_contact",
        "supplier_name", "supplier_address", "project_label", "project",
        "line_items", "items", "notes"
    ])
    
    if not has_metadata_fields:
        return None  # Vraiment aucune donnÃ©e

    # âœ… TOUJOURS crÃ©er structured_payload si on a des champs metadata
    structured_payload = {
        "doc_type": doc_type,
        "customer": customer,  # Toujours inclure, mÃªme si vide
        "supplier": supplier,  # Toujours inclure, mÃªme si vide
        "line_items": line_items,
    }
    if project_label:
        structured_payload["project_label"] = project_label
    if notes:
        structured_payload["notes"] = notes
    
    return structured_payload


def _format_ai_reply(reply: Any) -> str:
    """Formate la rÃ©ponse AI pour le frontend."""
    if isinstance(reply, str):
        return reply
    
    if isinstance(reply, dict):
        if "reply" in reply:
            text = str(reply.get("reply") or "")
            todo = reply.get("todo") or []
            if isinstance(todo, list) and todo:
                bullets = [f"- {item}" for item in todo if item]
                if text:
                    bullets.insert(0, text)
                return "\n".join(bullets)
            return text

        doc = reply.get("document") or reply.get("data") or reply.get("payload") or {}
        corrections = reply.get("corrections") or reply.get("issues") or []
        missing_fields = reply.get("missing_fields") or reply.get("missing") or []

        parts: list[str] = []
        if isinstance(doc, dict) and doc:
            doc_type = doc.get("doc_type") or "document"
            totals = doc.get("totals") or {}
            total_ht = totals.get("total_ht")
            total_ttc = totals.get("total_ttc")
            if total_ht is not None or total_ttc is not None:
                parts.append(f"{doc_type.upper()} | HT: {total_ht} | TTC: {total_ttc}")

        if corrections:
            parts.append("Corrections :")
            for c in corrections[:6]:
                issue = c.get("issue") if isinstance(c, dict) else c
                field = c.get("field") if isinstance(c, dict) else None
                parts.append(f"- {field or ''}: {issue}")

        if missing_fields:
            parts.append("Champs manquants :")
            for mf in missing_fields[:6]:
                parts.append(f"- {mf}")

        if parts:
            return "\n".join(parts)

    try:
        import json as _json
        return _json.dumps(reply, ensure_ascii=False, indent=2)
    except Exception:
        return str(reply)


def _client_fast_reply(message: str) -> str | None:
    """Réponse rapide conseiller (client), sans accès Supabase."""
    system_prompt = (
        "Tu es un conseiller BTP pour un particulier. "
        "Réponds en français, en 2-3 phrases maximum, avec des mots simples. "
        "Si on te demande des étapes, donne 3 étapes courtes. "
        "Si on te demande d'expliquer un terme, donne une définition simple et un exemple concret. "
        "Ne pose pas de question."
    )
    try:
        llm = get_fast_llm()
        result = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=message)])
        return _format_ai_reply(getattr(result, "content", None) or str(result)).strip() or None
    except Exception as exc:
        logger.warning("Client fast reply failed: %s", exc)
        return None


def _pro_fast_reply(message: str) -> str | None:
    """Réponse rapide pour un professionnel (sans workflow complet)."""
    system_prompt = (
        "Tu es un assistant BTP pour un professionnel. "
        "Réponds directement à la question POSÉE, sans te présenter et sans lister toutes tes capacités. "
        "Donne une réponse concrète en français, structurée en 3 à 5 puces maximum. "
        "Si la question porte sur les matériaux, propose des familles de matériaux avec avantages/inconvénients simples. "
        "Ne parle pas du formulaire, ne parle pas de champs manquants, ne renvoie pas de JSON."
    )
    try:
        llm = get_fast_llm()
        result = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=message)])
        content = getattr(result, "content", None) or str(result)
        return str(content).strip() or None
    except Exception as exc:
        logger.warning("Pro fast reply failed: %s", exc)
        return None


# ==================== Devis terms UI (front-end cards) ====================

_DEVIS_TERMS_HINT_RE = re.compile(
    r"\b(termes?|mots?|jargon|lexique|glossaire|clarif|expliq|défin|defin|décortiq|decortiq|comprendr)\b",
    re.IGNORECASE,
)
_DEVIS_CONTEXT_RE = re.compile(r"\b(devis|facture)\b", re.IGNORECASE)
_TERM_DEFINITION_RE = re.compile(
    r"\b(c['’]est quoi|ça veut dire|ca veut dire|qu['’]est-ce que|definition|définition)\b",
    re.IGNORECASE,
)
_BTP_TERMS_LIKELY_RE = re.compile(
    r"\b(acompte|tva|décennale|decennale|rc\s*pro|dommages?-ouvrage|ipn|poutre|ragréage|ragreage|chape|étanchéité|etancheite|consuel|plomberie|gros\s*œuvre|gros\s*oeuvre|démolition|demolition)\b",
    re.IGNORECASE,
)


def _should_show_devis_terms_ui(message: str) -> bool:
    msg = (message or "").strip()
    if not msg:
        return False

    msg_lower = msg.lower()

    if _DEVIS_CONTEXT_RE.search(msg) and _DEVIS_TERMS_HINT_RE.search(msg):
        return True

    if _TERM_DEFINITION_RE.search(msg) and _BTP_TERMS_LIKELY_RE.search(msg):
        return True

    if "devis" in msg_lower and ("explique" in msg_lower or "clarifie" in msg_lower):
        return True

    return False


def _build_devis_terms_ui_reply(message: str) -> str:
    payload_query = (message or "").strip()
    lowered = payload_query.lower()
    if any(k in lowered for k in ("termes", "mots", "jargon", "lexique", "glossaire")):
        payload_query = ""

    payload = {"query": payload_query}
    payload_json = json.dumps(payload, ensure_ascii=False)

    return (
        "Je vous aide à comprendre les termes techniques qu’on voit souvent sur un devis BTP.\n\n"
        "```devis-terms\n"
        f"{payload_json}\n"
        "```\n\n"
        "Si vous avez votre devis sous les yeux, copiez/collez une ligne (ou joignez le PDF) "
        "et je vous l’explique ligne par ligne."
    )


# ==================== ModÃ¨les Pydantic ====================

class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatInput(BaseModel):
    message: str
    thread_id: Optional[str] = None
    history: Optional[List[ChatHistoryItem]] = None
    metadata: Optional[Dict[str, Any]] = None
    force_prepare: Optional[bool] = None


class ProjectChatInput(BaseModel):
    project_id: str
    user_id: str
    message: str
    history: Optional[List[ChatHistoryItem]] = None
    force_plan: Optional[bool] = None
    user_role: Optional[str] = None
    file_context: Optional[str] = None   # contenu extrait d'un fichier joint
    file_name: Optional[str] = None      # nom du fichier joint


class ProSearchInput(BaseModel):
    message: str
    city: Optional[str] = None
    postal_code: Optional[str] = None
    limit: int = Field(20, ge=1, le=50)


def _maybe_parse_json(text: str) -> Any:
    try:
        import json as _json
        return _json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return _json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def _strip_accents(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _normalize_search_text(value: str) -> str:
    if not value:
        return ""
    normalized = _strip_accents(value.lower())
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _load_allowed_tags(sb, limit: int = 200) -> list[str]:
    try:
        rows = sb.table("pro_tag_scores").select("tag").limit(limit).execute().data or []
    except Exception:
        return []
    tags = []
    seen = set()
    for row in rows:
        tag = (row.get("tag") or "").strip().lower()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def _parse_tag_payload(data: dict, allowed: set[str]) -> tuple[dict[str, float], str | None, str | None]:
    tag_weights: dict[str, float] = {}
    city = None
    postal_code = None

    if isinstance(data.get("city"), str) and data["city"].strip():
        city = data["city"].strip()
    if isinstance(data.get("postal_code"), str) and data["postal_code"].strip():
        postal_code = data["postal_code"].strip()

    raw_tags = data.get("tags")
    if isinstance(raw_tags, list):
        for item in raw_tags:
            if isinstance(item, dict):
                tag = str(item.get("tag") or "").strip().lower()
                weight = item.get("weight", 1)
            else:
                tag = str(item).strip().lower()
                weight = 1
            if not tag or tag not in allowed:
                continue
            try:
                weight_val = float(weight)
            except Exception:
                weight_val = 1.0
            weight_val = max(0.0, min(1.0, weight_val))
            if tag not in tag_weights or weight_val > tag_weights[tag]:
                tag_weights[tag] = weight_val

    return tag_weights, city, postal_code


def _extract_tags_with_llm(message: str, allowed_tags: list[str]) -> tuple[dict[str, float], str | None, str | None]:
    if not allowed_tags:
        return {}, None, None
    allowed_block = ", ".join(allowed_tags)
    system_prompt = (
        "You are a classifier for construction services. "
        "Return ONLY valid JSON with keys: tags, city, postal_code. "
        "tags is a list of objects with keys: tag, weight (0..1). "
        "Use only tags from the allowed list. No extra text."
    )
    user_prompt = f"Allowed tags: {allowed_block}\nUser request: {message}"
    try:
        llm = get_llm()
        result = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        content = getattr(result, "content", None) or str(result)
        parsed = _maybe_parse_json(content)
        if isinstance(parsed, dict):
            return _parse_tag_payload(parsed, set(allowed_tags))
    except Exception:
        pass
    return {}, None, None


def _fallback_tags(message: str, allowed_tags: list[str]) -> dict[str, float]:
    if not message or not allowed_tags:
        return {}
    norm_message = _normalize_search_text(message)
    tag_weights: dict[str, float] = {}
    for tag in allowed_tags:
        if not tag:
            continue
        if _normalize_search_text(tag) in norm_message:
            tag_weights[tag] = 1.0
    return tag_weights


def _extract_time_range(description: str | None) -> str | None:
    if not description:
        return None
    match = re.match(r"^\[\[time:([^\]]+)\]\]\s*(.*)$", description)
    if not match:
        return None
    return match.group(1).strip() or None


def _format_devis_title(metadata: dict | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    for key in ["project_label", "title", "label", "name"]:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _format_project_context(context: dict) -> str:
    project = context.get("project") or {}
    participants = context.get("participants") or []
    messages = context.get("messages") or []
    tasks = context.get("tasks") or []
    learning_stats = context.get("learning_stats") or []
    devis = context.get("devis") or []
    devis_items = context.get("devis_items") or []
    devis_pdf_texts = context.get("devis_pdf_texts") or []

    lines: list[str] = []

    # ── Projet ────────────────────────────────────────────────────────────
    if project:
        lines.append("=== PROJET ===")
        lines.append(f"Nom: {project.get('name') or 'Sans titre'}")
        if project.get("project_type"):
            lines.append(f"Type de travaux: {project.get('project_type')}")
        address = project.get("address")
        city = project.get("city")
        if address or city:
            lines.append(f"Lieu: {(address or '')} {(city or '')}".strip())
        if project.get("status"):
            lines.append(f"Statut: {project.get('status')}")
        if project.get("description"):
            lines.append(f"Description: {project.get('description')}")

    # ── Équipe ────────────────────────────────────────────────────────────
    if participants:
        lines.append("\n=== EQUIPE ===")
        for p in participants[:6]:
            profile = p.get("profiles") or {}
            name = profile.get("full_name") or profile.get("company_name") or p.get("invited_email") or "Inconnu"
            role = p.get("role") or "membre"
            status = p.get("status") or ""
            lines.append(f"- {name} | {role} | {status}")

    # ── Tâches ────────────────────────────────────────────────────────────
    if tasks:
        lines.append("\n=== TACHES ===")
        for task in tasks[:10]:
            label = task.get("name") or "Tache"
            status = task.get("status") or "en cours"
            period = ""
            if task.get("start_date") and task.get("end_date"):
                period = f" | du {task['start_date']} au {task['end_date']}"
            elif task.get("start_date"):
                period = f" | a partir du {task['start_date']}"
            time_range = _extract_time_range(task.get("description"))
            time_part = f" | horaires: {time_range}" if time_range else ""
            lines.append(f"- {label} | {status}{period}{time_part}")

    # ── Devis avec détail financier complet ───────────────────────────────
    if devis:
        lines.append("\n=== DEVIS ===")
        items_by_devis: dict[str, list] = {}
        for di in devis_items:
            did = di.get("devis_id") or ""
            items_by_devis.setdefault(did, []).append(di)

        # Index PDF texts by devis_id
        pdf_by_devis = {p["devis_id"]: p for p in devis_pdf_texts}

        for d in devis[:3]:
            title = _format_devis_title(d.get("metadata")) or "Devis"
            status = d.get("status") or "brouillon"
            total = d.get("total")
            total_label = format_eur(total) if isinstance(total, (int, float)) else "montant n/a"
            lines.append(f"\nDevis: {title} | Statut: {status} | Total: {total_label}")

            did = d.get("id") or ""
            postes = items_by_devis.get(did) or []
            if postes:
                lines.append("  Postes (description | qte | prix unitaire HT | total HT):")
                for poste in postes[:20]:
                    desc = poste.get("description") or "Poste"
                    qty = poste.get("qty")
                    unit_price = poste.get("unit_price")
                    total_poste = poste.get("total")
                    qty_label = f"qte {qty}" if qty is not None else ""
                    price_label = f"PU {format_eur(unit_price)}" if isinstance(unit_price, (int, float)) else ""
                    total_p_label = f"total {format_eur(total_poste)} HT" if isinstance(total_poste, (int, float)) else ""
                    detail = " | ".join(p for p in [qty_label, price_label, total_p_label] if p)
                    lines.append(f"  - {desc}" + (f" ({detail})" if detail else ""))
            elif did in pdf_by_devis:
                # Aucun poste en DB mais texte PDF extrait : l'inclure pour que l'IA puisse l'analyser
                pdf_entry = pdf_by_devis[did]
                pdf_total = pdf_entry.get("total")
                if isinstance(pdf_total, (int, float)):
                    lines.append(f"  Total extrait du PDF: {format_eur(pdf_total)}")
                lines.append(f"  Contenu du PDF (brut, max 4000 car.):")
                lines.append(pdf_entry["text"][:4000])

    # ── Messages récents du projet ─────────────────────────────────────────
    if messages:
        lines.append("\n=== MESSAGES RECENTS DU PROJET ===")
        for msg in messages[-6:]:
            content = (msg.get("message") or "").strip()
            date_str = (msg.get("created_at") or "")[:10]
            if content:
                lines.append(f"[{date_str}] {content[:300]}")

    # ── Références durées BTP ─────────────────────────────────────────────
    if learning_stats:
        lines.append("\n=== REFERENCES DUREES BTP (projets similaires) ===")
        for item in learning_stats[:6]:
            name = item.get("example_name") or item.get("normalized_label") or "Tache"
            duration = item.get("avg_duration_hours")
            count = item.get("sample_count") or 0
            duration_label = f"{duration:.1f}h" if isinstance(duration, (int, float)) else "duree n/a"
            lines.append(f"- {name} : {duration_label} en moyenne ({count} projets similaires)")

    return "\n".join(lines)

def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except Exception:
        return None


def _parse_time_range_start(value: str | None) -> tuple[int, int] | None:
    if not value or "-" not in value:
        return None
    try:
        start = value.split("-", 1)[0]
        hour_str, minute_str = start.split(":", 1)
        return int(hour_str), int(minute_str)
    except Exception:
        return None


def _apply_planning_guardrails(proposal: dict, now: datetime) -> dict:
    tasks = proposal.get("tasks")
    if not isinstance(tasks, list):
        return proposal
    today = now.date()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        start_date = _parse_date(task.get("start_date"))
        end_date = _parse_date(task.get("end_date")) or start_date
        if not start_date:
            continue
        shift_days = 0
        if start_date < today:
            shift_days = (today - start_date).days
        elif start_date == today:
            time_range = _parse_time_range_start(task.get("time_range"))
            if time_range:
                start_hour, start_minute = time_range
                if (start_hour, start_minute) <= (now.hour, now.minute):
                    shift_days = 1
        if shift_days > 0:
            start_date = start_date + timedelta(days=shift_days)
            if end_date:
                end_date = end_date + timedelta(days=shift_days)
            task["start_date"] = start_date.isoformat()
            task["end_date"] = (end_date or start_date).isoformat()
    proposal["tasks"] = tasks
    return proposal

def _build_project_context(sb, project_id: str, user_id: str) -> dict:
    cache_key = f"{project_id}:{user_id}"
    cached = _get_project_context_cache(cache_key)
    if cached is not None:
        return cached

    # Vérification d'accès (bloquante, nécessaire avant de continuer)
    membership = (
        sb.table("project_members")
        .select("id,role,status")
        .eq("project_id", project_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not membership.data:
        return {"error": "not_allowed"}

    # Appels Supabase indépendants en parallèle
    def _fetch_project():
        rows = (
            sb.table("projects")
            .select("id,name,description,project_type,city,address,status,created_at,updated_at")
            .eq("id", project_id)
            .limit(1)
            .execute()
        ).data or []
        return rows[0] if rows else None

    def _fetch_participants():
        return (
            sb.table("project_members")
            .select("user_id,role,status,invited_email,profiles:profiles!project_members_user_id_fkey(id,full_name,email,company_name)")
            .eq("project_id", project_id)
            .execute()
        ).data or []

    def _fetch_messages():
        return (
            sb.table("project_messages")
            .select("message,created_at,sender_id")
            .eq("project_id", project_id)
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        ).data or []

    def _fetch_tasks():
        return (
            sb.table("project_tasks")
            .select("name,status,start_date,end_date,description,completed_at")
            .eq("project_id", project_id)
            .execute()
        ).data or []

    def _fetch_devis():
        return (
            sb.table("devis")
            .select("id,status,total,metadata,created_at")
            .eq("project_id", project_id)
            .order("created_at", desc=True)
            .execute()
        ).data or []

    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(_fetch_project): "project",
            pool.submit(_fetch_participants): "participants",
            pool.submit(_fetch_messages): "messages",
            pool.submit(_fetch_tasks): "tasks",
            pool.submit(_fetch_devis): "devis",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception:
                results[key] = [] if key != "project" else None

    project = results.get("project")
    messages = results.get("messages") or []
    devis = results.get("devis") or []

    # learning_stats dépend de project_type → après le fetch parallèle
    learning_stats = []
    try:
        trade = project.get("project_type") if project else None
        stats_query = sb.table("task_learning_stats").select(
            "normalized_label,example_name,avg_duration_hours,avg_start_hour,avg_end_hour,sample_count,trade"
        )
        if trade:
            stats_query = stats_query.eq("trade", trade)
        learning_stats = stats_query.order("sample_count", desc=True).limit(6).execute().data or []
    except Exception:
        learning_stats = []

    # devis_items dépend des IDs devis → après le fetch parallèle
    devis_items = []
    if devis:
        devis_ids = [item["id"] for item in devis if item.get("id")]
        if devis_ids:
            try:
                devis_items = (
                    sb.table("devis_items")
                    .select("devis_id,description,qty,unit_price,total")
                    .in_("devis_id", devis_ids)
                    .execute()
                ).data or []
            except Exception:
                devis_items = []

    # Pour les devis sans postes (PDF uploadé sans extraction), tenter d'extraire le texte du PDF
    devis_pdf_texts: list[dict] = []
    if devis:
        devis_with_items = {item["devis_id"] for item in devis_items if item.get("devis_id")}
        for dv in devis:
            dv_id = dv.get("id") or ""
            if dv_id and dv_id not in devis_with_items:
                meta = dv.get("metadata") or {}
                if not isinstance(meta, dict):
                    continue
                # Seulement si le devis a un PDF en storage
                if meta.get("pdf_bucket") or meta.get("pdf_url"):
                    pdf_text = _download_devis_pdf_text(sb, dv)
                    if pdf_text:
                        title = _format_devis_title(meta) or "Devis"
                        total = dv.get("total")
                        # Si le total DB est null, essayer de l'extraire du PDF
                        if total is None:
                            extracted_total = _extract_total_from_pdf_text(pdf_text)
                            if extracted_total is not None:
                                total = extracted_total
                                try:
                                    sb.table("devis").update({"total": extracted_total}).eq("id", dv_id).execute()
                                    dv["total"] = extracted_total  # mise à jour locale
                                except Exception:
                                    pass
                        devis_pdf_texts.append({
                            "devis_id": dv_id,
                            "title": title,
                            "text": pdf_text,
                            "total": total,
                        })
                        logger.info("📄 devis PDF extracted: %s (%d chars)", title, len(pdf_text))

    ctx = {
        "project": project,
        "participants": results.get("participants") or [],
        "messages": list(reversed(messages)),
        "tasks": results.get("tasks") or [],
        "learning_stats": learning_stats,
        "devis": devis,
        "devis_items": devis_items,
        "devis_pdf_texts": devis_pdf_texts,
    }
    _set_project_context_cache(cache_key, ctx)
    return ctx


class PrepareItem(BaseModel):
    description: str = Field(..., min_length=3, max_length=500)
    quantity: float = Field(..., gt=0)
    unit_price_ht: float = Field(..., gt=0)
    vat_rate: float = Field(20, ge=0, le=100)
    discount_rate: float = Field(0, ge=0, le=100)
    unit: Optional[str] = Field(None, max_length=20)

    @validator("description")
    def description_not_empty(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("La description ne peut pas Ãªtre vide")
        return v.strip()


class PrepareDevisPayload(BaseModel):
    client_name: str = Field(..., min_length=2, max_length=200)
    client_address: Optional[str] = Field(None, max_length=500)
    client_contact: Optional[str] = Field(None, max_length=100)
    project_label: str = Field(..., min_length=3, max_length=300)
    items: List[PrepareItem] = Field(..., min_length=1, max_length=100)
    notes: Optional[str] = Field(None, max_length=2000)
    payment_terms: Optional[str] = Field(None, max_length=500)
    penalties_late_payment: Optional[str] = Field(None, max_length=500)
    professional_liability: Optional[str] = Field(None, max_length=500)
    dtu_references: Optional[List[str]] = Field(None, max_length=20)
    doc_type: Literal["quote", "invoice"] = "quote"
    thread_id: Optional[str] = None

    @property
    def line_items(self):
        return [
            {
                "description": i.description,
                "quantity": i.quantity,
                "unit": i.unit,
                "unit_price_ht": i.unit_price_ht,
                "vat_rate": i.vat_rate,
                "discount_rate": i.discount_rate,
            }
            for i in self.items
        ]


def _session_thread_id(thread_id: str, mode: str) -> str:
    """SÃ©pare les mÃ©moires LangGraph pour chat vs formulaire."""
    return f"{mode}:{thread_id}"


# ==================== Endpoints API ====================

@app.post("/chat-legacy")
async def chat_legacy(payload: ChatInput):
    """Chat conversationnel avec donnÃ©es du formulaire."""
    # ==================== Fast path: aide generique pour faire un devis ====================
    text_lower = (payload.message or "").lower()
    meta_dict = payload.metadata if isinstance(payload.metadata, dict) else {}

    # Cas: pro qui demande simplement de l'aide pour faire un devis, sans formulaire structure
    user_role = str(meta_dict.get("user_role") or "").lower()
    is_pro = user_role in {"professionnel", "pro"}

    if "devis" in text_lower and ("aide" in text_lower or "aider" in text_lower or "m'aider" in text_lower) and is_pro:
        # Liste des champs minimums a remplir pour demarrer un devis
        missing_fields = [
            "customer.name",
            "customer.address",
            "customer.contact",
            "supplier.name",
            "supplier.address",
            "supplier.contact",
            "line_items",
        ]

        reply_lines = [
            "Oui, je peux t'aider a preparer ton devis. Pour commencer, il faut remplir quelques informations :",
            "- Les coordonnees du client (nom, adresse, telephone ou email)",
            "- Les informations de ton entreprise (nom, adresse, contact)",
            "- Au moins une ligne de prestation avec description, quantite et prix HT",
            "",
            "Commence par renseigner ces champs dans le formulaire a gauche, puis je pourrai t'aider a verifier le devis et calculer les totaux.",
        ]

        reply_text = "\n".join(reply_lines)

        return JSONResponse(
            {
                "reply": reply_text,
                "raw_output": {
                    "mode": "fast_help_devis",
                    "missing_fields": missing_fields,
                },
                "corrections": [],
                "totals": {},
                "missing_fields": missing_fields,
            }
        )

    # Cas: question pro sans données structurées -> réponse rapide pro sans passer par LangGraph
    if is_pro:
        # On considère rapide si aucune metadata de formulaire n'est fournie
        has_form_metadata = any(k in meta_dict for k in ["customer_name", "client_name", "line_items", "items"])
        if not has_form_metadata:
            fast = _pro_fast_reply(payload.message)
            if fast:
                return JSONResponse(
                    {
                        "reply": fast,
                        "raw_output": {"mode": "fast_pro", "user_role": user_role},
                        "corrections": [],
                        "totals": {},
                        "missing_fields": [],
                    }
                )

    # ==================== Flux normal LangGraph ====================
    messages = []
    if payload.history:
        for item in payload.history:
            if item.role == "assistant":
                messages.append(SystemMessage(content=item.content))  # âš ï¸ Devrait Ãªtre AIMessage mais gardons pour compatibilitÃ©
            elif item.role == "system":
                messages.append(SystemMessage(content=item.content))
            else:
                messages.append(HumanMessage(content=item.content))
    messages.append(HumanMessage(content=payload.message))
    
    base_thread_id = payload.thread_id or "session"
    thread_id = _session_thread_id(base_thread_id, "chat")
    
    # âœ… Reconstruire structured_payload depuis metadata
    meta = payload.metadata if isinstance(payload.metadata, dict) else {}
    structured = _structured_from_metadata(payload.metadata) or SESSION_PAYLOADS.get(base_thread_id)
    
    logger.info("Metadata reÃ§u: %s", meta)
    logger.info("Structured reconstruit: %s", structured)
    
    # DÃ©tecter le mode (validate ou chat)
    meta_mode = str(meta.get("mode") or "").lower()
    is_validation_request = meta_mode in {"validate", "validation"} or bool(meta.get("validate_section"))
    
    if not is_validation_request:
        text_lower = payload.message.lower()
        is_validation_request = any(token in text_lower for token in ["valide", "validation", "corrige", "section"])
    
    intent = "validate" if is_validation_request else "chat"
    
    if structured:
        normalized = {
            "intent": intent,
            "doc_type": structured.get("doc_type", "quote"),
            "structured_payload": structured,
        }
        logger.info("Structured customer: %s", structured.get("customer"))
    else:
        normalized = {}
    
    # Heuristique: si message demande un devis/facture
    text_lower = payload.message.lower()
    wants_quote = "devis" in text_lower
    wants_invoice = "facture" in text_lower
    if not is_validation_request and (payload.force_prepare or (intent == "chat" and (wants_quote or wants_invoice))):
        intent = "prepare_devis"
        doc_type = "invoice" if wants_invoice else "quote"
        if not normalized:
            normalized = {"intent": intent, "doc_type": doc_type, "structured_payload": {"doc_type": doc_type}}
    
    state = {
        "messages": messages,
        "metadata": payload.metadata,
        "intent": intent,
        "normalized": normalized,
        "validate_section": meta.get("validate_section") if isinstance(meta, dict) else None,
    }
    
    result = invoke_agent(state, thread_id=thread_id)
    reply = result.get("output")
    if reply is None and result.get("messages"):
        last_msg = result["messages"][-1]
        reply = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    reply_text = _format_ai_reply(reply)
    corrections = result.get("corrections") or []
    totals = result.get("totals") or {}
    missing_fields = result.get("missing_fields") or []
    
    return JSONResponse({
        "reply": reply_text,
        "raw_output": reply,
        "corrections": corrections,
        "totals": totals,
        "missing_fields": missing_fields
    })


@app.post("/project-chat")
async def project_chat(payload: ProjectChatInput):
    """Chat IA par projet avec proposition de planning."""
    is_client = (payload.user_role or "").lower() in {"particulier", "client"}

    # ── Fast-path : pas besoin du contexte projet ──────────────────────────
    trivial = _project_trivial_reply(payload.message)
    if trivial:
        return JSONResponse({
            "reply": trivial,
            "proposal": None,
            "requires_devis": False,
            "quick_actions": _project_quick_actions("", is_client),
        })

    sb = get_client()
    if not sb:
        return JSONResponse({"reply": "Connexion Supabase indisponible.", "proposal": None, "requires_devis": True})

    context = _build_project_context(sb, payload.project_id, payload.user_id)
    if context.get("error") == "not_allowed":
        return JSONResponse({"reply": "Accès refusé au projet.", "proposal": None, "requires_devis": True}, status_code=403)

    devis_count = len(context.get("devis") or [])
    devis_items_count = len(context.get("devis_items") or [])
    devis_pdf_count = len(context.get("devis_pdf_texts") or [])
    # has_devis = vrai si on a des postes détaillés OU du texte PDF extrait
    has_devis = devis_count > 0 and (devis_items_count > 0 or devis_pdf_count > 0)

    now = datetime.now().astimezone()
    now_label = now.strftime("%Y-%m-%d %H:%M")
    tz_label = now.tzname() or "local"

    persona = (
        "Tu es un conseiller BTP bienveillant pour un particulier. "
        "Ton rôle est de l'aider à comprendre son projet, le devis, les étapes, et les termes techniques. "
        "Explique simplement avec des mots du quotidien, rassure, évite le jargon technique, "
        "et propose des options concrètes. Sois pédagogue et patient. "
        "Écris en français correct avec accents et ponctuation."
        if is_client
        else "Tu es un assistant BTP expert pour un professionnel. "
        "Ton rôle est de l'aider à optimiser son projet, analyser la rentabilité, vérifier la conformité, "
        "et proposer des améliorations. Sois précis, technique quand nécessaire, et orienté résultats. "
        "Écris en français correct avec accents et ponctuation."
    )

    client_guidance = ""
    pro_guidance = ""
    
    if is_client:
        client_guidance = """
 Guidance spécifique pour conseiller particulier:
 - Si on te demande d'expliquer le devis: Détaille les postes principaux, explique ce qui est inclus dans chaque ligne, et donne une vision d'ensemble du budget.
 - Si on te demande les étapes: Liste les phases principales du projet (ex: préparation, travaux, finitions) avec des exemples concrets.
 - Si on te demande le budget: Résume le total, explique les postes les plus importants, et mentionne les éventuels coûts supplémentaires à prévoir.
 - Si on te demande de clarifier des termes: Donne une définition simple avec un exemple concret du quotidien.
 - Si on te demande les délais: Explique la durée de chaque étape et les facteurs qui peuvent influencer les délais.
 - Si on te demande des points d'attention: Mentionne les précautions importantes, les autorisations nécessaires, et les risques à éviter.
 - Toujours utiliser des exemples concrets et des analogies pour faciliter la compréhension.
 - Format des montants: toujours en euros au format français (ex: 170,00 €).
 """
    else:
        pro_guidance = """
 Guidance spécifique pour assistant professionnel:
 - Si on te demande d'analyser le devis: Identifie les postes principaux, vérifie les cohérences,
   signale les écarts potentiels, et évalue la structure tarifaire.
 - Si on te demande de vérifier la conformité: Contrôle TVA, mentions obligatoires, références DTU,
   pénalités de retard, RC pro, et conformité réglementaire.
 - Si on te demande de calculer les marges: Analyse la rentabilité par poste, identifie les postes
   les plus rentables, et signale les postes à faible marge.
 - Si on te demande d'optimiser les coûts: Propose des alternatives de matériaux ou méthodes,
   identifie les postes surdimensionnés, et suggère des économies sans impacter la qualité.
 - Si on te demande les risques: Identifie les risques techniques, financiers, et réglementaires,
   et propose des mesures de mitigation.
 - Si on te demande des améliorations: Propose des alternatives techniques, des optimisations de process,
   ou des améliorations de rentabilité.
 - Toujours être précis avec les chiffres, les références réglementaires, et les calculs.
 - Utiliser le vocabulaire technique BTP quand c'est approprié.
 - Format des montants: toujours en euros au format français (ex: 170,00 €).
 """

    doc_rule = (
        f"- Un document a été fourni par l'utilisateur ({payload.file_name or 'fichier'}). "
        "Analyse-le en priorité pour répondre à la question. Cite les éléments pertinents du document dans ta réponse.\n"
        if payload.file_context
        else ""
    )

    system_prompt = f"""
{persona}
{client_guidance}
{pro_guidance}
Regles:
{doc_rule}- Le contexte fourni contient les VRAIES DONNEES du projet : postes du devis avec quantites et prix, taches, messages. Utilise ces chiffres reels dans tes reponses.
- Ne parle jamais d'un autre projet.
- Propose un planning uniquement si l'utilisateur le demande ou si force_plan est vrai.
- has_devis={has_devis}. Si has_devis=True, ne demande pas d'ajouter un devis. Si has_devis=False, tu peux demander un devis mais une seule fois.
- Reponds en JSON strict avec les cles: reply, proposal, requires_devis.
- Dans le champ "reply", ecris la reponse COMPLETE en markdown (titres ##, listes -, gras **). Ne tronque jamais ce champ.
- proposal est null ou {{ "summary": "...", "tasks": [ {{ "name": "", "description": "", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "time_range": "HH:MM-HH:MM" }} ] }}.
- Le planning ne doit pas commencer avant la date/heure actuelles.
- Si une tache est prevue aujourd'hui, son heure de debut doit etre apres l'heure actuelle, sinon decale au lendemain.
Style de reponse:
- Questions simples (statut, résumé) : 2-4 phrases, citer un chiffre clé du projet.
- Questions d'analyse (risques, devis, optimisation, conformité, planning, marges) : réponse LONGUE et STRUCTURÉE avec titres ##, listes à puces, et CHIFFRES RÉELS extraits du contexte (montants, quantités, pourcentages calculés). Minimum 150 mots.
- Toujours citer les données réelles du projet (noms de postes, montants, statuts de tâches).
- Reponds TOUJOURS en francais précis et professionnel.
- Ne renvoie jamais d'identifiants UUID ou de JSON brut dans reply.
- Si l'utilisateur demande un autre projet, propose d'ouvrir l'autre projet ou d'en creer un nouveau.
- Si la derniere reponse assistant est similaire a ce que tu allais dire, reformule en apportant une nouvelle information/action.
"""
    context_summary = _format_project_context(context)

    history_lines = []
    last_assistant = ""
    if payload.history:
        for item in payload.history[-6:]:
            history_lines.append(f"{item.role}: {item.content}")
        for item in reversed(payload.history):
            if item.role == "assistant":
                last_assistant = item.content
                break

    file_section = ""
    if payload.file_context:
        fn = payload.file_name or "fichier"
        file_section = f"\n\nDocument fourni par l'utilisateur ({fn}):\n---\n{payload.file_context}\n---\n"

    user_block = f"""
Contexte:
{context_summary}
{file_section}
Date/heure actuelles:
{now_label} ({tz_label})

Historique recent:
{chr(10).join(history_lines)}
Derniere reponse assistant:
{last_assistant}

Message utilisateur:
{payload.message}

force_plan: {bool(payload.force_plan)}
"""

    llm = get_fast_llm()
    result = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_block),
    ])
    content = getattr(result, "content", None) or str(result)
    parsed = _maybe_parse_json(content)

    if not isinstance(parsed, dict):
        parsed = {
            "reply": content if isinstance(content, str) else "Je peux aider a structurer le planning.",
            "proposal": None,
            "requires_devis": False,
        }

    parsed.setdefault("reply", "Je peux proposer un planning base sur le devis.")
    parsed.setdefault("proposal", None)
    parsed["requires_devis"] = False if has_devis else parsed.get("requires_devis", True)

    if isinstance(parsed.get("proposal"), dict):
        parsed["proposal"] = _apply_planning_guardrails(parsed["proposal"], now)

    # Anti-repetition et ajustement pour les particuliers
    reply_text = parsed.get("reply", "")
    
    # Correction : On ne déclenche le message "déjà lié" que si le message utilisateur semble vouloir lier un devis
    # et que la réponse de l'IA est trop générique ou répétitive à ce sujet.
    wants_to_link = any(k in payload.message.lower() for k in ("lier", "ajouter", "mettre", "joint", "televerser", "uploader"))
    is_quote_mention = "devis" in reply_text.lower()
    
    if has_devis and wants_to_link and is_quote_mention and ("ajout" in reply_text.lower() or "lier" in reply_text.lower()):
        reply_text = "Le devis est déjà lié au projet. Je peux analyser son contenu et proposer les prochaines étapes ou répondre à vos questions."
    
    if last_assistant and reply_text.strip().lower() == last_assistant.strip().lower():
        reply_text = reply_text + " Je peux aussi vous donner un résumé rapide du projet ou des prochaines étapes, dites-moi ce que vous préférez."

    if is_client and _should_show_devis_terms_ui(payload.message):
        reply_text = _build_devis_terms_ui_reply(payload.message)

    parsed["reply"] = normalize_eur_in_text(reply_text)
    parsed["quick_actions"] = _project_quick_actions(payload.message, is_client)

    return JSONResponse(parsed)


@app.post("/project-chat-file")
async def project_chat_file(
    project_id: str = Form(...),
    user_id: str = Form(...),
    message: str = Form(...),
    user_role: Optional[str] = Form(None),
    force_plan: Optional[bool] = Form(False),
    file: Optional[UploadFile] = File(None),
):
    """Chat IA projet avec fichier joint (PDF, DOCX, image).
    Accepte multipart/form-data. Le contenu du fichier est injecté dans le contexte.
    """
    file_context: str | None = None
    file_name: str | None = None

    if file and file.filename:
        file_text, file_name = _extract_file_text(file)
        if file_text.strip():
            file_context = file_text[:6000]  # limite pour ne pas exploser le contexte
        logger.info("📎 project-chat-file: fichier=%s, chars=%d", file_name, len(file_context or ""))

    # On construit un ProjectChatInput enrichi avec le contenu du fichier
    payload = ProjectChatInput(
        project_id=project_id,
        user_id=user_id,
        message=message,
        user_role=user_role,
        force_plan=force_plan,
        file_context=file_context,
        file_name=file_name,
    )
    return await project_chat(payload)


@app.post("/project-chat-client")
async def project_chat_client(payload: ProjectChatInput):
    """Chat IA dédié aux particuliers (conseiller)."""
    if _should_show_devis_terms_ui(payload.message):
        return JSONResponse(
            {
                "reply": _build_devis_terms_ui_reply(payload.message),
                "proposal": None,
                "requires_devis": False,
                "orchestrator": {"action": "answer", "needs_data": False, "reason": "devis_terms_ui"},
            }
        )

    # Fast answer sans données
    fast = _client_fast_reply(payload.message)
    keywords = ["projet", "planning", "tache", "tâche", "devis", "budget", "avancement", "membre", "message"]
    needs_data = any(k in (payload.message or "").lower() for k in keywords)

    if fast and not needs_data:
        return JSONResponse({
            "reply": fast,
            "proposal": None,
            "requires_devis": False,
            "orchestrator": {"action": "answer", "needs_data": False, "reason": "client_fast"},
        })

    # Sinon on délègue au flux projet avec ton conseiller (user_role=particulier)
    payload.user_role = "particulier"
    return await project_chat(payload)


class RefreshBudgetInput(BaseModel):
    project_id: str
    user_id: str


@app.post("/project-refresh-budget")
async def project_refresh_budget(payload: RefreshBudgetInput):
    """Extrait et met à jour le total des devis PDF liés au projet (sans items en DB).
    Appelé après avoir lié un devis au projet pour peupler le budget estimé.
    """
    sb = get_client()
    if not sb:
        return JSONResponse({"error": "no_db"}, status_code=500)

    # Invalider le cache pour forcer un re-fetch complet
    cache_key = f"{payload.project_id}:{payload.user_id}"
    _PROJECT_CONTEXT_CACHE.pop(cache_key, None)

    # Rebuild context (télécharge et extrait les PDFs, met à jour devis.total en DB)
    context = _build_project_context(sb, payload.project_id, payload.user_id)
    if context.get("error") == "not_allowed":
        return JSONResponse({"error": "not_allowed"}, status_code=403)

    devis = context.get("devis") or []
    return JSONResponse({
        "devis": [
            {
                "id": d.get("id"),
                "total": d.get("total"),
                "status": d.get("status"),
            }
            for d in devis
        ]
    })


@app.post("/pro-search")
async def pro_search(payload: ProSearchInput):
    """Recherche de professionnels a partir d'une demande libre."""
    sb = get_client()
    if not sb:
        return JSONResponse({"error": "supabase_not_configured", "results": []}, status_code=500)

    message = (payload.message or "").strip()
    if not message:
        return JSONResponse({"error": "empty_query", "results": []}, status_code=400)

    allowed_tags = _load_allowed_tags(sb)
    tag_weights, inferred_city, inferred_postal = _extract_tags_with_llm(message, allowed_tags)
    if not tag_weights:
        tag_weights = _fallback_tags(message, allowed_tags)

    city = payload.city.strip() if isinstance(payload.city, str) and payload.city.strip() else inferred_city
    postal_code = (
        payload.postal_code.strip()
        if isinstance(payload.postal_code, str) and payload.postal_code.strip()
        else inferred_postal
    )

    tags = list(tag_weights.keys())
    if not tags:
        query = sb.table("public_pro_profiles").select(
            "pro_id,display_name,company_name,city,postal_code,company_description,company_website,email,phone,address"
        )
        if city:
            query = query.ilike("city", f"%{city}%")
        if postal_code:
            query = query.ilike("postal_code", f"%{postal_code}%")
        fallback_rows = query.limit(payload.limit).execute().data or []
        return JSONResponse({
            "interpreted": {"tags": [], "city": city, "postal_code": postal_code},
            "results": fallback_rows,
        })

    score_rows = (
        sb.table("pro_tag_scores")
        .select("pro_id,tag,confidence,evidence_count")
        .eq("source", "computed")
        .in_("tag", tags)
        .execute()
    ).data or []

    score_by_pro: dict[str, float] = {}
    matched_tags: dict[str, list[str]] = {}
    for row in score_rows:
        pro_id = row.get("pro_id")
        tag = (row.get("tag") or "").strip().lower()
        if not pro_id or tag not in tag_weights:
            continue
        weight = tag_weights.get(tag, 0.0)
        try:
            confidence = float(row.get("confidence") or 0)
        except Exception:
            confidence = 0.0
        score_by_pro[pro_id] = score_by_pro.get(pro_id, 0.0) + (confidence * weight)
        matched_tags.setdefault(pro_id, [])
        if tag not in matched_tags[pro_id]:
            matched_tags[pro_id].append(tag)

    ranked = sorted(score_by_pro.items(), key=lambda item: item[1], reverse=True)
    ranked_ids = [item[0] for item in ranked]
    if not ranked_ids:
        return JSONResponse({
            "interpreted": {"tags": tags, "city": city, "postal_code": postal_code},
            "results": [],
        })

    profiles_rows = (
        sb.table("public_pro_profiles")
        .select("pro_id,display_name,company_name,city,postal_code,company_description,company_website,email,phone,address")
        .in_("pro_id", ranked_ids)
        .execute()
    ).data or []

    profiles_by_id = {row.get("pro_id"): row for row in profiles_rows}
    results: list[dict] = []
    for pro_id, score in ranked:
        profile = profiles_by_id.get(pro_id)
        if not profile:
            continue
        if city and city.lower() not in (profile.get("city") or "").lower():
            continue
        if postal_code and postal_code not in (profile.get("postal_code") or ""):
            continue
        enriched = dict(profile)
        enriched["score"] = round(score, 4)
        enriched["matched_tags"] = matched_tags.get(pro_id, [])
        results.append(enriched)
        if len(results) >= payload.limit:
            break

    return JSONResponse({
        "interpreted": {"tags": tags, "city": city, "postal_code": postal_code},
        "results": results,
    })


@app.post("/validate-section")
async def validate_section(payload: ChatInput):
    """Validation d'une section spÃ©cifique."""
    # Alias vers /chat avec mode validate
    payload.metadata = payload.metadata or {}
    payload.metadata["mode"] = "validate"

    optimized_payload = OptimizedChatInput(**payload.model_dump())
    result = await handle_chat_non_stream(optimized_payload)
    return JSONResponse(result)


@app.post("/analyze")
async def analyze(file: UploadFile, doc_type: str = "auto", thread_id: str | None = None):
    """Analyse d'un fichier uploadÃ©."""
    path = save_upload(file)
    initial_payload = {
        "doc_type": doc_type if doc_type != "auto" else "quote",
        "structured_payload": {"doc_type": doc_type if doc_type != "auto" else "quote"},
        "files": [path],
        "intent": "analyze",
    }
    state = {
        "intent": "analyze",
        "files": [path],
        "normalized": initial_payload,
        "messages": [HumanMessage(content=f"Analyse du fichier {file.filename}")],
    }
    result = invoke_agent(state, thread_id=thread_id or "analyze")
    output = result.get("output") or result
    totals = result.get("totals") or {}
    corrections = result.get("corrections") or []

    formatted = _format_ai_reply(output)
    return JSONResponse({
        "data": output,
        "formatted": formatted,
        "totals": totals,
        "corrections": corrections,
    })


@app.get("/prepare-devis/prefill")
async def prefill(user_id: str | None = None, client_prefix: str | None = None):
    """PrÃ©-remplissage depuis Supabase."""
    lookup = supabase_lookup_tool.invoke({"query": client_prefix, "mode": "prefill"})
    return {"supabase": lookup.get("results", {}), "error": lookup.get("error")}


@app.post("/prepare-devis")
async def prepare_devis(payload: PrepareDevisPayload):
    """PrÃ©paration complÃ¨te d'un devis."""
    base_thread_id = payload.thread_id or "session"
    thread_id = _session_thread_id(base_thread_id, "form")
    
    logger.info("Reception devis: client=%s, lignes=%s", payload.client_name, len(payload.items))
    
    # Construire structured_payload avec les VRAIES donnÃ©es
    structured_payload = {
        "doc_type": payload.doc_type,
        "number": f"AUTO-{date.today().strftime('%y%m%d')}",
        "date": str(date.today()),
        "customer": {
            "name": payload.client_name,
            "address": payload.client_address or "",
            "contact": payload.client_contact,
            "siret": None,
            "tva_number": None,
        },
        "supplier": {
            "name": "Auto",
            "address": "",
            "contact": "",
            "siret": None,
            "tva_number": None,
        },
        "line_items": payload.line_items,
        "notes": payload.notes,
        "payment_terms": payload.payment_terms,
        "penalties_late_payment": payload.penalties_late_payment,
        "professional_liability": payload.professional_liability,
        "dtu_references": payload.dtu_references,
    }
    
    logger.info("Structured payload client=%s", structured_payload["customer"])
    SESSION_PAYLOADS[base_thread_id] = structured_payload

    # Nettoyage et calculs
    cleaned = clean_lines_tool.invoke({"lines": structured_payload["line_items"]})
    structured_payload["line_items"] = cleaned.get("lines", structured_payload["line_items"])
    
    totals_calc = calculate_totals_tool.invoke({
        "lines": structured_payload["line_items"],
        "doc_type": payload.doc_type
    })
    totals = totals_calc.get("totals", {})

    # PrÃ©-remplissage depuis Supabase
    supabase_lookup = supabase_lookup_tool.invoke({
        "query": payload.client_name,
        "mode": "clients"
    })
    supabase_results = supabase_lookup.get("results", {})
    clients = supabase_results.get("clients", [])
    
    if clients:
        first_client = clients[0]
        if not structured_payload["customer"]["address"] and first_client.get("address"):
            structured_payload["customer"]["address"] = first_client["address"]
        if not structured_payload["customer"]["contact"] and first_client.get("contact"):
            structured_payload["customer"]["contact"] = first_client["contact"]
        structured_payload["customer"]["siret"] = first_client.get("siret")
        structured_payload["customer"]["tva_number"] = first_client.get("tva_number")

    # Construction du state pour LangGraph
    state = {
        "intent": "prepare_devis",
        "normalized": {
            "intent": "prepare_devis",
            "doc_type": payload.doc_type,
            "structured_payload": structured_payload,
            "summary": f"Devis pour {payload.client_name} - {payload.project_label}",
        },
        "messages": [
            HumanMessage(content=f"PrÃ©pare un {payload.doc_type} pour {payload.client_name} ({payload.project_label})")
        ],
        "totals": totals,
        "supabase_context": supabase_results,
    }

    result = invoke_agent(state, thread_id=thread_id)
    output = result.get("output") or result

    # Validation finale
    validation = validate_devis_tool.invoke({"payload": structured_payload})
    corrections = validation.get("issues", result.get("corrections") or [])
    totals = validation.get("totals", totals) or totals

    # Upsert Supabase
    supabase_result = upsert_document(structured_payload)

    formatted = _format_ai_reply(output)
    return JSONResponse({
        "data": output,
        "formatted": formatted,
        "totals": totals,
        "corrections": corrections,
        "supabase": supabase_result,
    })


class GenerateChecklistPdfPayload(BaseModel):
    project_name: str | None = None
    conversation_context: str
    query: str | None = None


@app.post("/generate-checklist-pdf")
async def generate_checklist_pdf(payload: GenerateChecklistPdfPayload):
    """Génère un PDF de checklist à partir du contexte (extraction LLM + design pro)."""
    response_text = (payload.conversation_context or "").strip()
    if not response_text:
        return JSONResponse({"error": "conversation_context manquant"}, status_code=400)

    extracted = extract_checklist_info_with_llm(response_text)

    project_name = (str(extracted.get("project_name") or "") or "").strip()
    if not project_name:
        project_name = (payload.project_name or "").strip() or "Diagnostic BTP"

    checkpoints = extracted.get("checkpoints") or []
    alerts = extracted.get("alerts") or []
    photos = extracted.get("photos") or []
    materials = extracted.get("materials") or []

    generator = ChecklistPDFGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = generator.generate_checklist_pdf(
        project_name=project_name,
        checkpoints=[str(x) for x in checkpoints],
        alerts=[str(x) for x in alerts],
        photos_needed=[str(x) for x in photos],
        materials=[str(x) for x in materials] if materials else None,
        output_path=OUTPUT_DIR / f"checklist_{timestamp}.pdf",
    )

    filename = f"NEXTMIND_Checklist_{project_name[:20]}.pdf"
    filename = re.sub(r"[^\\w\\s-]", "", filename).replace(" ", "_")

    return FileResponse(str(pdf_path), media_type="application/pdf", filename=filename)


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "version": "2.0"}
