"""Fast-path answer for simple chat questions.

Goals:
- < 1s perceived latency for simple questions (fast model)
- zero tools
- zero RAG
- minimal tokens

Key behavior:
- Gives elements of answer first (with assumptions if needed)
- Asks at most ONE follow-up question (only if necessary)
- Uses a short conversation context to avoid weird follow-up behavior
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .config import get_fast_llm
from .logging_config import logger
from .rag.retriever import is_corps_metier_question


class FastPathPayload(BaseModel):
    answer: str = Field(..., min_length=1)
    question: str | None = None


_GREETING_RE = re.compile(r"^\s*(bonjour|salut|hello|hey|coucou|bonsoir)\b", re.IGNORECASE)
_THANKS_RE = re.compile(r"\b(merci|thanks|super|parfait|top)\b", re.IGNORECASE)
_WHO_RE = re.compile(r"^\s*(t\s*qui|t'es\s*qui|tu\s*es\s*qui|qui\s*es[- ]tu)\b", re.IGNORECASE)
_REFERENCE_RE = re.compile(r"\b(projet(s)?\s+de\s+r[eé]f[eé]rence|r[eé]f[eé]rences?\s+projet|portfolio)\b", re.IGNORECASE)

# If the message explicitly asks for an action that benefits from tools/RAG,
# skip fast-path so the main pipeline can handle it.
_FULL_PIPELINE_HINT_RE = re.compile(
    r"\b("
    r"pdf|docx|fichier|piece\s+jointe|pi[eèé]ce\s+jointe|analyse|extraction|extraire|"
    r"supabase|historique|prefill|pre[- ]rempl|pr[eé][- ]rempl|base\s+de\s+donn[eé]es|bdd"
    r")\b",
    re.IGNORECASE,
)

_FULL_PIPELINE_TERMS_RE = re.compile(
    r"\b("
    r"probleme|panne|defaut|fuite|fissure|casse|ne\s+marche\s+pas|dysfonctionnement|"
    r"prix|cout|budget|devis|combien|tarif|taux\s+horaire|"
    r"temps|duree|delai|combien\s+de\s+temps|"
    r"comment|pourquoi|methode|procedure|"
    r"diagnostic|diagnostiquer|verifier|controler|checklist|preparer|"
    r"materiau|materiaux|quantite|liste|"
    r"refaire|renover|poser|installer|remplacer|reparer|"
    r"toiture|charpente|zinguerie|gouttiere|"
    r"plomberie|plombier|electricite|electricien|peinture|peintre|"
    r"carrelage|carreleur|faience|placo|plaquiste|isolation|vmc|chauffage|pac"
    r")\b",
    re.IGNORECASE,
)

_FAST_PATH_DEFINITION_RE = re.compile(
    r"^\s*("
    r"c['’]?est\s+quoi|ca\s+veut\s+dire|"
    r"definition\s+de|abreviation\s+de|sigle\s+de|"
    r"ca\s+signifie\s+quoi|"
    r"que\s+veut\s+dire"
    r")\b",
    re.IGNORECASE,
)

_FAST_PATH_META_RE = re.compile(r"\b(sens\s+de\s+la\s+vie|metaphys|philosoph)\b", re.IGNORECASE)


_FAST_PATH_SYSTEM_PROMPT = (
    "Tu es NEXTMIND, assistant BTP France, en mode réponse rapide.\n"
    "\n"
    "Réponds en 2-4 lignes maximum, style professionnel chantier.\n"
    "Si possible, donne AU MOINS 1 info concrète utile (ordre de grandeur prix, durée typique, matériau courant, ou point de vigilance).\n"
    "\n"
    "Évite les phrases vides type \"ça dépend\" sans préciser de quoi.\n"
    "Pas de questions dans ta réponse (remplace tout \"?\" par \".\").\n"
    "\n"
    "Si la question nécessite clairement plus de détail ou un diagnostic :\n"
    "termine par \"Pour un diagnostic complet avec prix détaillés, je peux t'aider avec plus d'infos.\". Réponds STRICTEMENT en JSON valide, sans texte autour:\n{\"answer\": \"...\", \"question\": null}\n"
)


def _heuristic_fast_reply(message: str) -> str | None:
    msg = (message or "").strip()
    if not msg:
        return "Je peux t'aider sur tes devis/factures BTP. Quelle est ta question ?"

    if _WHO_RE.search(msg):
        return "Je suis NEXTMIND, ton assistant IA BTP (devis, factures, travaux, matériaux, conformité)."

    if _GREETING_RE.search(msg):
        return "Bonjour ! Dis-moi ce dont tu as besoin (devis, facture, estimation, travaux, matériaux…)."

    if _THANKS_RE.search(msg) and len(msg) < 60:
        return "Avec plaisir. Tu veux que je t'aide sur quoi maintenant ?"

    if _REFERENCE_RE.search(msg):
        return (
            "Oui. Exemples de projets “références” BTP (résidentiel) :\n"
            "- Rénovation complète de salle de bain (plomberie + carrelage + ventilation)\n"
            "- Rénovation cuisine (réseaux + finitions)\n"
            "- Peinture + sols (remise en état)\n"
            "- Rénovation toiture/isolations\n"
            "- Extension/garage\n\n"
            "Tu cherches des références pour quel type de travaux ?"
        )

    return None


def _has_structured_metadata(metadata: dict[str, Any] | None) -> bool:
    if not isinstance(metadata, dict):
        return False
    if isinstance(metadata.get("structured_payload"), dict):
        return True
    return any(
        key in metadata
        for key in (
            "customer_name",
            "client_name",
            "supplier_name",
            "line_items",
            "items",
            "doc_type",
            "docType",
            "validate_section",
            "mode",
            "files",
        )
    )


def _should_use_full_pipeline(message: str, metadata: dict[str, Any] | None) -> bool:
    """Conservative gate: only route to full pipeline when tools/RAG are truly needed."""
    if _has_structured_metadata(metadata):
        return True
    msg = (message or "").strip()
    if not msg:
        return False
    # Existing hint regex (legacy) + robust normalized keyword matching (strict).
    if _FULL_PIPELINE_HINT_RE.search(msg):
        return True
    normalized = unicodedata.normalize("NFD", msg)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = re.sub(r"\s+", " ", normalized).lower().strip()
    return bool(_FULL_PIPELINE_TERMS_RE.search(normalized))


def _is_fast_path_candidate(message: str) -> bool:
    """Ultra-restrictive allow-list for fast-path."""
    msg = (message or "").strip()
    if not msg:
        return True
    if _GREETING_RE.search(msg) or _THANKS_RE.search(msg) or _WHO_RE.search(msg):
        return True
    normalized = unicodedata.normalize("NFD", msg)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = re.sub(r"\s+", " ", normalized).lower().strip()
    if _FAST_PATH_META_RE.search(normalized):
        return True
    if _FAST_PATH_DEFINITION_RE.search(normalized) and len(normalized) <= 90:
        return True
    return False


def _format_history(history: list[dict[str, str]] | None) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for item in history[-4:]:
        if not isinstance(item, dict):
            continue
        role = (item.get("role") or "").strip().lower()
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if len(content) > 240:
            content = content[:240] + "…"
        if role not in {"user", "assistant"}:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _maybe_parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
    except Exception:
        return None
    return None


def _normalize_followup_question(value: str | None) -> str | None:
    if not value:
        return None
    q = value.strip()
    if not q:
        return None
    # Keep only one question (first line / first '?')
    q = q.splitlines()[0].strip()
    if "?" in q:
        q = q.split("?", 1)[0].strip() + " ?"
    if not q.endswith("?"):
        q = q.rstrip(".") + " ?"
    # Compact whitespace
    q = re.sub(r"\s+", " ", q).strip()
    return q


async def try_fast_path(
    message: str,
    *,
    metadata: dict[str, Any] | None = None,
    user_role: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str | None:
    """Return a fast answer if the request is simple; otherwise return None."""
    heuristic = _heuristic_fast_reply(message)
    if heuristic:
        logger.info("⚡ fast-path route=fast (heuristic)")
        logger.info("⚡ fast-path heuristic hit")
        return heuristic

    # Let the pipeline use the dedicated RAG corpus for trade questions.
    if is_corps_metier_question(message):
        logger.info("⚡ fast-path route=full (reason=corps_metier)")
        return None

    if _should_use_full_pipeline(message, metadata):
        logger.info("⚡ fast-path route=full (reason=full_gate)")
        return None

    if not _is_fast_path_candidate(message):
        logger.info("⚡ fast-path route=full (reason=not_candidate)")
        return None

    msg = (message or "").strip()
    if not msg:
        return "Je peux t'aider sur tes devis/factures BTP. Quelle est ta question ?"

    history_text = _format_history(history)
    user_text = []
    if history_text:
        user_text.append("Contexte (derniers échanges):\n" + history_text)
    user_text.append(f"user_role={user_role or 'unknown'}")
    user_text.append("Message:\n" + msg)

    llm = get_fast_llm(temperature=0.2).bind(max_tokens=280)
    try:
        raw = await llm.ainvoke(
            [
                SystemMessage(content=_FAST_PATH_SYSTEM_PROMPT),
                HumanMessage(content="\n\n".join(user_text)),
            ]
        )
    except Exception as exc:
        logger.warning("fast-path answer failed: %s", exc)
        return None

    content = getattr(raw, "content", None)
    text = content if isinstance(content, str) else ""
    if not text:
        return None

    parsed = _maybe_parse_json(text)
    if isinstance(parsed, dict):
        try:
            payload = FastPathPayload.model_validate(parsed)
            answer = payload.answer.strip()
            question = _normalize_followup_question(payload.question)
            # Keep "answer" as a statement part: no question marks.
            answer = answer.replace("?", ".").strip()
            if question:
                return f"{answer}\n\n{question}"
            return answer
        except Exception:
            pass

    # Fallback: return the text, but avoid "question lists" as much as possible.
    # Keep only the first 6 lines and collapse multiple questions.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    compact = "\n".join(lines[:6]).strip()
    # If there are multiple questions, keep only the last one.
    if compact.count("?") > 1:
        parts = compact.split("?")
        kept_question = parts[-2].strip() if len(parts) >= 2 else ""
        statements = "?".join(parts[:-2]).replace("?", ".").strip()
        if statements and kept_question:
            compact = f"{statements}\n\n{kept_question} ?"
        else:
            compact = (statements or kept_question).strip()

    logger.info("⚡ fast-path LLM answered (role=%s)", (user_role or "unknown"))
    return compact
