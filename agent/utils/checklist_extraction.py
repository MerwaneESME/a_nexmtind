from __future__ import annotations

import re
from typing import Sequence


_NUM_ITEM_RE = re.compile(r"^\s*(\d{1,2})[.)]\s+(.*\S)\s*$")
_BULLET_RE = re.compile(r"^\s*[-•]\s+(.*\S)\s*$")
_JSON_START_RE = re.compile(r"\{", re.DOTALL)


def extract_checkpoints_from_text(text: str, *, max_items: int = 8) -> list[str]:
    checkpoints: list[str] = []
    for line in _iter_lines(text):
        match = _NUM_ITEM_RE.match(line) or _BULLET_RE.match(line)
        if not match:
            continue
        item = match.group(match.lastindex or 1).strip()
        if _looks_like_alert(item) or _looks_like_photo(item):
            continue
        checkpoints.append(_clean_item(item))
        if len(checkpoints) >= max_items:
            break
    return checkpoints or ["Point de contrôle à définir"]


def extract_alerts_from_text(text: str, *, max_items: int = 4) -> list[str]:
    alerts: list[str] = []
    for line in _iter_lines(text):
        raw = line.strip()
        if not raw:
            continue
        if "⚠" in raw or _looks_like_alert(raw):
            alerts.append(_clean_item(raw.replace("⚠️", "").replace("⚠", "")))
        if len(alerts) >= max_items:
            break
    return alerts or ["Vérifier l'état général avant intervention"]


def extract_photos_from_text(text: str, *, max_items: int = 6) -> list[str]:
    photos: list[str] = []
    for line in _iter_lines(text):
        raw = line.strip()
        if not raw:
            continue
        if _looks_like_photo(raw):
            match = _NUM_ITEM_RE.match(raw) or _BULLET_RE.match(raw)
            item = (match.group(match.lastindex or 1) if match else raw).strip()
            photos.append(_clean_item(item))
        if len(photos) >= max_items:
            break
    return photos or ["Vue d'ensemble de la zone concernée", "Zoom sur les points problématiques"]


def extract_materials_from_text(
    text: str,
    *,
    material_keywords: Sequence[str] | None = None,
    max_items: int = 8,
) -> list[str]:
    if material_keywords is None:
        material_keywords = (
            "enduit",
            "peinture",
            "placo",
            "carrelage",
            "ciment",
            "mortier",
            "joint",
            "bande",
            "silicone",
            "résine",
            "resine",
        )
    materials: list[str] = []
    for line in _iter_lines(text):
        raw = line.strip()
        if not raw:
            continue
        if any(k in raw.lower() for k in material_keywords):
            match = _NUM_ITEM_RE.match(raw) or _BULLET_RE.match(raw)
            item = (match.group(match.lastindex or 1) if match else raw).strip()
            materials.append(_clean_item(item))
        if len(materials) >= max_items:
            break
    return materials


async def extract_with_llm(response_text: str) -> dict | None:
    """Extraction JSON via LLM (optionnel). Retourne None si indisponible."""
    try:
        from langchain_core.messages import SystemMessage

        from agent.config import get_fast_llm
    except Exception:
        return None

    prompt = f"""
Tu es un extracteur d'informations.
À partir de cette réponse d'assistant BTP, extrais UNIQUEMENT les informations demandées au format JSON.

Réponse à analyser :
{response_text}

Extrait :
1. "checkpoints" : liste des points de contrôle/vérification (max 8)
2. "alerts" : liste des signaux d'alerte/urgences (max 4)
3. "photos" : liste des photos à prendre (max 6)
4. "materials" : liste des matériaux mentionnés avec quantités si précisées (max 8)

Réponds UNIQUEMENT en JSON valide :
{{
  "checkpoints": ["..."],
  "alerts": ["..."],
  "photos": ["..."],
  "materials": ["..."]
}}
""".strip()

    llm = get_fast_llm(temperature=0)
    try:
        raw = await llm.ainvoke([SystemMessage(content=prompt)])
    except Exception:
        return None

    text = getattr(raw, "content", None)
    if not isinstance(text, str) or not text.strip():
        return None
    parsed = _maybe_parse_json(text)
    if not isinstance(parsed, dict):
        return None
    return parsed


def _iter_lines(text: str):
    for line in (text or "").splitlines():
        yield line.rstrip()


def _looks_like_alert(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ("alerte", "urgence", "dangere", "danger", "sécurité", "securite", "risque"))


def _looks_like_photo(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ("photo", "photograph", "vue d'ensemble", "zoom", "angle"))


def _clean_item(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = cleaned.lstrip("-•").strip()
    return cleaned


def _maybe_parse_json(text: str) -> dict | None:
    import json

    try:
        return json.loads(text)
    except Exception:
        pass
    match = _JSON_START_RE.search(text)
    if not match:
        return None
    start = match.start()
    end = text.rfind("}")
    if end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None
