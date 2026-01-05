"""
Suggestions basées sur les données existantes : modèles de devis/factures, corrections auto.
"""
from typing import Any, Dict, List

from .supabase_client import get_client


def suggest_templates(limit: int = 5) -> List[Dict[str, Any]]:
    sb = get_client()
    if not sb:
        return []
    # On prend quelques devis/factures récents pour servir de modèles
    devis = sb.table("devis").select("id,metadata,total").order("created_at", desc=True).limit(limit).execute().data or []
    factures = sb.table("factures").select("id,metadata,total").order("created_at", desc=True).limit(limit).execute().data or []
    suggestions = []
    for d in devis:
        suggestions.append(
            {"type": "devis", "source_id": d["id"], "total": d.get("total"), "metadata": d.get("metadata")}
        )
    for f in factures:
        suggestions.append(
            {"type": "facture", "source_id": f["id"], "total": f.get("total"), "metadata": f.get("metadata")}
        )
    return suggestions


def auto_corrections() -> Dict[str, Any]:
    # Placeholders pour corrections automatiques simples
    return {
        "orthographe": "Correction orthographique des champs texte.",
        "champs_manquants": "Remplissage des champs obligatoires si manquants (siret, tva, penalites).",
        "mise_en_page": "Normalisation des champs adresse, formats de dates, totaux recalculés.",
    }
