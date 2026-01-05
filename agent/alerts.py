"""
Détection d'alertes simples : champs manquants (TVA, SIRET), incohérences ou retard.
"""
from datetime import datetime
from typing import Any, Dict, List

from .supabase_client import get_client


def detect_alerts(limit: int = 20) -> List[Dict[str, Any]]:
    sb = get_client()
    if not sb:
        return []

    alerts: List[Dict[str, Any]] = []

    devis = sb.table("devis").select("id,metadata,created_at").order("created_at", desc=True).limit(limit).execute().data or []
    factures = sb.table("factures").select("id,metadata,created_at").order("created_at", desc=True).limit(limit).execute().data or []

    for d in devis:
        meta = d.get("metadata") or {}
        cust = meta.get("customer") or {}
        missing = []
        if not cust.get("siret"):
            missing.append("siret client manquant")
        if not cust.get("tva_number"):
            missing.append("TVA client manquante")
        if missing:
            alerts.append(
                {"type": "incomplet", "document": "devis", "id": d.get("id"), "details": missing}
            )

    for f in factures:
        meta = f.get("metadata") or {}
        cust = meta.get("customer") or {}
        missing = []
        if not cust.get("siret"):
            missing.append("siret client manquant")
        if not cust.get("tva_number"):
            missing.append("TVA client manquante")
        # retard si due_date < aujourd'hui
        due = meta.get("due_date")
        if due:
            try:
                dt = datetime.fromisoformat(due)
                if dt.date() < datetime.utcnow().date():
                    alerts.append(
                        {
                            "type": "retard",
                            "document": "facture",
                            "id": f.get("id"),
                            "details": ["facture en retard (due_date passée)"],
                        }
                    )
            except Exception:
                pass
        if missing:
            alerts.append(
                {"type": "incomplet", "document": "facture", "id": f.get("id"), "details": missing}
            )

    return alerts
