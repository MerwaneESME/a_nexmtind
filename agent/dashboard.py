"""
Fonctions utilitaires pour le tableau de bord.
Retourne des agrégats simples à consommer par le front (derniers documents,
totaux par client/projet, statuts).
"""
from typing import Any, Dict, List

from .supabase_client import get_client


def list_recent(limit: int = 10) -> List[Dict[str, Any]]:
    sb = get_client()
    if not sb:
        return []
    devis = sb.table("devis").select("id,client_id,total,status,created_at").order("created_at", desc=True).limit(limit).execute().data
    factures = sb.table("factures").select("id,client_id,total,created_at").order("created_at", desc=True).limit(limit).execute().data
    # harmonise le champ status pour factures
    for f in factures:
        f["status"] = "facture"
    combined = (devis or []) + (factures or [])
    return sorted(combined, key=lambda x: x.get("created_at") or "", reverse=True)[:limit]


def totals_by_client(limit: int = 10) -> List[Dict[str, Any]]:
    sb = get_client()
    if not sb:
        return []
    res = (
        sb.rpc(
            "totals_by_client",
            # si la fonction SQL n'existe pas, on fallback plus bas
        )
        .execute()
        if hasattr(sb, "rpc")
        else None
    )
    if res and res.data:
        return res.data[:limit]
    # fallback simple si pas de RPC
    devis = sb.table("devis").select("client_id,total").execute().data or []
    factures = sb.table("factures").select("client_id,total").execute().data or []
    agg = {}
    for row in devis + factures:
        cid = row.get("client_id")
        if not cid:
            continue
        agg.setdefault(cid, 0)
        agg[cid] += float(row.get("total") or 0)
    return [{"client_id": cid, "total": total} for cid, total in list(agg.items())[:limit]]


def statuses(limit: int = 10) -> List[Dict[str, Any]]:
    sb = get_client()
    if not sb:
        return []
    devis = sb.table("devis").select("id,status,created_at").order("created_at", desc=True).limit(limit).execute().data
    factures = sb.table("factures").select("id,created_at").order("created_at", desc=True).limit(limit).execute().data
    for f in factures:
        f["status"] = "facture"
    return (devis or []) + (factures or [])
