"""
Recherche rapide avec filtres.
"""
from typing import Any, Dict, List, Optional

from .supabase_client import get_client


def search_documents(
    client_id: Optional[str] = None,
    min_total: Optional[float] = None,
    max_total: Optional[float] = None,
    status: Optional[str] = None,
    doc_type: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    sb = get_client()
    if not sb:
        return {"error": "Supabase non configuré"}

    results: Dict[str, List[Dict[str, Any]]] = {}

    if doc_type in (None, "quote"):
        q = sb.table("devis").select("id,client_id,total,status,created_at")
        if client_id:
            q = q.eq("client_id", client_id)
        if min_total is not None:
            q = q.gte("total", min_total)
        if max_total is not None:
            q = q.lte("total", max_total)
        if status:
            q = q.eq("status", status)
        results["devis"] = q.limit(limit).execute().data or []

    if doc_type in (None, "invoice"):
        f = sb.table("factures").select("id,client_id,total,created_at")
        if client_id:
            f = f.eq("client_id", client_id)
        if min_total is not None:
            f = f.gte("total", min_total)
        if max_total is not None:
            f = f.lte("total", max_total)
        results["factures"] = f.limit(limit).execute().data or []

    return results
