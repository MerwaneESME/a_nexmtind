"""
Helpers Supabase pour pré-remplir (clients récents, matériaux suggérés).
"""
from typing import List

from .supabase_client import get_client


def get_recent_clients(prefix: str = "") -> List[dict]:
    sb = get_client()
    if not sb:
        return []
    q = sb.table("clients").select("id,name,address,contact").order("created_at", desc=True).limit(10)
    if prefix:
        q = q.ilike("name", f"%{prefix}%")
    res = q.execute()
    return res.data or []


def get_materials_suggestions(user_id: str | None = None) -> List[str]:
    sb = get_client()
    if not sb:
        return []
    devis_items = sb.table("devis_items").select("description, unit_price, qty").limit(20).execute().data or []
    # Retourne descriptions uniques
    seen = set()
    suggestions = []
    for d in devis_items:
        desc = d.get("description")
        if desc and desc not in seen:
            seen.add(desc)
            suggestions.append(desc)
    return suggestions[:10]
