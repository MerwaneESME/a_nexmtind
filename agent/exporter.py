"""
Export des documents en JSON (résumé) et placeholder DOCX/PDF.
"""
from typing import Any, Dict

from fastapi.responses import JSONResponse
from .supabase_client import get_client


def export_document(doc_id: str, doc_type: str = "quote") -> Dict[str, Any]:
    sb = get_client()
    if not sb:
        return {"error": "Supabase non configuré"}
    if doc_type == "invoice":
        doc = sb.table("factures").select("*").eq("id", doc_id).single().execute().data
        items = sb.table("facture_items").select("*").eq("facture_id", doc_id).execute().data
    else:
        doc = sb.table("devis").select("*").eq("id", doc_id).single().execute().data
        items = sb.table("devis_items").select("*").eq("devis_id", doc_id).execute().data
    if not doc:
        return {"error": "not_found"}

    summary = {
        "document": doc,
        "items": items or [],
        "ready_for_supabase": True,
    }
    # Placeholder: on ne génère pas réellement DOCX/PDF ici
    summary["export"] = {"docx": f"/output/{doc_id}.docx", "pdf": f"/output/{doc_id}.pdf"}
    return summary
