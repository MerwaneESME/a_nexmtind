"""Client Supabase pour persistance des données."""
import os
from typing import Any, Optional

from fastapi.encoders import jsonable_encoder
from supabase import Client, create_client

from .config import InvoiceSchema, QuoteSchema


def get_client() -> Optional[Client]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def _get_or_create_client(sb: Client, name: str, address: Optional[str], contact: Optional[dict]):
    try:
        existing = sb.table("clients").select("id").eq("name", name).limit(1).execute()
        if existing.data:
            return existing.data[0]["id"]
        safe_contact: Any = {}
        if contact:
            if isinstance(contact, dict):
                safe_contact = contact
            elif isinstance(contact, str):
                safe_contact = {"value": contact}
        inserted = (
            sb.table("clients")
            .insert(
                {
                    "name": name,
                    "address": address,
                    "contact": safe_contact,
                }
            )
            .execute()
        )
        return inserted.data[0]["id"]
    except Exception:
        return None


def insert_quote(sb: Client, data: dict):
    quote = QuoteSchema(**data)
    totals = quote.totals()
    client_id = _get_or_create_client(
        sb, quote.customer.name, quote.customer.address, getattr(quote.customer, "contact", None)
    )
    payload = {
        "user_id": data.get("user_id"),
        "client_id": client_id,
        "status": data.get("status", "draft"),
        "raw_text": data.get("raw_text"),
        "metadata": jsonable_encoder(data),
        "total": float(totals["total_ttc"]),
    }
    devis = sb.table("devis").insert(payload).execute()
    devis_id = devis.data[0]["id"]

    items = []
    for line in quote.line_items:
        items.append(
            {
                "devis_id": devis_id,
                "product_id": None,
                "description": line.description,
                "qty": float(line.quantity),
                "unit_price": float(line.unit_price_ht),
                "total": float(line.total_ht),
            }
        )
    if items:
        sb.table("devis_items").insert(items).execute()
    return {"devis_id": devis_id, "client_id": client_id, "items_inserted": len(items)}


def insert_invoice(sb: Client, data: dict):
    invoice = InvoiceSchema(**data)
    totals = invoice.totals()
    client_id = _get_or_create_client(
        sb, invoice.customer.name, invoice.customer.address, getattr(invoice.customer, "contact", None)
    )
    payload = {
        "user_id": data.get("user_id"),
        "client_id": client_id,
        "raw_text": data.get("raw_text"),
        "metadata": jsonable_encoder(data),
        "total": float(totals["total_ttc"]),
        "devis_id": data.get("devis_id"),
    }
    fact = sb.table("factures").insert(payload).execute()
    facture_id = fact.data[0]["id"]

    items = []
    for line in invoice.line_items:
        qty_val = int(line.quantity) if line.quantity is not None else 0
        items.append(
            {
                "facture_id": facture_id,
                "product_id": None,
                "description": line.description,
                "qty": qty_val,
                "unit_price": float(line.unit_price_ht),
                "total": float(line.total_ht),
            }
        )
    if items:
        sb.table("facture_items").insert(items).execute()
    return {"facture_id": facture_id, "client_id": client_id, "items_inserted": len(items)}


def upsert_document(data: dict):
    """Insère ou met à jour un devis/facture dans Supabase."""
    sb = get_client()
    if not sb:
        return {"error": "Supabase client non configuré"}
    try:
        if data.get("doc_type") == "invoice":
            return insert_invoice(sb, data)
        else:
            return insert_quote(sb, data)
    except Exception as exc:
        return {"error": str(exc)}
