from fastapi import APIRouter
from pydantic import BaseModel, Field, validator
from datetime import date
from typing import List, Optional

from .supabase_client import upsert_document
from .supabase_helpers import get_recent_clients, get_materials_suggestions


class PrepareItem(BaseModel):
    description: str
    qty: float = Field(gt=0)
    unit_price_ht: float = Field(gt=0)
    vat_rate: float = 20
    discount_rate: float = 0
    unit: Optional[str] = "unit"


class PrepareDevis(BaseModel):
    client_name: str
    client_address: Optional[str] = None
    client_contact: Optional[str] = None
    project_label: str
    items: List[PrepareItem]
    notes: Optional[str] = None
    user_id: Optional[str] = None

    @validator("items")
    def non_empty_items(cls, v):
        if not v:
            raise ValueError("Au moins une ligne est requise")
        return v


router = APIRouter()


@router.get("/prepare-devis/prefill")
async def prefill(user_id: str | None = None, client_prefix: str | None = None):
    return {
        "clients": get_recent_clients(client_prefix or ""),
        "materials": get_materials_suggestions(user_id),
    }


@router.post("/prepare-devis")
async def prepare_devis(payload: PrepareDevis):
    total_ht = sum(i.qty * i.unit_price_ht * (1 - i.discount_rate / 100) for i in payload.items)
    total_tva = sum(
        i.qty * i.unit_price_ht * (1 - i.discount_rate / 100) * (i.vat_rate / 100) for i in payload.items
    )
    totals = {"total_ht": total_ht, "total_tva": total_tva, "total_ttc": total_ht + total_tva}
    supabase_payload = {
        "doc_type": "quote",
        "number": f"DEV-{int(date.today().strftime('%y%m%d'))}",
        "date": date.today(),
        "customer": {
            "name": payload.client_name,
            "address": payload.client_address,
            "contact": payload.client_contact,
        },
        "supplier": {"name": "Auto", "address": "", "contact": ""},  # à enrichir si user connecté
        "line_items": [
            {
                "description": i.description,
                "quantity": i.qty,
                "unit": i.unit,
                "unit_price_ht": i.unit_price_ht,
                "vat_rate": i.vat_rate,
                "discount_rate": i.discount_rate,
            }
            for i in payload.items
        ],
        "notes": payload.notes,
        "user_id": payload.user_id,
        "raw_text": "",
    }
    supabase_result = upsert_document(supabase_payload)
    return {"data": supabase_payload, "totals": totals, "supabase": supabase_result}
