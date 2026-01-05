import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .runtime import invoke_agent
from .supabase_client import upsert_document
from .tools import calculate_totals_tool, clean_lines_tool, supabase_lookup_tool, validate_devis_tool

ALLOWED_ORIGINS = os.getenv("AI_CORS_ALLOW_ORIGINS", "*")
origins = [origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()]
STATIC_DIR = Path(__file__).parent.parent / "static"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

app = FastAPI(title="Agent IA BTP - Devis & Factures (LangGraph)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
if OUTPUT_DIR.exists():
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


def save_upload(file: UploadFile) -> str:
    fd, path = tempfile.mkstemp(suffix=f"_{file.filename}")
    with os.fdopen(fd, "wb") as fh:
        fh.write(file.file.read())
    return path


# Formatage de la réponse AI avec puces ASCII pour éviter les problèmes d'encodage
def _format_ai_reply(reply: Any) -> str:
    """Rend une version lisible (puces ASCII) pour le front."""
    if isinstance(reply, str):
        return reply
    try:
        import json as _json
    except Exception:
        _json = None

    if isinstance(reply, dict):
        if "reply" in reply:
            text = str(reply.get("reply") or "")
            todo = reply.get("todo") or []
            if isinstance(todo, list) and todo:
                bullets = [f"- {item}" for item in todo if item]
                if text:
                    bullets.insert(0, text)
                return "\n".join(bullets)
            return text

        doc = reply.get("document") or reply.get("data") or reply.get("payload") or {}
        corrections = reply.get("corrections") or reply.get("issues") or []
        next_steps = reply.get("next_steps") or reply.get("suggestions") or []

        parts: list[str] = []
        if isinstance(doc, dict):
            doc_type = doc.get("doc_type") or "document"
            totals = doc.get("totals") or {}
            total_ht = totals.get("total_ht")
            total_ttc = totals.get("total_ttc")
            if total_ht is not None or total_ttc is not None:
                parts.append(f"{doc_type.upper()} | HT: {total_ht} | TTC: {total_ttc}")

            line_items = doc.get("line_items")
            if isinstance(line_items, list) and line_items:
                parts.append("Lignes :")
                for li in line_items[:6]:
                    desc = li.get("description")
                    qty = li.get("quantity") or li.get("qty")
                    up = li.get("unit_price_ht") or li.get("unit_price")
                    parts.append(f"- {desc} | {qty} x {up}")
                if len(line_items) > 6:
                    parts.append(f"- (+{len(line_items) - 6} lignes)")

        if corrections:
            parts.append("Corrections :")
            for c in corrections[:6]:
                issue = c.get("issue") if isinstance(c, dict) else c
                field = c.get("field") if isinstance(c, dict) else None
                sev = c.get("severity") if isinstance(c, dict) else None
                msg = f"- {field or ''}: {issue}"
                if sev:
                    msg += f" ({sev})"
                parts.append(msg)
            if len(corrections) > 6:
                parts.append(f"- (+{len(corrections) - 6} autres)")

        if next_steps:
            parts.append("A suivre :")
            for s in next_steps[:5]:
                parts.append(f"- {s}")

        missing_fields = reply.get("missing_fields") or reply.get("missing") or []
        if missing_fields:
            parts.append("Champs manquants :")
            for mf in missing_fields[:6]:
                parts.append(f"- {mf}")

        if parts:
            return "\n".join(parts)

    if _json:
        try:
            return _json.dumps(reply, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return str(reply)


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatInput(BaseModel):
    message: str
    thread_id: Optional[str] = None
    history: Optional[List[ChatHistoryItem]] = None
    metadata: Optional[Dict[str, Any]] = None
    force_prepare: Optional[bool] = None


@app.post("/chat")
async def chat(payload: ChatInput):
    messages = []
    if payload.history:
        for item in payload.history:
            if item.role == "assistant":
                messages.append(SystemMessage(content=item.content))
            elif item.role == "system":
                messages.append(SystemMessage(content=item.content))
            else:
                messages.append(HumanMessage(content=item.content))
    messages.append(HumanMessage(content=payload.message))
    # Si le front envoie un payload deja structure (customer/supplier/line_items), on bascule en prepare_devis.
    intent = "chat"
    normalized = {}
    has_struct = False
    if isinstance(payload.metadata, dict):
        has_struct = any(key in payload.metadata for key in ["customer", "supplier", "line_items", "doc_type"])
        if has_struct:
            intent = "prepare_devis"
            structured = dict(payload.metadata)
            structured.setdefault("doc_type", "quote")
            normalized = {"intent": "prepare_devis", "doc_type": structured.get("doc_type", "quote"), "structured_payload": structured}

    # Heuristique: si message demande un devis/facture ou flag force_prepare, basculer en prepare_devis.
    text_lower = payload.message.lower()
    wants_quote = "devis" in text_lower
    wants_invoice = "facture" in text_lower
    if payload.force_prepare or (intent == "chat" and (wants_quote or wants_invoice)):
        intent = "prepare_devis"
        doc_type = "invoice" if wants_invoice else "quote"
        if not normalized:
            normalized = {"intent": intent, "doc_type": doc_type, "structured_payload": {"doc_type": doc_type}}

    state = {"messages": messages, "metadata": payload.metadata, "intent": intent, "normalized": normalized}
    result = invoke_agent(state, thread_id=payload.thread_id or "chat")
    reply = result.get("output")
    if reply is None and result.get("messages"):
        last_msg = result["messages"][-1]
        reply = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    reply_text = _format_ai_reply(reply)

    corrections = result.get("corrections") or []
    totals = result.get("totals") or {}
    missing_fields = result.get("missing_fields") or []
    return JSONResponse({"reply": reply_text, "raw_output": reply, "corrections": corrections, "totals": totals, "missing_fields": missing_fields})


@app.post("/analyze")
async def analyze(file: UploadFile, doc_type: str = "auto", thread_id: str | None = None):
    path = save_upload(file)
    initial_payload = {
        "doc_type": doc_type if doc_type != "auto" else "quote",
        "structured_payload": {"doc_type": doc_type if doc_type != "auto" else "quote"},
        "files": [path],
        "intent": "analyze",
    }
    state = {
        "intent": "analyze",
        "files": [path],
        "normalized": initial_payload,
        "messages": [HumanMessage(content=f"Analyse du fichier {file.filename}")],
    }
    result = invoke_agent(state, thread_id=thread_id or "analyze")
    output = result.get("output") or result
    totals = result.get("totals") or (result.get("tool_results") or {}).get("totals", {}).get("totals", {})
    corrections = result.get("corrections") or []

    # Validation et upsert conditionnel
    supabase_result = None
    if isinstance(output, dict) and output.get("doc_type"):
        validation = validate_devis_tool.invoke({"payload": output})
        supabase_result = upsert_document(output)
        totals = validation.get("totals", totals) or totals
        corrections = validation.get("issues", corrections) or corrections
        errors = validation.get("errors") or []
    else:
        errors = []

    formatted = _format_ai_reply(output)
    return JSONResponse(
        {"data": output, "formatted": formatted, "totals": totals, "corrections": corrections, "errors": errors, "supabase": supabase_result}
    )


class PrepareItem(BaseModel):
    description: str
    quantity: float = Field(gt=0)
    unit_price_ht: float = Field(gt=0)
    vat_rate: float = 20
    discount_rate: float = 0
    unit: Optional[str] = None


class PrepareDevisPayload(BaseModel):
    client_name: str
    client_address: Optional[str] = None
    client_contact: Optional[str] = None
    project_label: str
    items: List[PrepareItem]
    notes: Optional[str] = None
    payment_terms: Optional[str] = None
    penalties_late_payment: Optional[str] = None
    professional_liability: Optional[str] = None
    dtu_references: Optional[List[str]] = None
    doc_type: Literal["quote", "invoice"] = "quote"
    thread_id: Optional[str] = None

    @property
    def line_items(self):
        return [
            {
                "description": i.description,
                "quantity": i.quantity,
                "unit": i.unit,
                "unit_price_ht": i.unit_price_ht,
                "vat_rate": i.vat_rate,
                "discount_rate": i.discount_rate,
            }
            for i in self.items
        ]


@app.get("/prepare-devis/prefill")
async def prefill(user_id: str | None = None, client_prefix: str | None = None):
    lookup = supabase_lookup_tool.invoke({"query": client_prefix, "mode": "prefill"})
    return {"supabase": lookup.get("results", {}), "error": lookup.get("error")}


@app.post("/prepare-devis")
async def prepare_devis(payload: PrepareDevisPayload):
    structured_payload = {
        "doc_type": payload.doc_type,
        "number": f"AUTO-{date.today().strftime('%y%m%d')}",
        "date": date.today(),
        "customer": {
            "name": payload.client_name,
            "address": payload.client_address or "",
            "contact": payload.client_contact,
        },
        "supplier": {"name": "Auto", "address": "", "contact": ""},
        "line_items": payload.line_items,
        "notes": payload.notes,
        "payment_terms": payload.payment_terms,
        "penalties_late_payment": payload.penalties_late_payment,
        "professional_liability": payload.professional_liability,
        "dtu_references": payload.dtu_references,
    }

    cleaned = clean_lines_tool.invoke({"lines": structured_payload["line_items"]})
    structured_payload["line_items"] = cleaned.get("lines", structured_payload["line_items"])
    totals_calc = calculate_totals_tool.invoke({"lines": structured_payload["line_items"], "doc_type": payload.doc_type})
    totals = totals_calc.get("totals", {})

    state = {
        "intent": "prepare_devis",
        "normalized": {"intent": "prepare_devis", "doc_type": payload.doc_type, "structured_payload": structured_payload},
        "messages": [HumanMessage(content=f"Prepare un {payload.doc_type} pour {payload.client_name} ({payload.project_label})")],
    }

    result = invoke_agent(state, thread_id=payload.thread_id or "prepare-devis")
    output = result.get("output") or result
    validation = validate_devis_tool.invoke({"payload": structured_payload})
    corrections = validation.get("issues", result.get("corrections") or [])
    totals = validation.get("totals", totals) or totals
    errors = validation.get("errors") or []
    supabase_result = upsert_document(structured_payload)
    formatted = _format_ai_reply(output)

    return JSONResponse(
        {
            "data": output,
            "formatted": formatted,
            "totals": totals,
            "corrections": corrections,
            "errors": errors,
            "supabase": supabase_result,
            "cleaning": cleaned,
        }
    )


@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"status": "ok", "message": "API is running"})
