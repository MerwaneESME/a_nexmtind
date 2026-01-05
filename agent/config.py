import os
import re
from datetime import date
from decimal import Decimal
from typing import List, Literal, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field, validator

load_dotenv()

SIRET_RE = re.compile(r"\b\d{14}\b")
SIREN_RE = re.compile(r"\b\d{9}\b")
TVA_FR_RE = re.compile(r"^FR[0-9A-Z]{0,2}\d{9}$")

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "gpt-4o")


SYSTEM_PROMPT = """
Tu es un agent IA specialise dans le BTP. Tu analyses, prepares et valides des devis/factures en garantissant la conformite metier (TVA, penalites, mentions obligatoires) et l'integration Supabase.
Respecte toujours le schema JSON attendu et garde les reponses concises, actionnables et orientees correction.
""".strip()


def _norm_digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


class LineItem(BaseModel):
    description: str = Field(..., min_length=2)
    quantity: Decimal = Field(..., ge=0)
    unit: Optional[str] = Field(None, max_length=16)
    unit_price_ht: Decimal = Field(..., ge=0)
    vat_rate: Decimal = Field(..., ge=0)
    discount_rate: Optional[Decimal] = Field(0, ge=0, le=100)

    @property
    def total_ht(self) -> Decimal:
        base = self.quantity * self.unit_price_ht
        if self.discount_rate:
            base = base * (Decimal(1) - self.discount_rate / Decimal(100))
        return base

    @property
    def total_tva(self) -> Decimal:
        return self.total_ht * (self.vat_rate / Decimal(100))

    @property
    def total_ttc(self) -> Decimal:
        return self.total_ht + self.total_tva


class Party(BaseModel):
    name: str
    address: str
    siret: Optional[str] = None
    siren: Optional[str] = None
    tva_number: Optional[str] = None
    contact: Optional[str] = None

    @validator("siret")
    def valid_siret(cls, v):
        if v:
            return _norm_digits(v)
        return v

    @validator("siren")
    def valid_siren(cls, v):
        if v:
            return _norm_digits(v)
        return v

    @validator("tva_number")
    def valid_tva(cls, v):
        if v:
            normalized = v.replace(" ", "").upper()
            return normalized
        return v


class QuoteSchema(BaseModel):
    doc_type: Literal["quote"]
    number: str
    date: date
    supplier: Party
    customer: Party
    line_items: List[LineItem]
    payment_terms: Optional[str] = None
    penalties_late_payment: Optional[str] = None
    professional_liability: Optional[str] = None
    dtu_references: Optional[List[str]] = None
    notes: Optional[str] = None

    @validator("line_items")
    def non_empty_items(cls, v):
        if not v:
            raise ValueError("Le devis doit contenir au moins une ligne")
        return v

    def totals(self):
        total_ht = sum(i.total_ht for i in self.line_items)
        total_tva = sum(i.total_tva for i in self.line_items)
        total_ttc = total_ht + total_tva
        return {"total_ht": total_ht, "total_tva": total_tva, "total_ttc": total_ttc}


class InvoiceSchema(QuoteSchema):
    doc_type: Literal["invoice"]
    quote_ref: Optional[str] = None
    due_date: Optional[date] = None
    amount_paid: Optional[Decimal] = Decimal(0)


def get_llm(model: str | None = None):
    """Retourne le modele principal avec fallback automatique."""
    primary = ChatOpenAI(model=model or DEFAULT_MODEL, temperature=0)
    fallback = ChatOpenAI(model=FALLBACK_MODEL, temperature=0)
    return primary.with_fallbacks([fallback])


def llm(model: str | None = None):
    """Alias retrocompatible."""
    return get_llm(model)


def get_embeddings():
    """Selectionne OpenAI embeddings par defaut, Mistral si configure."""
    try:
        from langchain_mistralai import MistralAIEmbeddings  # type: ignore
    except Exception:
        MistralAIEmbeddings = None

    mistral_key = os.getenv("MISTRAL_API_KEY")
    if mistral_key and MistralAIEmbeddings is not None:
        try:
            return MistralAIEmbeddings(model="mistral-embed")
        except Exception:
            pass
    return OpenAIEmbeddings()
