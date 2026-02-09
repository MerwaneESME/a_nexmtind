"""Configuration et modèles de données pour l'agent BTP V2."""
import os
import re
import logging
from datetime import date
from decimal import Decimal
from typing import List, Literal, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field, validator

# Retry logic for LLM calls
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError
    RETRY_AVAILABLE = True
except ImportError:
    RETRY_AVAILABLE = False

load_dotenv()

logger = logging.getLogger(__name__)

SIRET_RE = re.compile(r"\b\d{14}\b")
SIREN_RE = re.compile(r"\b\d{9}\b")
TVA_FR_RE = re.compile(r"^FR[0-9A-Z]{0,2}\d{9}$")


def get_postgres_url() -> str | None:
    """Construct PostgreSQL URL from Supabase credentials.

    Returns:
        PostgreSQL connection URL or None if credentials are missing.
    """
    # Try environment variable first
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    # Build from Supabase credentials
    db_host = os.getenv("SUPABASE_DB_HOST", "")
    db_port = os.getenv("SUPABASE_DB_PORT", "5432")
    db_name = os.getenv("SUPABASE_DB_NAME", "postgres")
    db_password = os.getenv("SUPABASE_DB_PASSWORD", "")

    if not db_host or not db_password:
        logger.warning("No database credentials configured - checkpoint persistence disabled")
        return None

    # Format: postgresql://postgres:password@host:port/database
    return f"postgresql://postgres:{db_password}@{db_host}:{db_port}/{db_name}"


DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")
FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "gpt-4o-mini")
FAST_MODEL = os.getenv("LLM_FAST_MODEL", "gpt-4o-mini")  # Modèle rapide et économique
FAST_FALLBACK_MODEL = os.getenv("LLM_FAST_FALLBACK_MODEL", "gpt-3.5-turbo")

SYSTEM_PROMPT = """
Tu es un agent IA spécialisé dans le BTP. Tu analyses, prépares et valides des devis/factures en garantissant la conformité métier (TVA, pénalités, mentions obligatoires) et l'intégration Supabase.
Respecte toujours le schéma JSON attendu et garde les réponses concises, actionnables et orientées correction.
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


class LLMWithRetry:
    """Wrapper pour ajouter retry logic à un LLM."""

    def __init__(self, llm):
        self._llm = llm

        if RETRY_AVAILABLE:
            # Créer la méthode invoke avec retry
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, APIError)),
                reraise=True
            )
            def invoke_with_retry(*args, **kwargs):
                return self._llm.invoke(*args, **kwargs)

            self._invoke_method = invoke_with_retry
        else:
            self._invoke_method = self._llm.invoke

    def invoke(self, *args, **kwargs):
        """Invoke avec retry logic automatique."""
        return self._invoke_method(*args, **kwargs)

    def __getattr__(self, name):
        """Déléguer tous les autres attributs au LLM sous-jacent."""
        return getattr(self._llm, name)


def get_llm(model: str | None = None, temperature: float = 0):
    """Retourne le modèle principal avec fallback automatique et retry logic.

    Retry automatique (3 tentatives avec exponential backoff) sur:
    - RateLimitError (rate limit API)
    - APIConnectionError (erreur réseau)
    - APITimeoutError (timeout)
    - APIError (erreur API générique)
    """
    primary = ChatOpenAI(model=model or DEFAULT_MODEL, temperature=temperature)
    fallback = ChatOpenAI(model=FALLBACK_MODEL, temperature=temperature)
    llm = primary.with_fallbacks([fallback])
    return LLMWithRetry(llm)


def get_fast_llm(temperature: float = 0):
    """Small/fast model for quick answers and routing (no extra context/tools).

    Inclut retry logic automatique sur erreurs réseau/API.
    """
    primary = ChatOpenAI(model=FAST_MODEL, temperature=temperature)
    fallback = ChatOpenAI(model=FAST_FALLBACK_MODEL, temperature=temperature)
    llm = primary.with_fallbacks([fallback])
    return LLMWithRetry(llm)


def get_embeddings():
    """Sélectionne OpenAI embeddings par défaut, Mistral si configuré."""
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
