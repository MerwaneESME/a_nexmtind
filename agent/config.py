"""Configuration et modèles de données pour l'agent BTP V2."""
import os
import re
import logging
from datetime import date
from decimal import Decimal
from typing import List, Literal, Optional, Any

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
    """Construct PostgreSQL URL from Supabase credentials."""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    db_host = os.getenv("SUPABASE_DB_HOST", "")
    db_port = os.getenv("SUPABASE_DB_PORT", "5432")
    db_name = os.getenv("SUPABASE_DB_NAME", "postgres")
    db_password = os.getenv("SUPABASE_DB_PASSWORD", "")

    if not db_host or not db_password:
        logger.warning("No database credentials configured - checkpoint persistence disabled")
        return None

    return f"postgresql://postgres:{db_password}@{db_host}:{db_port}/{db_name}"


# --- CONFIGURATION LLM ---

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "gpt-4o-mini")
FAST_MODEL = os.getenv("LLM_FAST_MODEL", "gpt-4o-mini")
FAST_FALLBACK_MODEL = os.getenv("LLM_FAST_FALLBACK_MODEL", "gpt-3.5-turbo")


# --- PROMPTS SYSTÈME ---

# 1. PROMPT COURT (Pour le Reasoning Node / Router)
# Utilisé pour la prise de décision technique et l'appel d'outils.
SYSTEM_PROMPT = """
Tu es un agent IA spécialisé dans le BTP. Tu analyses, prépares et valides des devis/factures en garantissant la conformité métier.
Respecte toujours le schéma JSON attendu.
""".strip()

# 2. PROMPT EXPERT (Pour le Synthesizer Node)
# C'est ici que réside l'intelligence "métier" et le style de réponse.
SYNTHESIZER_SYSTEM_PROMPT = """
## 1. IDENTITÉ & MISSION
Tu es NEXTMIND, l'assistant IA expert du BTP en France.
Cible : Artisans et particuliers.
Style : Direct, technique, "terrain", dense. Pas de formules de politesse excessives.
Rôle : Aider à chiffrer, diagnostiquer, vérifier les normes (DTU) et préparer les chantiers.

## 2. GESTION DU SAVOIR (RAG vs GÉNÉRAL)
- **Priorité absolue au CONTEXTE RAG** (référentiel métier) s'il est fourni.
- **Citation** : Si tu utilises une info du RAG (prix, technique), indique : "D'après le référentiel : [info]".
- **Absence RAG** : Si l'info manque, utilise tes connaissances générales BTP France en précisant : "Estimation hors référentiel (moyenne marché)".
- **Incertitude** : Ne jamais inventer. Si doute, propose une action : "À vérifier sur site par sondage".

## 3. RÈGLES D'OR ANTI-HALLUCINATION
1. **DTU/Normes** : Ne cite un numéro de DTU ou une loi que si tu es 100% sûr ou s'il est dans le RAG. Sinon, parle de "règles de l'art".
2. **Prix** : Donne toujours une **fourchette** (Min - Max) + le **facteur de variation** (ex: accès, état du support).
3. **Diagnostic** : N'affirme jamais la cause d'une panne à distance. Liste les causes probables par ordre de fréquence.

## 4. INTELLIGENCE SITUATIONNELLE & REFORMULATION
Avant de répondre, analyse l'intention.
Si la demande est floue, reformule implicitement en tâches techniques :
*Ex: "Refaire sdb" -> Dépose, plomberie/elec, étanchéité (SPEC/SELI), carrelage, appareillage.*

**CAS SPÉCIAL : DIAGNOSTIC / PANNE**
Si mots-clés : "problème", "fuite", "fissure", "panne", "bruit".
ADOPTE IMMÉDIATEMENT CETTE STRUCTURE SPÉCIALE :
1. **Sécurité/Urgence** : Y a-t-il un risque immédiat ? (Eau, Elec, Structure).
2. **Checklist Points de Contrôle** : 4 à 6 points à vérifier (du plus simple au plus complexe).
3. **Signaux d'Alerte** : Ce qui doit faire stopper les travaux.
4. **Estimation** : Seulement après les vérifications.

## 5. STRUCTURE DE RÉPONSE STANDARD (Hors Diagnostic)
Pour les questions de conseils, devis ou techniques :

1.  **Réponse Directe (La Synthèse)**
    * 1 phrase avec la réponse clé (Prix, Délai ou Faisabilité).
    * Utilise le **gras** pour les valeurs clés (ex: **45 - 60 €/m²**).

2.  **Détail Technique & Mise en Œuvre**
    * Étapes chronologiques logiques.
    * Matériaux recommandés et temps de mise en œuvre (Cadence).
    * Vocabulaire pro exigé : *Ragréage, primaire d'accrochage, calepinage, ébrasement, tableau, incorporation...*

3.  **Points de Vigilance (Risk Management)**
    * Lister 2-3 pièges classiques (ex: support bloqué, temps de séchage, compatibilité matériaux).
    * Mentionner "Signal d'alerte" si applicable.

4.  **Prochaine Action**
    * Termine par une question utile ou une proposition concrète (Mini-devis, Checklist).

## 6. FORMATAGE FINAL
- Utilise des listes à puces pour la lisibilité.
- Utilise le **gras** (`**valeur**`) UNIQUEMENT pour : Prix, Délais, Avertissements de sécurité.
- Pas de phrases vides ("J'espère que cela vous aide"). Va droit au but.
""".strip()


# --- DATA MODELS ---

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
        return _norm_digits(v) if v else v

    @validator("siren")
    def valid_siren(cls, v):
        return _norm_digits(v) if v else v

    @validator("tva_number")
    def valid_tva(cls, v):
        return v.replace(" ", "").upper() if v else v


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


class InvoiceSchema(QuoteSchema):
    doc_type: Literal["invoice"]
    quote_ref: Optional[str] = None
    due_date: Optional[date] = None
    amount_paid: Optional[Decimal] = Decimal(0)


# --- WRAPPER LLM ROBUSTE ---

class LLMWithRetry:
    """Wrapper pour ajouter retry logic à un LLM LangChain standard."""
    
    def __init__(self, llm):
        self._llm = llm

    def invoke(self, *args, **kwargs):
        if RETRY_AVAILABLE:
            return self._invoke_with_retry(*args, **kwargs)
        return self._llm.invoke(*args, **kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, APIError)),
        reraise=True
    )
    def _invoke_with_retry(self, *args, **kwargs):
        """Méthode interne décorée par tenacity."""
        return self._llm.invoke(*args, **kwargs)

    def bind_tools(self, tools: list):
        """Passe la méthode bind_tools au LLM sous-jacent."""
        return self._llm.bind_tools(tools)
    
    def with_fallbacks(self, fallbacks: list):
        """Permet de chainer les fallbacks."""
        return LLMWithRetry(self._llm.with_fallbacks(fallbacks))

    def __getattr__(self, name: str) -> Any:
        """Délègue les autres attributs au LLM sous-jacent."""
        return getattr(self._llm, name)


def get_llm(model: str | None = None, temperature: float = 0):
    """Retourne le modèle principal avec fallback automatique et retry logic."""
    # Modèle principal
    primary = ChatOpenAI(model=model or DEFAULT_MODEL, temperature=temperature)
    # Modèle de secours (souvent moins cher ou plus fiable sur la dispo)
    fallback = ChatOpenAI(model=FALLBACK_MODEL, temperature=temperature)
    
    # On lie le fallback au principal
    llm_with_fallback = primary.with_fallbacks([fallback])
    
    # On ajoute la couche de retry réseau
    return LLMWithRetry(llm_with_fallback)


def get_fast_llm(temperature: float = 0):
    """Modèle rapide pour le routing et les réponses courtes."""
    primary = ChatOpenAI(model=FAST_MODEL, temperature=temperature)
    fallback = ChatOpenAI(model=FAST_FALLBACK_MODEL, temperature=temperature)
    llm_with_fallback = primary.with_fallbacks([fallback])
    return LLMWithRetry(llm_with_fallback)


def get_embeddings():
    """Sélectionne OpenAI embeddings par défaut, Mistral si configuré."""
    try:
        from langchain_mistralai import MistralAIEmbeddings
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if mistral_key:
            return MistralAIEmbeddings(model="mistral-embed")
    except (ImportError, Exception):
        pass
        
    return OpenAIEmbeddings()