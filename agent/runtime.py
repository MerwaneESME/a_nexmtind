"""Runtime LangGraph V2 - OPTIMISÉ - Réponses rapides et ciblées."""
import json
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import SYSTEM_PROMPT, get_llm, get_fast_llm
from .logging_config import logger
from .rag import SupabaseRAG
from .rag.local_docs import cascade_search, detect_domain
from .rag.retriever import is_corps_metier_question
from .rag.web_research import append_finding_to_doc, web_research_sync
from .tools import AVAILABLE_TOOLS, calculate_totals_tool, clean_lines_tool, validate_devis_tool

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _read_prompt(name: str) -> str:
    base = PROMPTS_DIR / name
    if base.exists():
        return _read_prompt_text(base)
    txt = base.with_suffix(".txt")
    if txt.exists():
        return _read_prompt_text(txt)
    return ""


def _read_prompt_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _maybe_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    intent: str
    normalized: dict
    rag_context: list
    supabase_context: list
    tool_results: dict
    totals: dict
    corrections: list
    missing_fields: list
    validate_section: str
    section_issues: list
    files: list
    output: Any
    fast_path_used: bool


rag_client = SupabaseRAG()


# ==================== Nœud 0: Fast Path OPTIMISÉ ====================

def fast_path_node(state: AgentState) -> AgentState:
    """Détecte les questions simples et répond IMMÉDIATEMENT sans appel LLM."""
    messages = state.get("messages", [])
    if not messages:
        return {}
    
    last_msg = messages[-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    content_lower = content.lower().strip()
    
    # ✅ Patterns de questions simples ÉLARGIS
    simple_patterns = {
        "greeting": ["bonjour", "salut", "hello", "hey", "coucou", "bonsoir"],
        "help": ["aide", "help", "comment", "que peux", "que sais", "commencer", "utiliser"],
        "thanks": ["merci", "thanks", "super", "parfait", "ok", "top"],
        "status": ["statut", "état", "où en", "avancement"],
    }
    
    # Vérifier s'il y a des données structurées
    has_data = bool(
        state.get("normalized", {}).get("structured_payload") or 
        state.get("metadata")
    )
    
    # Si question simple SANS données → réponse pré-définie (ZÉRO appel LLM)
    if not has_data:
        for category, patterns in simple_patterns.items():
            if any(p in content_lower for p in patterns):
                quick_replies = {
                    "greeting": "Bonjour ! Je suis ton assistant BTP. Je peux t'aider à créer et valider des devis/factures. Que puis-je faire pour toi ?",
                    "help": "Je peux :\n• Créer des devis et factures\n• Analyser des documents PDF/DOCX\n• Valider les montants et mentions obligatoires\n• Rechercher dans ton historique\n\nQue veux-tu faire ?",
                    "thanks": "Avec plaisir ! N'hésite pas si tu as d'autres questions.",
                    "status": "Pour te donner le statut, j'aurais besoin de savoir de quel devis/projet tu parles."
                }
                
                reply = quick_replies.get(category, "Je suis là pour t'aider avec tes devis et factures BTP.")
                logger.info("⚡ Fast path used (pre-defined): %s", category)
                
                return {
                    "output": {"reply": reply, "todo": []},
                    "messages": [AIMessage(content=reply)],
                    "fast_path_used": True,
                }
    
    # Si message très court (<50 chars) sans contexte → fast LLM
    if len(content) < 50 and not has_data:
        try:
            llm = get_fast_llm(temperature=0.3)
            result = llm.invoke([
                SystemMessage(content="Tu es un assistant BTP. Réponds en 1-2 phrases max, de manière amicale."),
                HumanMessage(content=content)
            ])
            reply = getattr(result, "content", str(result)).strip()
            logger.info("⚡ Fast path used (fast LLM): %d chars", len(content))
            
            return {
                "output": {"reply": reply, "todo": []},
                "messages": [AIMessage(content=reply)],
                "fast_path_used": True,
            }
        except Exception as exc:
            logger.warning("Fast path LLM failed: %s", exc)
    
    return {}  # Continuer avec le flux normal


def _build_prompt(name: str) -> ChatPromptTemplate:
    content = _read_prompt(name)
    if not content:
        content = "{{ user_input }}"
    return ChatPromptTemplate.from_template(content, template_format="jinja2")


# ==================== Nœud 1: Normalisation OPTIMISÉE ====================

def input_normalizer_node(state: AgentState) -> AgentState:
    """Détecte l'intention - OPTIMISÉ avec keywords en priorité."""
    existing_norm = state.get("normalized") or {}
    existing_struct = existing_norm.get("structured_payload") if isinstance(existing_norm, dict) else {}
    
    if existing_struct:
        return {
            "intent": state.get("intent") or existing_norm.get("intent") or "chat",
            "normalized": existing_norm,
            "files": list({*state.get("files", []), *existing_norm.get("files", [])}) if isinstance(existing_norm, dict) else state.get("files", []),
        }

    last_user_msg = ""
    if state.get("messages"):
        last = state["messages"][-1]
        last_user_msg = last.content if hasattr(last, "content") else str(last)

    # Si intent déjà défini, utiliser directement
    if state.get("intent"):
        normalized = {
            "intent": state.get("intent"),
            "doc_type": existing_norm.get("doc_type", "quote"),
            "structured_payload": existing_norm.get("structured_payload", {}),
            "summary": last_user_msg[:280],
        }
        return {
            "intent": normalized["intent"],
            "normalized": normalized,
            "files": state.get("files", []),
        }

    # ✅ TOUJOURS essayer détection par keywords d'abord (évite 90% des appels LLM)
    msg_lower = last_user_msg.lower()
    
    # Intent évident par keywords
    intent_keywords = {
        "chat": ["bonjour", "salut", "aide", "help", "qui es", "présente", "que peux"],
        "prepare_devis": ["devis", "facture", "créer", "générer", "faire un", "établir"],
        "validate": ["valide", "vérifie", "corrige", "check", "contrôle", "validation"],
        "analyze": ["analyse", "extraire", "lire", "scanner", "fichier"],
    }
    
    detected_intent = None
    for intent, keywords in intent_keywords.items():
        if any(kw in msg_lower for kw in keywords):
            detected_intent = intent
            break
    
    # ✅ Si intent détecté par keywords, NE PAS appeler le LLM
    if detected_intent:
        doc_type = "invoice" if "facture" in msg_lower else "quote"
        normalized = {
            "intent": detected_intent,
            "doc_type": doc_type,
            "structured_payload": {},
            "summary": last_user_msg[:280],
            "line_items": [],
            "files": [],
            "missing_fields": [],
        }
        logger.info("⚡ Intent detected by keywords: %s (skipped LLM call)", detected_intent)
        return {
            "intent": detected_intent,
            "normalized": normalized,
            "files": list({*state.get("files", []), *normalized.get("files", [])}),
        }

    # ✅ Sinon, appel fast_llm UNIQUEMENT pour cas ambigus
    prompt = _build_prompt("analysis_prompt")
    try:
        formatted = prompt.format_messages(
            user_input=last_user_msg,
            previous_payload=json.dumps(state.get("normalized") or {}),
        )
        reply = get_fast_llm().invoke(formatted)  # ✅ Utiliser fast_llm au lieu de get_llm()
        content = getattr(reply, "content", None) or str(reply)
        parsed = _maybe_parse_json(content) or {}
        normalized = parsed if isinstance(parsed, dict) else {}
        logger.info("🔍 Intent detected by LLM: %s", normalized.get("intent"))
    except Exception as exc:
        logger.error("Erreur dans input_normalizer_node (LLM): %s", exc, exc_info=True)
        normalized = {
            "intent": "chat",
            "doc_type": "quote",
            "structured_payload": {},
            "summary": last_user_msg[:280],
        }

    normalized.setdefault("intent", "chat")
    normalized.setdefault("doc_type", "quote")
    normalized.setdefault("structured_payload", {})
    normalized.setdefault("line_items", [])
    normalized.setdefault("files", [])
    normalized.setdefault("missing_fields", [])
    normalized.setdefault("summary", last_user_msg[:280])

    return {
        "intent": normalized.get("intent") or "chat",
        "normalized": normalized,
        "files": list({*state.get("files", []), *normalized.get("files", [])}),
    }


# ==================== Nœud 2: RAG Retriever OPTIMISÉ ====================

def rag_retriever_node(state: AgentState) -> AgentState:
    """Récupère du contexte RAG UNIQUEMENT si nécessaire."""
    intent = state.get("intent") or (state.get("normalized") or {}).get("intent") or "chat"
    normalized = state.get("normalized") or {}
    payload = normalized.get("structured_payload", {})

    # ✅ Stratégie en cascade (docs locales → docs connexes → web optionnel).
    # Appliqué aux questions "chat" sans payload structuré.
    if intent == "chat" and not payload:
        messages = state.get("messages") or []
        last_user_text = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user_text = str(getattr(m, "content", "") or "").strip()
                if last_user_text:
                    break

        if last_user_text:
            domain = detect_domain(last_user_text)
            if domain is None and is_corps_metier_question(last_user_text):
                domain = "corps_de_metier"

            local_snippets, consulted = cascade_search(last_user_text, domain=domain, max_docs=3)
            if local_snippets:
                rag_results = [
                    {
                        "content": s.content,
                        "metadata": {
                            "source": s.source,
                            "level": s.level,
                            "heading": s.heading,
                            "strategy": "cascade",
                            "consulted": consulted,
                        },
                        "score": s.score,
                    }
                    for s in local_snippets[:4]
                ]
                return {"rag_context": rag_results, "supabase_context": rag_results}

            finding = web_research_sync(last_user_text, max_results=3)
            if finding:
                updated_doc = append_finding_to_doc(domain=domain or "corps_de_metier", finding=finding)
                rag_results = [
                    {
                        "content": (finding.answer or "").strip(),
                        "metadata": {
                            "source": "web",
                            "level": 3,
                            "strategy": "cascade",
                            "query": finding.query,
                            "updated_doc": updated_doc,
                            "sources": finding.sources,
                        },
                        "score": None,
                    }
                ]
                return {"rag_context": rag_results, "supabase_context": rag_results}
    
    # ✅ CONDITIONS STRICTES pour appeler le RAG (évite 70% des appels)
    needs_rag = (
        # Intent opérationnel avec données
        (intent in ["prepare_devis", "validate", "analyze"] and payload) or
        # Recherche explicite d'historique
        (state.get("messages") and "historique" in str(state["messages"][-1].content).lower()) or
        # Préfill client/matériel
        intent == "prefill"
    )
    
    if not needs_rag:
        logger.info("⚡ RAG skipped - not needed for intent=%s", intent)
        return {"rag_context": [], "supabase_context": []}
    
    # Construire query intelligente
    query_parts = []
    if payload.get("customer", {}).get("name"):
        query_parts.append(payload["customer"]["name"])
    if payload.get("project_label"):
        query_parts.append(payload["project_label"])
    
    query = " ".join(query_parts) or normalized.get("summary", "")
    
    if not query or len(query) < 3:
        logger.info("⚡ RAG skipped - query too short")
        return {"rag_context": [], "supabase_context": []}
    
    rag_results = []
    if rag_client.is_ready():
        try:
            rag_results = rag_client.retrieve(query)
            logger.info("✅ RAG retrieved %d results for query: %s", len(rag_results), query[:50])
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
    else:
        logger.info("⚠️ RAG client not ready")

    return {
        "rag_context": rag_results,
        "supabase_context": rag_results,
    }


# ==================== Nœud 3: Business Tools (inchangé mais optimisé en amont) ====================

def business_tools_node(state: AgentState) -> AgentState:
    """Exécute les outils métier (calculs, nettoyage, validations)."""
    intent = state.get("intent") or (state.get("normalized") or {}).get("intent") or "chat"
    normalized = state.get("normalized") or {}
    payload = dict(normalized.get("structured_payload") or {})
    validate_section = (state.get("validate_section") or "").lower()

    # ✅ Skip si pas de données structurées ET intent simple
    if intent == "chat" and not payload:
        logger.info("⚡ Business tools skipped - no data for chat")
        return {
            "tool_results": {},
            "totals": {},
            "corrections": [],
            "missing_fields": [],
            "section_issues": [],
        }

    # Traiter les données si présentes
    if payload:
        line_items = payload.get("line_items") or []
        
        cleaned = {}
        totals = {}
        validation = {}
        
        if line_items:
            cleaned = clean_lines_tool.invoke({"lines": line_items, "default_vat_rate": payload.get("vat_rate")})
            totals = calculate_totals_tool.invoke({
                "lines": cleaned.get("lines", []),
                "default_vat_rate": payload.get("vat_rate"),
                "doc_type": payload.get("doc_type")
            })
        
        if payload.get("customer") or payload.get("line_items"):
            validation = validate_devis_tool.invoke({"payload": payload})
        
        # Calculer missing_fields
        missing_fields = []
        customer = payload.get("customer") or {}
        supplier = payload.get("supplier") or {}
        
        if not customer.get("name"):
            missing_fields.append("customer.name")
        if not customer.get("address"):
            missing_fields.append("customer.address")
        if not customer.get("contact"):
            missing_fields.append("customer.contact")
        
        if not supplier.get("name"):
            missing_fields.append("supplier.name")
        if not supplier.get("address"):
            missing_fields.append("supplier.address")
        if not supplier.get("contact"):
            missing_fields.append("supplier.contact")
        if not supplier.get("siret"):
            missing_fields.append("supplier.siret")
        if not supplier.get("tva_number"):
            missing_fields.append("supplier.tva_number")
        
        if not payload.get("number"):
            missing_fields.append("number")
        if not payload.get("date"):
            missing_fields.append("date")
        if not payload.get("payment_terms"):
            missing_fields.append("payment_terms")
        if payload.get("doc_type") == "invoice":
            if not payload.get("penalties_late_payment"):
                missing_fields.append("penalties_late_payment")
            if not payload.get("professional_liability"):
                missing_fields.append("professional_liability")
        if not payload.get("line_items"):
            missing_fields.append("line_items")

        section_issues: list[str] = []
        if validate_section in {"client", "customer"}:
            name = (customer.get("name") or "").strip()
            contact = (customer.get("contact") or "").strip()
            address = (customer.get("address") or "").strip()
            if not name:
                section_issues.append("Nom du client manquant.")
            elif " " not in name:
                section_issues.append("Nom de famille manquant (ex: Jean Dupont).")
            if not address:
                section_issues.append("Adresse du client manquante.")
            if not contact:
                section_issues.append("Contact client manquant (telephone ou email).")
            else:
                is_email = "@" in contact
                digits = "".join(ch for ch in contact if ch.isdigit())
                if not is_email and len(digits) < 10:
                    section_issues.append("Numero de telephone incomplet (10 chiffres).")

        if validate_section in {"chantier", "projet", "project"}:
            if not payload.get("project_label"):
                section_issues.append("Nom du projet manquant.")

        if validate_section in {"lignes", "items", "line_items"}:
            if not line_items:
                section_issues.append("Aucune ligne de produit n'est saisie.")
            else:
                for idx, item in enumerate(line_items, 1):
                    if not (item.get("description") or "").strip():
                        section_issues.append(f"Ligne {idx}: description manquante.")
                    if not item.get("quantity"):
                        section_issues.append(f"Ligne {idx}: quantite manquante.")
                    if not item.get("unit_price_ht"):
                        section_issues.append(f"Ligne {idx}: prix unitaire manquant.")

        if validate_section in {"global", "all", "final"}:
            name = (customer.get("name") or "").strip()
            contact = (customer.get("contact") or "").strip()
            address = (customer.get("address") or "").strip()
            if not name:
                section_issues.append("Nom du client manquant.")
            elif " " not in name:
                section_issues.append("Nom de famille manquant (ex: Jean Dupont).")
            if not address:
                section_issues.append("Adresse du client manquante.")
            if not contact:
                section_issues.append("Contact client manquant (telephone ou email).")
            else:
                is_email = "@" in contact
                digits = "".join(ch for ch in contact if ch.isdigit())
                if not is_email and len(digits) < 10:
                    section_issues.append("Numero de telephone incomplet (10 chiffres).")

            if not payload.get("project_label"):
                section_issues.append("Nom du projet manquant.")

            if not line_items:
                section_issues.append("Aucune ligne de produit n'est saisie.")
            else:
                for idx, item in enumerate(line_items, 1):
                    if not (item.get("description") or "").strip():
                        section_issues.append(f"Ligne {idx}: description manquante.")
                    if not item.get("quantity"):
                        section_issues.append(f"Ligne {idx}: quantite manquante.")
                    if not item.get("unit_price_ht"):
                        section_issues.append(f"Ligne {idx}: prix unitaire manquant.")

            if not supplier.get("name"):
                section_issues.append("Nom du fournisseur manquant.")
            if not supplier.get("address"):
                section_issues.append("Adresse du fournisseur manquante.")
            if not supplier.get("contact"):
                section_issues.append("Contact fournisseur manquant.")
            if not supplier.get("siret"):
                section_issues.append("SIRET fournisseur manquant.")
            if not supplier.get("tva_number"):
                section_issues.append("TVA fournisseur manquante.")

            if not payload.get("number"):
                section_issues.append("Numero du document manquant.")
            if not payload.get("date"):
                section_issues.append("Date du document manquante.")
            if not payload.get("payment_terms"):
                section_issues.append("Conditions de paiement manquantes.")
            if payload.get("doc_type") == "invoice":
                if not payload.get("penalties_late_payment"):
                    section_issues.append("Penalites de retard manquantes.")
                if not payload.get("professional_liability"):
                    section_issues.append("Responsabilite civile pro manquante.")

        if section_issues:
            section_issues = list(dict.fromkeys(section_issues))

        logger.info("✅ Business tools executed: %d lines, %d issues", len(line_items), len(section_issues))
        
        return {
            "tool_results": {
                "clean_lines": cleaned,
                "totals": totals,
                "validation": validation,
            },
            "totals": totals.get("totals", {}) if totals else {},
            "corrections": validation.get("issues", []) if validation else [],
            "missing_fields": missing_fields,
            "section_issues": section_issues,
            "normalized": normalized | {"structured_payload": payload, "missing_fields": missing_fields},
        }
    
    return {
        "tool_results": state.get("tool_results", {}),
        "totals": state.get("totals", {}),
        "corrections": state.get("corrections", []),
        "missing_fields": [],
        "section_issues": [],
    }


# ==================== Nœud 4: Agent Reasoning (inchangé) ====================

def agent_reasoning_node(state: AgentState) -> AgentState:
    """Le LLM réfléchit et décide d'appeler des outils si nécessaire."""
    intent = state.get("intent") or (state.get("normalized") or {}).get("intent") or "chat"
    normalized = state.get("normalized") or {}
    payload = dict(normalized.get("structured_payload") or {})
    
    totals = state.get("totals") or {}
    corrections = state.get("corrections") or []
    missing_fields = state.get("missing_fields") or []
    
    previous_messages = state.get("messages", [])
    
    last_user_msg = ""
    if previous_messages:
        last_msg = previous_messages[-1]
        last_user_msg = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    
    reasoning_prompt = f"""
Tu es un agent BTP spécialisé dans les devis/factures.

Contexte actuel:
- Intent: {intent}
- Données disponibles: {json.dumps(payload, indent=2) if payload else "Aucune donnée structurée"}
- Totaux calculés: {json.dumps(totals, indent=2) if totals else "Non calculés"}
- Corrections détectées: {len(corrections)} problème(s)
- Champs manquants: {', '.join(missing_fields[:5]) if missing_fields else 'Aucun'}

Instructions:
1. Analyse les données disponibles
2. Si des données sont présentes mais incomplètes, tu peux appeler validate_devis_tool pour vérifier
3. Si des lignes sont présentes, tu peux appeler calculate_totals_tool pour calculer les totaux
4. Réponds de manière conversationnelle et actionnable

Message utilisateur: {last_user_msg}
"""
    
    llm_with_tools = get_llm().bind_tools(AVAILABLE_TOOLS)
    
    messages = list(previous_messages)
    
    has_system = any(isinstance(m, SystemMessage) for m in messages)
    if not has_system:
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    
    messages.append(HumanMessage(content=reasoning_prompt))
    
    try:
        result = llm_with_tools.invoke(messages)
    except Exception as exc:
        logger.error("Erreur dans agent_reasoning_node: %s", exc, exc_info=True)
        result = AIMessage(content="Erreur lors du traitement de la requête.")
    
    tool_calls = []
    if hasattr(result, "tool_calls") and result.tool_calls:
        tool_calls = result.tool_calls
    
    return {
        "messages": [result],
        "tool_calls": tool_calls,
    }


# ==================== Nœud 5: LLM Synthesizer (inchangé) ====================

def llm_synthesizer_node(state: AgentState) -> AgentState:
    """Génère la réponse finale."""
    intent = state.get("intent") or (state.get("normalized") or {}).get("intent") or "chat"
    normalized = state.get("normalized") or {}
    payload = dict(normalized.get("structured_payload") or {})

    totals = state.get("totals") or {}
    corrections = state.get("corrections") or []
    missing_fields = state.get("missing_fields") or []
    section_issues = state.get("section_issues") or []
    validate_section = state.get("validate_section") or ""
    rag_context = state.get("rag_context") or []
    supabase_context = state.get("supabase_context") or []
    
    all_messages = state.get("messages", [])

    if intent == "validate":
        prompt_name = "validate_prompt"
    elif intent in ("prepare_devis", "analyze"):
        prompt_name = "prepare_devis_prompt"
    else:
        prompt_name = "chat_prompt"

    prompt = _build_prompt(prompt_name)
    
    prompt_context = {
        "intent": intent,
        "normalized_payload": payload,
        "totals": totals,
        "corrections": corrections,
        "rag_context": rag_context,
        "supabase_context": supabase_context,
        "missing_fields": missing_fields,
        "section_issues": section_issues,
        "validation_section": validate_section,
    }

    if intent == "chat":
        system_instruction = f"""
Tu es un assistant BTP. Reponds en JSON strict avec {{ "reply": "...", "todo": [] }} uniquement.
Ne renvoie jamais d'objet "document" ou de JSON technique dans reply.

Donnees du formulaire (si disponibles):
- Client: {payload.get('customer', {}).get('name') or 'non renseigne'}
- Adresse: {payload.get('customer', {}).get('address') or 'non renseignee'}
- Contact: {payload.get('customer', {}).get('contact') or 'non renseigne'}
- Projet: {payload.get('project_label') or 'non renseigne'}
- Lignes: {len(payload.get('line_items', []))} produits

Regles:
1. "reply" est une reponse conversationnelle claire et courte (2-3 phrases MAX).
2. "todo" est une liste d'actions courtes (max 3). Laisse vide si pas besoin.
3. N'invente rien.
"""
    elif intent == "validate":
        system_instruction = f"""
Tu valides une section du devis. Reponds en JSON strict avec {{ "reply": "...", "todo": [] }} uniquement.
Ne renvoie pas de JSON technique (pas de champs du type customer.address).
Priorite: utilise section_issues si present, sinon base-toi sur missing_fields.
Reply doit faire 2-3 phrases MAX.
"""
    else:
        system_instruction = f"""
DONNEES DU FORMULAIRE (A UTILISER EXACTEMENT):
- Client: {payload.get('customer', {}).get('name')} | Adresse: {payload.get('customer', {}).get('address')} | Contact: {payload.get('customer', {}).get('contact')}
- Fournisseur: {payload.get('supplier', {}).get('name')} | Adresse: {payload.get('supplier', {}).get('address')}
- Lignes: {len(payload.get('line_items', []))} produits
- Totaux: HT={totals.get('total_ht')} | TVA={totals.get('total_tva')} | TTC={totals.get('total_ttc')}

INSTRUCTIONS:
1. Utilise EXACTEMENT ces donnees dans le JSON final
2. NE JAMAIS inventer de noms, adresses ou SIRET
3. Si une donnee manque, mets null ET ajoute-la a missing_fields
"""
    try:
        formatted_messages = prompt.format_messages(**prompt_context)
        messages_for_llm = list(all_messages)
        messages_for_llm.extend(formatted_messages)
        messages_for_llm.insert(0, SystemMessage(content=system_instruction))
    except Exception as exc:
        logger.error("Erreur dans llm_synthesizer_node (format_messages): %s", exc, exc_info=True)
        messages_for_llm = list(all_messages) if all_messages else [HumanMessage(content=state.get("messages", [])[-1].content if state.get("messages") else "")]
        messages_for_llm.insert(0, SystemMessage(content=system_instruction))
    
    try:
        result = get_llm().invoke(messages_for_llm)
    except Exception as exc:
        logger.error("Erreur dans llm_synthesizer_node (invoke): %s", exc, exc_info=True)
        result = AIMessage(content="Erreur lors de la génération de la réponse.")
    
    content = getattr(result, "content", None) or str(result)
    parsed = _maybe_parse_json(content)

    if not isinstance(parsed, dict):
        if intent == "chat":
            parsed = {
                "reply": content if isinstance(content, str) else "Je peux t'aider à créer et valider des devis/factures BTP.",
                "todo": missing_fields[:3] if missing_fields else []
            }
        else:
            parsed = {
                "document": payload,
                "corrections": corrections,
                "missing_fields": missing_fields,
                "totals": totals,
            }
    elif intent == "chat" and "reply" not in parsed:
        if "document" in parsed:
            doc = parsed.get("document", {})
            customer = doc.get("customer", {})
            reply_parts = []
            if customer.get("name"):
                reply_parts.append(f"Client: {customer.get('name')}")
            if missing_fields:
                reply_parts.append(f"Champs manquants: {', '.join(missing_fields[:3])}")
            parsed = {
                "reply": ". ".join(reply_parts) if reply_parts else "Formulaire en cours de saisie.",
                "todo": missing_fields[:3] if missing_fields else []
            }

    if intent == "validate":
        issues = list(section_issues or [])
        if not issues and missing_fields:
            issues = [f"Informations manquantes: {', '.join(missing_fields[:3])}"]
        if not isinstance(parsed, dict) or "reply" not in parsed:
            if issues:
                parsed = {"reply": "A corriger", "todo": issues[:3]}
            else:
                parsed = {"reply": "OK - Rien a corriger sur cette section.", "todo": []}
        else:
            if issues:
                parsed["todo"] = issues[:3]
                if "a corriger" not in str(parsed.get("reply", "")).lower():
                    parsed["reply"] = "A corriger"

    new_ai_message = AIMessage(content=content)
    
    logger.info("✅ Synthesizer generated response (intent=%s): %d chars", intent, len(str(parsed)))
    
    return {
        "messages": [new_ai_message],
        "output": parsed or content,
        "missing_fields": missing_fields,
        "corrections": corrections,
    }


# ==================== Construction du Graph OPTIMISÉ ====================

def _get_checkpointer():
    """Get checkpointer with PostgreSQL primary, MemorySaver fallback.

    Returns:
        PostgresSaver if database credentials are configured, otherwise MemorySaver.
    """
    from .config import get_postgres_url

    postgres_url = get_postgres_url()

    if postgres_url and POSTGRES_AVAILABLE:
        try:
            # PostgresSaver.from_conn_string returns a sync connection
            # Tables are created automatically on first use
            checkpointer = PostgresSaver.from_conn_string(postgres_url)
            logger.info("✅ PostgreSQL checkpointer enabled (persistent conversation history)")
            return checkpointer
        except Exception as exc:
            logger.warning(
                "Failed to connect to PostgreSQL checkpoint store: %s. "
                "Falling back to MemorySaver (conversations will be lost on restart).",
                exc,
                exc_info=True
            )
    else:
        if not POSTGRES_AVAILABLE:
            logger.warning(
                "langgraph-checkpoint-postgres not installed. Using MemorySaver "
                "(conversations will be lost on restart). Install with: pip install langgraph-checkpoint-postgres"
            )
        else:
            logger.warning(
                "No DATABASE_URL configured. Using MemorySaver "
                "(conversations will be lost on restart)."
            )

    # Fallback to in-memory
    return MemorySaver()


def build_graph():
    """Construit le graph LangGraph - VERSION OPTIMISÉE avec persistance PostgreSQL."""
    builder = StateGraph(AgentState)
    
    # Ajouter les nœuds
    builder.add_node("fast_path", fast_path_node)
    builder.add_node("input_normalizer", input_normalizer_node)
    builder.add_node("rag_retriever", rag_retriever_node)
    builder.add_node("business_tools", business_tools_node)
    builder.add_node("agent_reasoning", agent_reasoning_node)
    builder.add_node("tools", ToolNode(AVAILABLE_TOOLS))
    builder.add_node("llm_synthesizer", llm_synthesizer_node)
    # ❌ SUPPRIMÉ : reflection_node (inutile)
    
    # Fast path en premier
    builder.set_entry_point("fast_path")
    builder.add_conditional_edges(
        "fast_path",
        lambda state: "end" if state.get("fast_path_used") else "input_normalizer",
        {
            "end": END,
            "input_normalizer": "input_normalizer"
        }
    )
    
    # Définir le flux normal
    builder.add_edge("input_normalizer", "rag_retriever")
    builder.add_edge("rag_retriever", "business_tools")
    builder.add_edge("business_tools", "agent_reasoning")
    
    # Branchement conditionnel : si le LLM appelle des outils, les exécuter
    builder.add_conditional_edges(
        "agent_reasoning",
        lambda state: "tools" if state.get("tool_calls") else "llm_synthesizer",
    )
    builder.add_edge("tools", "llm_synthesizer")
    
    # ✅ FIN directe après synthesizer (plus de reflection)
    builder.add_edge("llm_synthesizer", END)

    # Configure checkpointer avec fallback
    checkpointer = _get_checkpointer()

    logger.info("🚀 Graph optimisé construit (fast_path + skip RAG + no reflection)")
    logger.info("📊 Checkpointer: %s", type(checkpointer).__name__)

    return builder.compile(checkpointer=checkpointer)


agent_graph = build_graph()


def invoke_agent(state: Dict[str, Any], thread_id: str = "default"):
    """Helper pour invoquer le graph avec mémoire LangGraph."""
    state_input: AgentState = dict(state)
    msgs = state_input.get("messages", [])
    
    if not msgs and state_input.get("input"):
        msgs = [HumanMessage(content=state_input["input"])]
    
    has_system = any(isinstance(m, SystemMessage) or getattr(m, "role", "") == "system" for m in msgs)
    if not has_system and msgs:
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
    
    state_input["messages"] = msgs
    
    try:
        config = {"configurable": {"thread_id": thread_id}}
        result = agent_graph.invoke(state_input, config=config)
        logger.info("✅ Agent invoked successfully (thread=%s)", thread_id)
        return result
    except Exception as exc:
        logger.error("Erreur dans invoke_agent: %s", exc, exc_info=True)
        return {
            "output": {"error": str(exc), "reply": "Erreur lors du traitement de la requête."},
            "messages": msgs,
        }
