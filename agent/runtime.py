"""Runtime LangGraph V2 - Agent intelligent avec function calling et réflexion."""
import json
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import SYSTEM_PROMPT, get_llm
from .logging_config import logger
from .rag import SupabaseRAG
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
    # Tolere plusieurs encodages Windows (utf-16/cp1252) en plus de utf-8.
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
    should_continue: bool
    reflection_reason: str


rag_client = SupabaseRAG()


def _build_prompt(name: str) -> ChatPromptTemplate:
    content = _read_prompt(name)
    if not content:
        content = "{{ user_input }}"
    return ChatPromptTemplate.from_template(content, template_format="jinja2")


# ==================== Nœud 1: Normalisation de l'input ====================

def input_normalizer_node(state: AgentState) -> AgentState:
    """Détecte l'intention et normalise l'entrée utilisateur."""
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

    # Sinon, utiliser le prompt d'analyse
    prompt = _build_prompt("analysis_prompt")
    try:
        formatted = prompt.format_messages(
            user_input=last_user_msg,
            previous_payload=json.dumps(state.get("normalized") or {}),
        )
        reply = get_llm().invoke(formatted)
        content = getattr(reply, "content", None) or str(reply)
        parsed = _maybe_parse_json(content) or {}
        normalized = parsed if isinstance(parsed, dict) else {}
    except Exception:
        # Fallback si le prompt ne peut pas etre formate
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


# ==================== Nœud 2: RAG Retriever ====================

def rag_retriever_node(state: AgentState) -> AgentState:
    """Récupère du contexte RAG depuis SupabaseVectorStore."""
    normalized = state.get("normalized") or {}
    payload = normalized.get("structured_payload", {})
    query = normalized.get("summary") or payload.get("project_label") or payload.get("notes") or ""
    
    rag_results = []
    if query and rag_client.is_ready():
        try:
            rag_results = rag_client.retrieve(query)
        except Exception:
            pass

    return {
        "rag_context": rag_results,
        "supabase_context": rag_results,
    }


# ==================== Nœud 3: Business Tools ====================

def business_tools_node(state: AgentState) -> AgentState:
    """Exécute les outils métier (calculs, nettoyage, validations)."""
    intent = state.get("intent") or (state.get("normalized") or {}).get("intent") or "chat"
    normalized = state.get("normalized") or {}
    payload = dict(normalized.get("structured_payload") or {})
    validate_section = (state.get("validate_section") or "").lower()

    # ✅ TOUJOURS traiter les données si présentes, même en mode chat
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
    
    # Pas de données structurées
    return {
        "tool_results": state.get("tool_results", {}),
        "totals": state.get("totals", {}),
        "corrections": state.get("corrections", []),
        "missing_fields": [],
        "section_issues": [],
    }


# ==================== Nœud 4: Agent Reasoning (Function Calling) ====================

def agent_reasoning_node(state: AgentState) -> AgentState:
    """Le LLM réfléchit et décide d'appeler des outils si nécessaire."""
    intent = state.get("intent") or (state.get("normalized") or {}).get("intent") or "chat"
    normalized = state.get("normalized") or {}
    payload = dict(normalized.get("structured_payload") or {})
    
    totals = state.get("totals") or {}
    corrections = state.get("corrections") or []
    missing_fields = state.get("missing_fields") or []
    
    # Construire le contexte pour le LLM
    reasoning_prompt = f"""
Tu es un agent BTP spécialisé dans les devis/factures.

Contexte actuel:
- Intent: {intent}
- Données disponibles: {json.dumps(payload, indent=2) if payload else "Aucune donnée structurée"}
- Totaux calculés: {json.dumps(totals, indent=2) if totals else "Non calculés"}
- Corrections détectées: {len(corrections)} problème(s)
- Champs manquants: {', '.join(missing_fields) if missing_fields else 'Aucun'}

Instructions:
1. Analyse les données disponibles
2. Si des données sont présentes mais incomplètes, tu peux appeler validate_devis_tool pour vérifier
3. Si des lignes sont présentes, tu peux appeler calculate_totals_tool pour calculer les totaux
4. Réponds de manière conversationnelle et actionnable

Message utilisateur: {state.get("messages", [])[-1].content if state.get("messages") else ""}
"""
    
    # ✅ Donner les outils au LLM pour qu'il décide
    llm_with_tools = get_llm().bind_tools(AVAILABLE_TOOLS)
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=reasoning_prompt),
    ]
    
    result = llm_with_tools.invoke(messages)
    
    # Vérifier si le LLM a appelé des outils
    tool_calls = []
    if hasattr(result, "tool_calls") and result.tool_calls:
        tool_calls = result.tool_calls
    
    return {
        "messages": [result] if tool_calls else [],
        "tool_calls": tool_calls,
    }


# ==================== Nœud 5: LLM Synthesizer ====================

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

    # Choisir le bon prompt selon l'intent
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

    # Adapter system_instruction selon l'intent
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
1. "reply" est une reponse conversationnelle claire et courte.
2. "todo" est une liste d'actions courtes (max 4). Laisse vide si pas besoin.
3. N'invente rien.
"""
    elif intent == "validate":
        system_instruction = f"""
Tu valides une section du devis. Reponds en JSON strict avec {{ "reply": "...", "todo": [] }} uniquement.
Ne renvoie pas de JSON technique (pas de champs du type customer.address).
Priorite: utilise section_issues si present, sinon base-toi sur missing_fields.
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
        messages = prompt.format_messages(**prompt_context)
    except Exception:
        messages = [HumanMessage(content=state.get("messages", [])[-1].content if state.get("messages") else "")]
    messages.insert(0, SystemMessage(content=system_instruction))
    
    result = get_llm().invoke(messages)
    content = getattr(result, "content", None) or str(result)
    parsed = _maybe_parse_json(content)

    # Fallback selon l'intent
    if not isinstance(parsed, dict):
        if intent == "chat":
            parsed = {
                "reply": content if isinstance(content, str) else "Je peux t'aider à créer et valider des devis/factures BTP.",
                "todo": missing_fields[:4] if missing_fields else []
            }
        else:
            parsed = {
                "document": payload,
                "corrections": corrections,
                "missing_fields": missing_fields,
                "totals": totals,
            }
    elif intent == "chat" and "reply" not in parsed:
        # Corriger le format si nécessaire
        if "document" in parsed:
            doc = parsed.get("document", {})
            customer = doc.get("customer", {})
            reply_parts = []
            if customer.get("name"):
                reply_parts.append(f"Client: {customer.get('name')}")
            if missing_fields:
                reply_parts.append(f"Champs manquants: {', '.join(missing_fields[:4])}")
            parsed = {
                "reply": ". ".join(reply_parts) if reply_parts else "Formulaire en cours de saisie.",
                "todo": missing_fields[:4] if missing_fields else []
            }

    if intent == "validate":
        issues = list(section_issues or [])
        if not issues and missing_fields:
            issues = [f"Informations manquantes: {', '.join(missing_fields[:4])}"]
        if not isinstance(parsed, dict) or "reply" not in parsed:
            if issues:
                parsed = {"reply": "A corriger", "todo": issues[:4]}
            else:
                parsed = {"reply": "OK - Rien a corriger sur cette section.", "todo": []}
        else:
            if issues:
                parsed["todo"] = issues[:4]
                if "a corriger" not in str(parsed.get("reply", "")).lower():
                    parsed["reply"] = "A corriger"

    return {
        "messages": [AIMessage(content=content)],
        "output": parsed or content,
        "missing_fields": missing_fields,
        "corrections": corrections,
    }


# ==================== Nœud 6: Reflection ====================

def reflection_node(state: AgentState) -> AgentState:
    """L'agent réfléchit sur sa réponse et décide si continuer."""
    output = state.get("output", {})
    missing_fields = state.get("missing_fields") or []
    corrections = state.get("corrections") or []
    
    # Si on a des champs manquants ou des corrections, on pourrait continuer
    # Mais pour l'instant, on arrête toujours (peut être amélioré plus tard)
    should_continue = False
    
    return {
        "should_continue": should_continue,
        "reflection_reason": "Réponse complète" if not should_continue else "Données incomplètes",
    }


def should_continue(state: AgentState) -> str:
    """Décide si continuer ou arrêter."""
    if state.get("should_continue"):
        return "continue"
    return "end"


# ==================== Construction du Graph ====================

def build_graph():
    """Construit le graph LangGraph avec function calling."""
    builder = StateGraph(AgentState)
    
    # Ajouter les nœuds
    builder.add_node("input_normalizer", input_normalizer_node)
    builder.add_node("rag_retriever", rag_retriever_node)
    builder.add_node("business_tools", business_tools_node)
    builder.add_node("agent_reasoning", agent_reasoning_node)
    builder.add_node("tools", ToolNode(AVAILABLE_TOOLS))  # ✅ Nœud pour exécuter les outils
    builder.add_node("llm_synthesizer", llm_synthesizer_node)
    builder.add_node("reflection", reflection_node)
    
    # Définir le flux
    builder.set_entry_point("input_normalizer")
    builder.add_edge("input_normalizer", "rag_retriever")
    builder.add_edge("rag_retriever", "business_tools")
    builder.add_edge("business_tools", "agent_reasoning")
    
    # ✅ Branchement conditionnel : si le LLM appelle des outils, les exécuter
    builder.add_conditional_edges(
        "agent_reasoning",
        lambda state: "tools" if state.get("tool_calls") else "llm_synthesizer",
    )
    builder.add_edge("tools", "llm_synthesizer")  # Après exécution des outils, générer la réponse
    builder.add_edge("llm_synthesizer", "reflection")
    
    # ✅ Boucle de réflexion
    builder.add_conditional_edges(
        "reflection",
        should_continue,
        {
            "continue": "business_tools",  # Boucle pour réflexion
            "end": END
        }
    )
    
    return builder.compile(checkpointer=MemorySaver())


agent_graph = build_graph()


def invoke_agent(state: Dict[str, Any], thread_id: str = "default"):
    """Helper pour invoquer le graph avec mémoire LangGraph."""
    state_input: AgentState = dict(state)
    msgs = state_input.get("messages", [])
    if not msgs and state_input.get("input"):
        msgs = [HumanMessage(content=state_input["input"])]
    
    # Injecter system prompt si absent
    has_system = any(isinstance(m, SystemMessage) or getattr(m, "role", "") == "system" for m in msgs)
    if not has_system:
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
    
    state_input["messages"] = msgs
    
    return agent_graph.invoke(state_input, config={"configurable": {"thread_id": thread_id}})
