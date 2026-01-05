import json
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .config import SYSTEM_PROMPT, get_llm
from .rag import SupabaseRAG
from .tools import (
    calculate_totals_tool,
    clean_lines_tool,
    extract_pdf_tool,
    supabase_lookup_tool,
    validate_devis_tool,
)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _read_prompt(name: str) -> str:
    base = PROMPTS_DIR / name
    if base.exists():
        return base.read_text(encoding="utf-8")
    txt = base.with_suffix(".txt")
    if txt.exists():
        return txt.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt {name} introuvable dans {PROMPTS_DIR}")


def _maybe_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


class NormalizedOutput(BaseModel):
    intent: Literal["analyze", "prepare_devis", "validate", "chat", "prefill", "unknown"]
    doc_type: Literal["quote", "invoice", "unknown"] = "quote"
    summary: Optional[str] = None
    structured_payload: Dict[str, Any] = Field(default_factory=dict)
    line_items: List[dict] = Field(default_factory=list)
    files: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)


class RagQuery(BaseModel):
    queries: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    intent: str
    normalized: dict
    rag_context: list
    supabase_context: list
    tool_results: dict
    totals: dict
    corrections: list
    last_quote: dict | None
    files: list
    output: Any


rag_client = SupabaseRAG()


def _build_prompt(name: str) -> ChatPromptTemplate:
    content = f"{{% raw %}}{SYSTEM_PROMPT}\n\n{_read_prompt(name)}{{% endraw %}}"
    return ChatPromptTemplate.from_template(content, template_format="jinja2")


def input_normalizer_node(state: AgentState) -> AgentState:
    """Detecte l'intention et normalise l'entree utilisateur."""
    # Bypass LLM si on a deja une normalisation fournie (ex: analyse/prepare avec payload).
    existing_norm = state.get("normalized") or {}
    existing_struct = existing_norm.get("structured_payload") if isinstance(existing_norm, dict) else {}
    if existing_struct:
        files = list({*state.get("files", []), *existing_norm.get("files", [])}) if isinstance(existing_norm, dict) else state.get("files", [])
        return {
            "intent": existing_norm.get("intent") or state.get("intent") or "chat",
            "normalized": existing_norm,
            "files": files,
        }

    last_user_msg = ""
    if state.get("messages"):
        last = state["messages"][-1]
        last_user_msg = last.content if hasattr(last, "content") else str(last)

    prompt = _build_prompt("analysis_prompt")
    formatted = prompt.format_messages(
        user_input=last_user_msg,
        previous_payload=json.dumps(state.get("normalized") or {}),
    )
    reply = get_llm().invoke(formatted)
    content = getattr(reply, "content", None) or str(reply)
    parsed = _maybe_parse_json(content) or {}
    normalized = parsed if isinstance(parsed, dict) else {}

    # Valeurs par defaut pour eviter les KeyError
    normalized.setdefault("intent", state.get("intent") or "chat")
    normalized.setdefault("doc_type", "quote")
    normalized.setdefault("structured_payload", {})
    normalized.setdefault("line_items", [])
    normalized.setdefault("files", [])
    normalized.setdefault("missing_fields", [])
    # Heuristique: si l'utilisateur parle de creer/preparer un devis ou une facture, orienter vers prepare_devis.
    text_lower = last_user_msg.lower()
    wants_quote = "devis" in text_lower
    wants_invoice = "facture" in text_lower
    wants_prepare = any(term in text_lower for term in ["prepare", "faire", "redige", "realise"])
    if normalized.get("intent") in ("chat", "unknown") and (wants_quote or wants_invoice or wants_prepare):
        normalized["intent"] = "prepare_devis"
        normalized["doc_type"] = "invoice" if wants_invoice else "quote"
    payload = normalized.get("structured_payload") or {}
    payload.setdefault("doc_type", normalized.get("doc_type", "quote"))
    payload.setdefault("line_items", normalized.get("line_items", []))
    normalized["structured_payload"] = payload
    normalized["line_items"] = payload.get("line_items", [])
    normalized.setdefault("summary", normalized.get("summary") or last_user_msg[:280])
    files = list({*state.get("files", []), *normalized.get("files", [])})

    return {
        "intent": normalized.get("intent") or "chat",
        "normalized": normalized,
        "files": files,
    }


def rag_retriever_node(state: AgentState) -> AgentState:
    """Recupere du contexte RAG depuis SupabaseVectorStore."""
    normalized = state.get("normalized") or {}
    payload = normalized.get("structured_payload", {})
    query = normalized.get("summary") or payload.get("project_label") or payload.get("notes") or ""
    rag_results = []
    filters: Dict[str, Any] = {}

    if query:
        try:
            rag_prompt = _build_prompt("rag_prompt")
            rag_plan = get_llm().with_structured_output(RagQuery, method="function_calling").invoke(
                rag_prompt.format_messages(intent=normalized.get("intent") or state.get("intent"), normalized_payload=payload, user_query=query)
            )
            queries = rag_plan.queries if hasattr(rag_plan, "queries") else []
            filters = rag_plan.filters if hasattr(rag_plan, "filters") else {}
        except Exception:
            queries = [query]
        for q in queries or [query]:
            rag_results.extend(rag_client.retrieve(q))

    # Filtre doc_type si fourni
    doc_type_filter = filters.get("doc_type") if isinstance(filters, dict) else None
    if doc_type_filter and doc_type_filter in {"quote", "invoice"}:
        rag_results = [r for r in rag_results if (r.get("metadata") or {}).get("doc_type") in (doc_type_filter, None)]

    return {
        "rag_context": rag_results,
        "supabase_context": rag_results,
    }


def business_tools_node(state: AgentState) -> AgentState:
    """Execute les outils metier (calculs, nettoyage, validations, lookup Supabase)."""
    normalized = state.get("normalized") or {}
    payload = dict(normalized.get("structured_payload") or {})
    line_items = payload.get("line_items") or normalized.get("line_items") or []

    cleaned = clean_lines_tool.invoke({"lines": line_items, "default_vat_rate": payload.get("vat_rate")})
    totals = calculate_totals_tool.invoke({"lines": cleaned.get("lines", []), "default_vat_rate": payload.get("vat_rate")})

    payload["line_items"] = cleaned.get("lines", [])
    payload.setdefault("doc_type", normalized.get("doc_type", "quote"))

    validation = validate_devis_tool.invoke({"payload": payload})

    supabase_prefill = supabase_lookup_tool.invoke(
        {"query": (payload.get("customer") or {}).get("name"), "mode": "auto"}
    )
    # Auto-prefill client depuis Supabase si disponible
    try:
        results = supabase_prefill.get("results") if isinstance(supabase_prefill, dict) else {}
        candidates = results.get("clients") if isinstance(results, dict) else None
        if candidates and isinstance(candidates, list):
            first = candidates[0]
            customer = payload.get("customer") or {}
            if not customer.get("name") and first.get("name"):
                customer["name"] = first.get("name")
            if not customer.get("address") and first.get("address"):
                customer["address"] = first.get("address")
            if not customer.get("contact") and first.get("contact"):
                customer["contact"] = first.get("contact")
            payload["customer"] = customer
    except Exception:
        pass

    extracted_files = []
    text_blobs = []
    for file_path in state.get("files", []):
        extraction = extract_pdf_tool.invoke({"file_path": file_path, "doc_type": payload.get("doc_type", "quote")})
        extracted_files.append(extraction)
        if isinstance(extraction, dict) and extraction.get("parsed_text"):
            text_blobs.append(extraction["parsed_text"])
    if text_blobs and not payload.get("raw_text"):
        payload["raw_text"] = "\n\n".join(text_blobs)

    # Champs manquants (basique) pour guider le chat unique
    missing_fields = []
    customer = payload.get("customer") or {}
    if not customer.get("name"):
        missing_fields.append("customer.name")
    if not customer.get("address"):
        missing_fields.append("customer.address")
    if not payload.get("payment_terms"):
        missing_fields.append("payment_terms")
    if payload.get("doc_type") == "invoice":
        if not payload.get("penalties_late_payment"):
            missing_fields.append("penalties_late_payment")
        if not payload.get("professional_liability"):
            missing_fields.append("professional_liability")
    if not payload.get("line_items"):
        missing_fields.append("line_items")

    return {
        "tool_results": {
            "clean_lines": cleaned,
            "totals": totals,
            "validation": validation,
            "supabase": supabase_prefill,
            "files": extracted_files,
        },
        "totals": totals.get("totals", {}),
        "corrections": validation.get("issues", []),
        "supabase_context": supabase_prefill.get("results", {}),
        "normalized": normalized | {"structured_payload": payload, "missing_fields": missing_fields},
        "files": state.get("files", []),
        "last_quote": payload if payload.get("doc_type") == "quote" else state.get("last_quote"),
        "missing_fields": missing_fields,
    }


def llm_synthesizer_node(state: AgentState) -> AgentState:
    """Genere la reponse finale (JSON strict ou texte)."""
    intent = state.get("intent") or (state.get("normalized") or {}).get("intent") or "chat"
    normalized = state.get("normalized") or {}
    payload = dict(normalized.get("structured_payload") or {})
    payload.setdefault("line_items", payload.get("line_items") or (state.get("tool_results") or {}).get("clean_lines", {}).get("lines", []))
    totals = state.get("totals") or {}

    rag_context = state.get("rag_context") or []
    supabase_context = state.get("supabase_context") or []
    corrections = state.get("corrections") or []
    missing_fields = state.get("missing_fields") or normalized.get("missing_fields") or []

    prompt_name = "prepare_devis_prompt" if intent in ("prepare_devis", "analyze") else "validate_prompt" if intent == "validate" else "chat_prompt"
    prompt = _build_prompt(prompt_name)

    chain = prompt | get_llm()
    result = chain.invoke(
        {
            "intent": intent,
            "normalized_payload": payload,
            "totals": totals,
            "corrections": corrections,
            "rag_context": rag_context,
            "supabase_context": supabase_context,
            "missing_fields": missing_fields,
        }
    )
    content = getattr(result, "content", None) or str(result)
    parsed = _maybe_parse_json(content)
    ai_message = AIMessage(content=content)

    return {
        "messages": [ai_message],
        "output": parsed or content,
        "last_quote": payload if payload.get("doc_type") == "quote" else state.get("last_quote"),
        "corrections": corrections,
        "missing_fields": missing_fields,
    }


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("input_normalizer", input_normalizer_node)
    builder.add_node("rag_retriever", rag_retriever_node)
    builder.add_node("business_tools", business_tools_node)
    builder.add_node("llm_synthesizer", llm_synthesizer_node)

    builder.set_entry_point("input_normalizer")
    builder.add_edge("input_normalizer", "rag_retriever")
    builder.add_edge("rag_retriever", "business_tools")
    builder.add_edge("business_tools", "llm_synthesizer")
    builder.add_edge("llm_synthesizer", END)

    return builder.compile(checkpointer=MemorySaver())


agent_graph = build_graph()


def invoke_agent(state: Dict[str, Any], thread_id: str = "default"):
    """Helper pour invoquer le graph avec memoire LangGraph."""
    state_input: AgentState = dict(state)
    msgs = state_input.get("messages", [])
    if not msgs and state_input.get("input"):
        msgs = [HumanMessage(content=state_input["input"])]
    # Inject system prompt au debut si absent
    has_system = any(isinstance(m, SystemMessage) or getattr(m, "role", "") == "system" for m in msgs)
    if not has_system:
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
    state_input["messages"] = msgs

    return agent_graph.invoke(state_input, config={"configurable": {"thread_id": thread_id}})
