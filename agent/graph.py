"""Minimal LangGraph chat pipeline (router → tools → synthesizer).

Design goals:
- fast-path handled outside of this module
- optional RAG (only when needed)
- strict tool usage: at most 1 tool per request
- optimized prompts to reduce tokens

This module exposes:
- `chat_graph`: compiled 3-node LangGraph graph
- `prepare_state(...)`: runs router+tools (no final LLM)
- `synthesize(...)`: non-streaming final answer
- `stream_synthesize(...)`: token streaming for SSE/WebSocket
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, AsyncIterator, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .config import DEFAULT_MODEL, FALLBACK_MODEL, get_fast_llm
from .logging_config import logger
from .prompts import GRAPH_ROUTER_PROMPT, SYNTHESIZER_SYSTEM_PROMPT
from .rag_classifier import should_use_rag
from .rag.retriever import get_corps_metier_retriever, get_general_retriever, is_corps_metier_question
from .tools import AVAILABLE_TOOLS


try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore


class ToolCall(TypedDict):
    name: str
    args: dict[str, Any]


class ChatState(TypedDict, total=False):
    message: str
    history: list[dict[str, str]]
    metadata: dict[str, Any]
    user_role: str
    structured_payload: dict[str, Any] | None
    intent: Literal["chat", "validate", "analyze", "lookup"]
    use_rag: bool
    rag_filter_type: str | None
    rag_context: list[dict[str, Any]]
    tool_call: ToolCall | None
    tool_result: Any
    reply: str


_TOOLS_BY_NAME = {tool.name: tool for tool in AVAILABLE_TOOLS}
_ALLOWED_TOOL_NAMES = set(_TOOLS_BY_NAME.keys())

_LOOKUP_HINT_RE = re.compile(r"\b(client|clients|materiau|mat[eé]riaux|historique|prefill)\b", re.IGNORECASE)
_LOOKUP_ACTION_RE = re.compile(r"\b(trouve|trouver|cherche|chercher|recherche|rechercher|retrouve|retrouver|lookup)\b", re.IGNORECASE)
_TOTALS_HINT_RE = re.compile(r"\b(total|totaux|tva|ttc|ht)\b", re.IGNORECASE)
_CLEAN_HINT_RE = re.compile(r"\b(nettoi|d[eé]doubl|uniformi)\b", re.IGNORECASE)
_VALIDATE_HINT_RE = re.compile(r"\b(valid|corrig|conform)\b", re.IGNORECASE)
_ANALYZE_HINT_RE = re.compile(r"\b(analy|analyse|pdf|docx|document|fichier|pi[eè]ce jointe)\b", re.IGNORECASE)

_DEVIS_TERMS_HINT_RE = re.compile(
    r"\b(termes?|mots?|jargon|lexique|glossaire|clarif|expliq|d[eé]fin|d[eé]cortiq|comprendr)\b",
    re.IGNORECASE,
)
_DEVIS_CONTEXT_RE = re.compile(r"\b(devis|facture)\b", re.IGNORECASE)
_TERM_DEFINITION_RE = re.compile(
    r"\b(c['’]est quoi|ça veut dire|ca veut dire|qu['’]est-ce que|definition|définition)\b",
    re.IGNORECASE,
)
_BTP_TERMS_LIKELY_RE = re.compile(
    r"\b(acompte|tva|d[eé]cennale|rc\s*pro|dommages?-ouvrage|ipn|poutre|ragr[eé]age|chape|[eé]tanch[eé]it[eé]|consuel|plomberie|gros\s*œuvre|gros\s*oeuvre|d[eé]molition)\b",
    re.IGNORECASE,
)


def _should_show_devis_terms_ui(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False

    q_lower = q.lower()

    if _DEVIS_CONTEXT_RE.search(q) and _DEVIS_TERMS_HINT_RE.search(q):
        return True

    if _TERM_DEFINITION_RE.search(q) and _BTP_TERMS_LIKELY_RE.search(q):
        return True

    if "devis" in q_lower and ("explique" in q_lower or "clarifie" in q_lower):
        return True

    return False


def _build_devis_terms_ui_reply(query: str) -> str:
    payload_query = (query or "").strip()
    q_lower = payload_query.lower()
    if any(k in q_lower for k in ("termes", "mots", "jargon", "lexique", "glossaire")):
        payload_query = ""

    payload = {"query": payload_query}
    payload_json = json.dumps(payload, ensure_ascii=False)

    return (
        "Je vous aide à comprendre les termes techniques qu’on voit souvent sur un devis BTP.\n\n"
        "```devis-terms\n"
        f"{payload_json}\n"
        "```\n\n"
        "Si vous le souhaitez, vous pouvez aussi me copier/coller une ligne du devis (ou envoyer le PDF) "
        "et je vous l’explique poste par poste."
    )


def generate_response_with_actions(
    *,
    query: str,
    response_text: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Enrichit une réponse texte avec des actions rapides contextuelles."""
    query_lower = (query or "").lower()

    quick_actions: list[dict[str, str]] = []

    if _should_show_devis_terms_ui(query):
        response_text = _build_devis_terms_ui_reply(query)
        quick_actions.append(
            {
                "id": "devis_terms",
                "label": "Lexique du devis",
                "type": "devis",
                "icon": "📚",
            }
        )

    # Action 1 : Checklist diagnostic (problèmes / diagnostics)
    if any(
        keyword in query_lower
        for keyword in ("problème", "probleme", "panne", "fuite", "fissure", "défaut", "defaut", "casse", "diagnostic")
    ):
        quick_actions.append(
            {
                "id": "generate_checklist",
                "label": "Générer checklist diagnostic",
                "type": "diagnostic",
                "icon": "📋",
            }
        )

    # Action 2 : Mini-devis (estimation / prix)
    if any(keyword in query_lower for keyword in ("prix", "coût", "cout", "budget", "devis", "combien", "estim")):
        quick_actions.append(
            {
                "id": "create_estimate",
                "label": "Créer un mini-devis",
                "type": "pricing",
                "icon": "💰",
            }
        )

    # Action 3 : Liste matériaux (travaux / installation)
    if any(
        keyword in query_lower
        for keyword in ("matériau", "materiau", "matériaux", "materiaux", "refaire", "poser", "installer", "rénover", "renover")
    ):
        quick_actions.append(
            {
                "id": "materials_list",
                "label": "Liste matériaux + quantités",
                "type": "materials",
                "icon": "📊",
            }
        )

    # Action 4 : Guide photos (diagnostic)
    if any(keyword in query_lower for keyword in ("problème", "probleme", "fuite", "fissure", "diagnostic", "vérifier", "verifier")):
        quick_actions.append(
            {
                "id": "photo_guide",
                "label": "Que photographier ?",
                "type": "photos",
                "icon": "📸",
            }
        )

    # Fallback
    if not quick_actions:
        quick_actions.append(
            {
                "id": "generate_checklist",
                "label": "Organiser ce projet",
                "type": "general",
                "icon": "📋",
            }
        )

    return {
        "response": response_text,
        "quick_actions": quick_actions[:3],  # max 3 actions
        "metadata": metadata or {},
    }


def _maybe_parse_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except Exception:
        return None
    return None


def _infer_user_role(metadata: dict[str, Any] | None) -> str:
    if not isinstance(metadata, dict):
        return "particulier"
    raw = str(metadata.get("user_role") or metadata.get("role") or "").strip().lower()
    if raw in {"professionnel", "pro", "professional"}:
        return "professionnel"
    return "particulier"


def _structured_from_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None

    if isinstance(metadata.get("structured_payload"), dict):
        structured = dict(metadata["structured_payload"])
        structured.setdefault("doc_type", structured.get("doc_type", "quote"))
        return structured

    # Flat form fields (keep lightweight; only what tools need)
    has_any = any(
        k in metadata
        for k in (
            "client_name",
            "customer_name",
            "client_address",
            "client_contact",
            "supplier_name",
            "supplier_address",
            "supplier_contact",
            "line_items",
            "items",
            "doc_type",
            "docType",
        )
    )
    if not has_any:
        return None

    doc_type = str(metadata.get("doc_type") or metadata.get("docType") or "quote").strip() or "quote"
    line_items = metadata.get("line_items") or metadata.get("items") or []
    if not isinstance(line_items, list):
        line_items = []

    structured: dict[str, Any] = {
        "doc_type": doc_type,
        "customer": {
            "name": metadata.get("client_name") or metadata.get("customer_name") or "",
            "address": metadata.get("client_address") or "",
            "contact": metadata.get("client_contact") or "",
            "siret": metadata.get("client_siret") or None,
            "tva_number": metadata.get("client_tva") or None,
        },
        "supplier": {
            "name": metadata.get("supplier_name") or "",
            "address": metadata.get("supplier_address") or "",
            "contact": metadata.get("supplier_contact") or "",
            "siret": metadata.get("supplier_siret") or None,
            "tva_number": metadata.get("supplier_tva") or None,
        },
        "line_items": line_items,
    }
    if metadata.get("notes"):
        structured["notes"] = metadata["notes"]
    return structured


def _summarize_structured_payload(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    customer = payload.get("customer") if isinstance(payload.get("customer"), dict) else {}
    supplier = payload.get("supplier") if isinstance(payload.get("supplier"), dict) else {}
    items = payload.get("line_items") if isinstance(payload.get("line_items"), list) else []
    doc_type = str(payload.get("doc_type") or "quote")
    parts = [
        f"doc_type={doc_type}",
        f"customer={(customer.get('name') or '')}".strip(),
        f"supplier={(supplier.get('name') or '')}".strip(),
        f"line_items={len(items)}",
    ]
    return ", ".join(p for p in parts if p and not p.endswith("="))


def _extract_first_file(metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    files = metadata.get("files")
    if isinstance(files, list) and files:
        first = files[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return None


def _heuristic_tool_choice(message: str, metadata: dict[str, Any] | None, structured: dict[str, Any] | None) -> ToolCall | None:
    msg = (message or "").lower()
    mode = str((metadata or {}).get("mode") or "").lower() if isinstance(metadata, dict) else ""
    validate_section = bool((metadata or {}).get("validate_section")) if isinstance(metadata, dict) else False

    intent_validate = mode in {"validate", "validation"} or validate_section or bool(_VALIDATE_HINT_RE.search(msg))
    if intent_validate and isinstance(structured, dict) and "line_items" in structured:
        return {"name": "validate_devis_tool", "args": {"payload": structured}}

    if isinstance(structured, dict) and _TOTALS_HINT_RE.search(msg):
        return {
            "name": "calculate_totals_tool",
            "args": {"lines": structured.get("line_items") or [], "doc_type": structured.get("doc_type") or "quote"},
        }

    if isinstance(structured, dict) and _CLEAN_HINT_RE.search(msg):
        return {"name": "clean_lines_tool", "args": {"lines": structured.get("line_items") or []}}

    file_path = _extract_first_file(metadata)
    if file_path and _ANALYZE_HINT_RE.search(msg):
        doc_type = "auto"
        if isinstance(structured, dict):
            doc_type = str(structured.get("doc_type") or "auto")
        return {"name": "extract_pdf_tool", "args": {"file_path": file_path, "doc_type": doc_type}}

    if _LOOKUP_HINT_RE.search(msg):
        if not (_LOOKUP_ACTION_RE.search(msg) or ("historique" in msg) or ("prefill" in msg)):
            return None
        mode = "auto"
        if "materiau" in msg or "matériau" in msg:
            mode = "materials"
        elif "historique" in msg:
            mode = "history"
        elif "prefill" in msg or "pré-rempl" in msg:
            mode = "prefill"
        query = None
        if isinstance(structured, dict):
            customer = structured.get("customer") if isinstance(structured.get("customer"), dict) else {}
            query = customer.get("name") or None
        return {"name": "supabase_lookup_tool", "args": {"query": query, "mode": mode, "limit": 8}}

    return None


def _validate_tool_call(tool_call: ToolCall | None, *, metadata: dict[str, Any] | None, structured: dict[str, Any] | None) -> ToolCall | None:
    if not tool_call:
        return None
    name = tool_call.get("name")
    if not isinstance(name, str) or name not in _ALLOWED_TOOL_NAMES:
        return None
    args = tool_call.get("args")
    if not isinstance(args, dict):
        args = {}

    # Ensure required inputs exist (avoid calling tools with empty payloads).
    if name in {"validate_devis_tool"} and not isinstance(args.get("payload"), dict):
        if isinstance(structured, dict):
            args["payload"] = structured
        else:
            return None
    if name in {"calculate_totals_tool", "clean_lines_tool"} and not isinstance(args.get("lines"), list):
        if isinstance(structured, dict):
            args["lines"] = structured.get("line_items") or []
        else:
            return None
    if name == "extract_pdf_tool" and not isinstance(args.get("file_path"), str):
        file_path = _extract_first_file(metadata)
        if file_path:
            args["file_path"] = file_path
        else:
            return None
    if name == "supabase_lookup_tool":
        query = args.get("query")
        mode = str(args.get("mode") or "auto")
        # Allow empty query only for explicit prefill use-cases.
        if query is None or (isinstance(query, str) and not query.strip()):
            if mode != "prefill":
                return None
            args["query"] = None

    return {"name": name, "args": args}


async def router_node(state: ChatState) -> ChatState:
    """Decide optional RAG + a single tool call (or none)."""
    message = state.get("message") or ""
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    structured = state.get("structured_payload")
    if structured is None:
        structured = _structured_from_metadata(metadata)

    user_role = state.get("user_role") or _infer_user_role(metadata)

    # Heuristic first: avoid router LLM for common cases.
    heuristic_tool = _heuristic_tool_choice(message, metadata, structured)
    tool_call = _validate_tool_call(heuristic_tool, metadata=metadata, structured=structured)

    # Determine intent (lightweight)
    msg_lower = message.lower()
    intent: Literal["chat", "validate", "analyze", "lookup"] = "chat"
    if tool_call and tool_call["name"] in {"validate_devis_tool"}:
        intent = "validate"
    elif tool_call and tool_call["name"] in {"extract_pdf_tool"}:
        intent = "analyze"
    elif tool_call and tool_call["name"] in {"supabase_lookup_tool"}:
        intent = "lookup"
    elif _VALIDATE_HINT_RE.search(msg_lower):
        intent = "validate"

    use_rag = False
    rag_filter_type: str | None = None
    rag_context: list[dict[str, Any]] = []

    # ✅ Router métier: if the question is about a BTP trade, force the dedicated corpus retriever.
    if is_corps_metier_question(message):
        use_rag = True
        rag_filter_type = "corps_metier"

    # If heuristics didn't decide, ask the fast router LLM.
    # When the trade router forces `corps_metier` RAG, skip the router LLM (lower latency)
    # and avoid letting it override the forced RAG decision.
    if tool_call is None and rag_filter_type != "corps_metier":
        llm = get_fast_llm(temperature=0.0).bind(max_tokens=140)
        summary = _summarize_structured_payload(structured)
        file_path = _extract_first_file(metadata)
        try:
            raw = await llm.ainvoke(
                [
                    SystemMessage(content=GRAPH_ROUTER_PROMPT),
                    HumanMessage(
                        content=(
                            f"user_role={user_role}\n"
                            f"has_structured={bool(summary)}\n"
                            f"structured_summary={summary}\n"
                            f"has_file={bool(file_path)}\n"
                            f"question={message}"
                        )
                    ),
                ]
            )
            parsed = _maybe_parse_json(str(getattr(raw, "content", None) or str(raw)))
        except Exception as exc:
            logger.warning("graph router LLM failed: %s", exc)
            parsed = None

        if isinstance(parsed, dict):
            llm_intent = parsed.get("intent")
            if llm_intent in {"chat", "validate", "analyze", "lookup"}:
                intent = llm_intent
            suggested_tool = parsed.get("tool")
            tool_call = None
            if isinstance(suggested_tool, dict):
                tool_call = _validate_tool_call(
                    {"name": suggested_tool.get("name"), "args": suggested_tool.get("args") or {}},  # type: ignore[arg-type]
                    metadata=metadata,
                    structured=structured,
                )
            llm_use_rag = parsed.get("use_rag")
            if rag_filter_type is None:
                use_rag = bool(llm_use_rag) if isinstance(llm_use_rag, bool) else False

    # Final RAG gate via dedicated classifier (only for general RAG).
    if use_rag and rag_filter_type is None:
        try:
            use_rag = await should_use_rag(message, metadata=metadata)
        except Exception:
            use_rag = False

    # Retrieval (optional, strict)
    if use_rag:
        retriever = (
            get_corps_metier_retriever(k=int(os.getenv("RAG_TOP_K", "4")), score_threshold=float(os.getenv("RAG_THRESHOLD", "0.75")))
            if rag_filter_type == "corps_metier"
            else get_general_retriever(k=int(os.getenv("RAG_TOP_K", "4")), score_threshold=float(os.getenv("RAG_THRESHOLD", "0.75")))
        )

        if retriever is not None:
            try:
                docs = await asyncio.to_thread(retriever.invoke, message)
            except Exception:
                docs = []

            rag_context = []
            for doc in docs or []:
                content = getattr(doc, "page_content", None)
                meta = getattr(doc, "metadata", None)
                if isinstance(content, str) and content.strip():
                    rag_context.append(
                        {
                            "content": content,
                            "metadata": meta if isinstance(meta, dict) else {},
                            "score": None,
                        }
                    )

            # Truncate to keep prompts small
            for item in rag_context:
                content = item.get("content")
                if isinstance(content, str) and len(content) > 900:
                    item["content"] = content[:900] + "…"

    return {
        "user_role": user_role,
        "structured_payload": structured,
        "intent": intent,
        "tool_call": tool_call,
        "use_rag": use_rag,
        "rag_filter_type": rag_filter_type,
        "rag_context": rag_context,
    }


async def tools_node(state: ChatState) -> ChatState:
    """Execute at most one tool call."""
    tool_call = state.get("tool_call")
    if not tool_call:
        return {"tool_result": None}

    name = tool_call.get("name")
    tool = _TOOLS_BY_NAME.get(name) if isinstance(name, str) else None
    if not tool:
        return {"tool_result": {"error": "unknown_tool", "tool": name}}

    args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
    try:
        result = await asyncio.to_thread(tool.invoke, args)  # tools are sync; keep event-loop free
        return {"tool_result": result}
    except Exception as exc:
        logger.warning("tool %s failed: %s", name, exc)
        return {"tool_result": {"error": str(exc), "tool": name}}


def _build_messages_for_synthesis(state: ChatState) -> list[BaseMessage]:
    """Build minimal messages for the final LLM call."""
    message = state.get("message") or ""
    history = state.get("history") if isinstance(state.get("history"), list) else []

    messages: list[BaseMessage] = [SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT)]

    structured_summary = _summarize_structured_payload(state.get("structured_payload"))
    tool_call = state.get("tool_call") or None
    tool_result = state.get("tool_result")
    rag_context = state.get("rag_context") if isinstance(state.get("rag_context"), list) else []

    context_lines: list[str] = []
    if structured_summary:
        context_lines.append(f"Contexte devis/facture: {structured_summary}")
    if tool_call and tool_result is not None:
        try:
            compact = json.dumps(tool_result, ensure_ascii=False)
        except Exception:
            compact = str(tool_result)
        if len(compact) > 1200:
            compact = compact[:1200] + "…"
        context_lines.append(f"tool={tool_call.get('name')}: {compact}")
    if rag_context:
        snippets: list[str] = []
        for doc in rag_context[:4]:
            content = doc.get("content")
            if isinstance(content, str) and content.strip():
                snippet = content.strip().replace("\n", " ")
                if len(snippet) > 350:
                    snippet = snippet[:350] + "…"
                snippets.append(f"- {snippet}")
        if snippets:
            context_lines.append("Extraits pertinents:\n" + "\n".join(snippets))

    if context_lines:
        messages.append(SystemMessage(content="\n".join(context_lines)))

    # Keep only last 6 turns to reduce tokens.
    for item in history[-6:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    messages.append(HumanMessage(content=message))
    return messages


def _get_synth_llm(streaming: bool):
    if ChatOpenAI is None:  # pragma: no cover
        raise RuntimeError("langchain_openai is required for this pipeline.")

    model = os.getenv("LLM_PIPELINE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    fallback_model = os.getenv("LLM_PIPELINE_FALLBACK_MODEL", FALLBACK_MODEL).strip() or FALLBACK_MODEL
    # Completion token budget for the final answer.
    # Default raised to avoid truncated structured answers (finish_reason="length").
    max_tokens = int(os.getenv("LLM_PIPELINE_MAX_TOKENS", "1500"))
    max_tokens = max(128, min(max_tokens, 2048))

    def is_reasoning_model(name: str) -> bool:
        lowered = (name or "").lower()
        return lowered.startswith("gpt-5") or lowered.startswith("o1") or lowered.startswith("o3")

    reasoning_effort = None
    if is_reasoning_model(model):
        reasoning_effort = os.getenv("LLM_PIPELINE_REASONING_EFFORT", "minimal").strip() or "minimal"

    primary_kwargs: dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "streaming": streaming,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        primary_kwargs["reasoning_effort"] = reasoning_effort

    fallback_kwargs: dict[str, Any] = {
        "model": fallback_model,
        "temperature": 0.0,
        "streaming": streaming,
        "max_tokens": max_tokens,
    }
    if is_reasoning_model(fallback_model):
        fallback_kwargs["reasoning_effort"] = os.getenv("LLM_PIPELINE_REASONING_EFFORT", "minimal").strip() or "minimal"

    primary = ChatOpenAI(**primary_kwargs)
    fallback = ChatOpenAI(**fallback_kwargs)
    return primary.with_fallbacks([fallback])


def _log_llm_diagnostics(*, raw: Any, reply: str, label: str) -> None:
    """Debug token usage / finish_reason to diagnose truncation."""
    if os.getenv("LLM_DEBUG_TOKENS", "0").strip() != "1":
        return

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None

    usage_metadata = getattr(raw, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        prompt_tokens = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens")
        completion_tokens = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens")
        total_tokens = usage_metadata.get("total_tokens")

    response_metadata = getattr(raw, "response_metadata", None)
    if isinstance(response_metadata, dict):
        finish_reason = response_metadata.get("finish_reason") or finish_reason
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            prompt_tokens = token_usage.get("prompt_tokens") or prompt_tokens
            completion_tokens = token_usage.get("completion_tokens") or completion_tokens
            total_tokens = token_usage.get("total_tokens") or total_tokens

    if prompt_tokens is not None and completion_tokens is not None and total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens

    logger.info("=" * 80)
    logger.info("TOKEN USAGE (%s)", label)
    logger.info("Prompt tokens: %s", prompt_tokens if prompt_tokens is not None else "n/a")
    logger.info("Completion tokens: %s", completion_tokens if completion_tokens is not None else "n/a")
    logger.info("Total tokens: %s", total_tokens if total_tokens is not None else "n/a")
    logger.info("Finish reason: %s", finish_reason if finish_reason else "n/a")
    if finish_reason == "length":
        logger.warning("RESPONSE TRUNCATED: finish_reason=length (increase LLM_PIPELINE_MAX_TOKENS).")
    elif finish_reason == "stop":
        logger.info("Response complete: finish_reason=stop.")
    if reply:
        logger.info("Reply tail: ...%s", reply[-200:])
    logger.info("=" * 80)


def _log_llm_request_config(*, messages: list[BaseMessage], state: ChatState, label: str) -> None:
    """Debug request configuration before calling the LLM."""
    if os.getenv("LLM_DEBUG_TOKENS", "0").strip() != "1":
        return

    model_name = os.getenv("LLM_PIPELINE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    raw_env_max = os.getenv("LLM_PIPELINE_MAX_TOKENS")
    try:
        max_tokens = int(raw_env_max or "1500")
    except Exception:
        max_tokens = 1500
    max_tokens = max(128, min(max_tokens, 2048))

    system_len = len(SYNTHESIZER_SYSTEM_PROMPT or "")
    total_chars = 0
    context_chars = 0
    for idx, m in enumerate(messages):
        content = _coerce_text_content(getattr(m, "content", None))
        total_chars += len(content)
        if idx > 0 and isinstance(m, SystemMessage):
            context_chars += len(content)

    tool_call = state.get("tool_call")
    tool_name = tool_call.get("name") if isinstance(tool_call, dict) else None

    logger.info("=" * 80)
    logger.info("DEBUG LLM CONFIG (%s)", label)
    logger.info("Model: %s", model_name)
    logger.info("Max tokens from env: %s", raw_env_max if raw_env_max is not None else "NOT SET")
    logger.info("Max tokens effective: %s", max_tokens)
    logger.info("Messages: %d", len(messages))
    logger.info("System prompt length: %d chars", system_len)
    logger.info("Context (RAG/tool/structured) length: %d chars", context_chars)
    logger.info("All messages length: %d chars", total_chars)
    logger.info(
        "Routing: intent=%s use_rag=%s rag_filter_type=%s rag_docs=%s tool=%s",
        state.get("intent"),
        state.get("use_rag"),
        state.get("rag_filter_type"),
        len(state.get("rag_context") or []) if isinstance(state.get("rag_context"), list) else 0,
        tool_name,
    )
    logger.info("=" * 80)


def _coerce_text_content(content: Any) -> str:
    """Extract visible text from content which can be str or multi-part."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    value = part.get("content")
                    if isinstance(value, str):
                        parts.append(value)
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


async def synthesizer_node(state: ChatState) -> ChatState:
    """Final answer (non-streaming)."""
    llm = _get_synth_llm(streaming=False)
    messages = _build_messages_for_synthesis(state)
    _log_llm_request_config(messages=messages, state=state, label="synthesizer")
    try:
        raw = await llm.ainvoke(messages)
    except Exception as exc:
        logger.error("synthesizer failed: %s", exc)
        return {"reply": "Désolé, une erreur s'est produite. Peux-tu réessayer ?"}

    reply = _coerce_text_content(getattr(raw, "content", None)).strip()
    if reply:
        _log_llm_diagnostics(raw=raw, reply=reply, label="synthesizer")
        return {"reply": reply}

    # Some reasoning models can consume the whole budget in hidden reasoning and return empty content.
    # In that case, force a non-reasoning fallback (fast to return a visible answer).
    fallback_model = os.getenv("LLM_PIPELINE_FALLBACK_MODEL", FALLBACK_MODEL).strip() or FALLBACK_MODEL
    if ChatOpenAI is not None:
        try:
            fallback = ChatOpenAI(model=fallback_model, temperature=0.0, streaming=False, max_tokens=320)
            raw2 = await fallback.ainvoke(messages)
            reply2 = _coerce_text_content(getattr(raw2, "content", None)).strip()
            if reply2:
                _log_llm_diagnostics(raw=raw2, reply=reply2, label="synthesizer_fallback_visible")
                return {"reply": reply2}
        except Exception:
            pass

    return {"reply": "Je n'ai pas réussi à générer une réponse. Peux-tu reformuler en 1 phrase ?"}


async def prepare_state(
    *,
    message: str,
    history: list[dict[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ChatState:
    """Run router + tools; returns state ready for synthesis."""
    base: ChatState = {
        "message": message,
        "history": history or [],
        "metadata": metadata or {},
        "user_role": _infer_user_role(metadata),
        "structured_payload": _structured_from_metadata(metadata),
    }
    routed = await router_node(base)
    tool_state = await tools_node({**base, **routed})
    return {**base, **routed, **tool_state}


async def synthesize(state: ChatState) -> str:
    result = await synthesizer_node(state)
    return (result.get("reply") or "").strip()


async def stream_synthesize(state: ChatState) -> AsyncIterator[str]:
    """Stream the final answer tokens (for SSE/WebSocket)."""
    llm = _get_synth_llm(streaming=True)
    messages = _build_messages_for_synthesis(state)

    finish_reason: str | None = None
    tail = ""
    async for chunk in llm.astream(messages):
        response_metadata = getattr(chunk, "response_metadata", None)
        if isinstance(response_metadata, dict) and response_metadata.get("finish_reason"):
            finish_reason = str(response_metadata["finish_reason"])
        token = _coerce_text_content(getattr(chunk, "content", None))
        if token:
            tail = (tail + token)[-200:]
            yield token

    if os.getenv("LLM_DEBUG_TOKENS", "0").strip() == "1":
        logger.info("Stream finish reason: %s", finish_reason if finish_reason else "n/a")
        if tail:
            logger.info("Stream reply tail: ...%s", tail)


def build_graph():
    """3 nodes only: router -> tools -> synthesizer."""
    builder = StateGraph(ChatState)
    builder.add_node("router", router_node)
    builder.add_node("tools", tools_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.set_entry_point("router")
    builder.add_edge("router", "tools")
    builder.add_edge("tools", "synthesizer")
    builder.add_edge("synthesizer", END)
    return builder.compile()


chat_graph = build_graph()
