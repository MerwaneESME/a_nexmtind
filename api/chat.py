"""Streaming /chat endpoint for the optimized NEXTMIND pipeline.

Supports:
- JSON response (legacy-compatible)
- SSE streaming when requested (Accept: text/event-stream or ?stream=true or payload.stream=true)
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from agent.cache import CacheEntry, get_chat_cache, normalize_question
from agent.fast_path import try_fast_path
from agent.graph import generate_response_with_actions, prepare_state, stream_synthesize, synthesize
from agent.logging_config import logger
from agent.monitoring import track_request, track_cache_hit
from agent.utils.conversation_store import get_conversation_store

# Rate limiting (imported from parent app)
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    limiter = Limiter(key_func=get_remote_address)
except ImportError:
    limiter = None

router = APIRouter()


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatInput(BaseModel):
    message: str
    thread_id: Optional[str] = None
    conversation_id: Optional[str] = None
    history: Optional[List[ChatHistoryItem]] = None
    metadata: Optional[Dict[str, Any]] = None
    force_prepare: Optional[bool] = None
    stream: Optional[bool] = None
    clear_cache: Optional[bool] = None
    clear_history: Optional[bool] = None


def _wants_stream(payload: ChatInput, request: Request) -> bool:
    if payload.stream is True:
        return True
    q = (request.query_params.get("stream") or "").strip().lower()
    if q in {"1", "true", "yes", "on"}:
        return True
    accept = (request.headers.get("accept") or "").lower()
    return "text/event-stream" in accept


def _wants_clear_cache(payload: ChatInput, request: Request) -> bool:
    if payload.clear_cache is True:
        return True
    q = (request.query_params.get("clear_cache") or "").strip().lower()
    return q in {"1", "true", "yes", "on"}


def _sse(event: str, data: Any) -> str:
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False)
    lines = str(data).splitlines() or [""]
    out = f"event: {event}\n"
    for line in lines:
        out += f"data: {line}\n"
    return out + "\n"


def _is_cacheable(metadata: dict[str, Any] | None, history: list[dict[str, str]] | None) -> bool:
    """Cache only stateless questions (avoid mixing user-specific form data)."""
    if history:
        return False
    if not isinstance(metadata, dict) or not metadata:
        return True
    if isinstance(metadata.get("structured_payload"), dict):
        return False
    if any(k in metadata for k in ("line_items", "items", "client_name", "customer_name", "files", "validate_section")):
        return False
    return True


def _tool_to_response_extras(state: dict[str, Any]) -> tuple[dict, list, dict]:
    tool_call = state.get("tool_call") if isinstance(state.get("tool_call"), dict) else None
    tool_result = state.get("tool_result")
    corrections: list = []
    totals: dict = {}

    if tool_call and isinstance(tool_result, dict):
        name = tool_call.get("name")
        if name == "validate_devis_tool":
            corrections = tool_result.get("issues") or []
            totals = tool_result.get("totals") or {}
        elif name == "calculate_totals_tool":
            corrections = tool_result.get("issues") or []
            totals = tool_result.get("totals") or {}

    return tool_result or {}, corrections, totals


def _is_context_dependent(query: str, history: list[dict[str, Any]]) -> bool:
    if not history:
        return False

    q = (query or "").strip().lower()
    if not q:
        return False

    context_keywords = (
        "aussi",
        "et",
        "en plus",
        "pareil",
        "même chose",
        "meme chose",
        "combien en tout",
        "total",
        "au final",
        "pour ça",
        "pour ca",
        "pour cela",
        "dans ce cas",
        "et pour",
        "et du coup",
        "pour le plafond",
        "pour le mur",
        "pour la toiture",
        "il",
        "elle",
        "ça",
        "ca",
        "cela",
        "eux",
    )

    # Questions courtes + pronoms / références => probablement contextuelles
    if len(q) < 40 and any(k in q for k in context_keywords):
        return True

    # "Et pour X ?" / "Et X ?" => contextuel si on a déjà une discussion
    if q.startswith("et ") or q.startswith("et pour"):
        return True

    return False


@track_request("chat")
async def handle_chat_non_stream(payload: ChatInput) -> dict[str, Any]:
    cache = get_chat_cache()
    normalized = normalize_question(payload.message)
    store = get_conversation_store()
    conversation_id = (payload.conversation_id or payload.thread_id or "").strip() or None

    if store.enabled and conversation_id and payload.clear_history is True:
        await store.clear_history(conversation_id=conversation_id)

    history_dump = [item.model_dump() for item in (payload.history or [])]
    stored_history: list[dict[str, Any]] = []
    if store.enabled and conversation_id and not history_dump:
        stored_history = await store.get_history(conversation_id=conversation_id, limit=10)
        if _is_context_dependent(payload.message, stored_history):
            history_dump = [{"role": m.get("role", ""), "content": m.get("content", "")} for m in stored_history]

    cacheable = _is_cacheable(payload.metadata, history_dump) and not bool(stored_history)
    # Clear cache for this question when requested (dev/test).
    # Note: we still allow caching of the fresh response afterward.
    if cache.enabled and cacheable and payload.clear_cache is True:
        await cache.delete(normalized)

    if cache.enabled and cacheable:
        hit = await cache.get(normalized)
        if hit:
            track_cache_hit(True)
            meta = hit.meta or {}
            enriched = generate_response_with_actions(query=payload.message, response_text=hit.reply, metadata={"route": meta.get("route") or "cache"})
            return {
                "reply": enriched["response"],
                "quick_actions": enriched["quick_actions"],
                "conversation_id": conversation_id,
                "raw_output": {"cache": "hit", "route": meta.get("route")},
                "corrections": [],
                "totals": {},
                "missing_fields": [],
            }
        else:
            track_cache_hit(False)

    user_role = str((payload.metadata or {}).get("user_role") or "")
    fast = await try_fast_path(
        payload.message,
        metadata=payload.metadata,
        user_role=user_role,
        history=history_dump,
    )
    if fast:
        logger.info("=" * 80)
        logger.info("🔍 ROUTING DECISION | route=fast | clear_cache=%s", bool(payload.clear_cache))
        logger.info("Query: %s", (payload.message or "")[:120])
        logger.info("Fast answer: %s", fast[:120])
        logger.info("=" * 80)
        if cache.enabled and cacheable:
            await cache.set(normalized, CacheEntry(reply=fast, meta={"route": "fast"}))
        enriched = generate_response_with_actions(query=payload.message, response_text=fast, metadata={"route": "fast"})
        if store.enabled and conversation_id:
            await store.add_message(conversation_id=conversation_id, role="user", content=payload.message, metadata={})
            await store.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=enriched["response"],
                metadata={"route": "fast", "quick_actions": enriched["quick_actions"]},
            )
        return {
            "reply": enriched["response"],
            "quick_actions": enriched["quick_actions"],
            "conversation_id": conversation_id,
            "raw_output": {"route": "fast"},
            "corrections": [],
            "totals": {},
            "missing_fields": [],
        }

    state = await prepare_state(
        message=payload.message,
        history=history_dump,
        metadata=payload.metadata or {},
    )
    reply = await synthesize(state)
    tool_result, corrections, totals = _tool_to_response_extras(state)
    enriched = generate_response_with_actions(
        query=payload.message,
        response_text=reply,
        metadata={
            "route": "full",
            "rag_used": bool(state.get("rag_context")),
            "tool_used": (state.get("tool_call") or {}).get("name") if isinstance(state.get("tool_call"), dict) else None,
            "intent": state.get("intent"),
        },
    )

    logger.info("=" * 80)
    logger.info("🔍 ROUTING DECISION | route=full | clear_cache=%s", bool(payload.clear_cache))
    logger.info("Query: %s", (payload.message or "")[:120])
    logger.info("rag_used=%s tool=%s intent=%s", bool(state.get("rag_context")), state.get("tool_call"), state.get("intent"))
    logger.info("=" * 80)

    if cache.enabled and cacheable and reply:
        await cache.set(normalized, CacheEntry(reply=reply, meta={"route": "full"}))

    if store.enabled and conversation_id:
        await store.add_message(conversation_id=conversation_id, role="user", content=payload.message, metadata={})
        await store.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=enriched["response"],
            metadata={
                "route": "full",
                "rag_used": bool(state.get("rag_context")),
                "tool": state.get("tool_call"),
                "quick_actions": enriched["quick_actions"],
            },
        )

    return {
        "reply": enriched["response"],
        "quick_actions": enriched["quick_actions"],
        "conversation_id": conversation_id,
        "raw_output": {
            "route": "full",
            "intent": state.get("intent"),
            "tool": state.get("tool_call"),
            "tool_result": tool_result,
            "rag_used": bool(state.get("rag_context")),
        },
        "corrections": corrections,
        "totals": totals,
        "missing_fields": [],
    }


@track_request("chat_stream")
async def handle_chat_stream(payload: ChatInput) -> AsyncIterator[str]:
    cache = get_chat_cache()
    normalized = normalize_question(payload.message)
    store = get_conversation_store()
    conversation_id = (payload.conversation_id or payload.thread_id or "").strip() or None

    if store.enabled and conversation_id and payload.clear_history is True:
        await store.clear_history(conversation_id=conversation_id)

    history_dump = [item.model_dump() for item in (payload.history or [])]
    stored_history: list[dict[str, Any]] = []
    if store.enabled and conversation_id and not history_dump:
        stored_history = await store.get_history(conversation_id=conversation_id, limit=10)
        if _is_context_dependent(payload.message, stored_history):
            history_dump = [{"role": m.get("role", ""), "content": m.get("content", "")} for m in stored_history]

    cacheable = _is_cacheable(payload.metadata, history_dump) and not bool(stored_history)

    if cache.enabled and cacheable and payload.clear_cache is True:
        await cache.delete(normalized)

    if cache.enabled and cacheable:
        hit = await cache.get(normalized)
        if hit:
            track_cache_hit(True)
            enriched = generate_response_with_actions(query=payload.message, response_text=hit.reply, metadata={"route": "cache"})
            yield _sse("meta", {"cache": "hit"})
            yield _sse("delta", enriched["response"])
            yield _sse(
                "done",
                {"reply": enriched["response"], "quick_actions": enriched["quick_actions"], "conversation_id": conversation_id},
            )
            return
        else:
            track_cache_hit(False)

    user_role = str((payload.metadata or {}).get("user_role") or "")
    fast = await try_fast_path(
        payload.message,
        metadata=payload.metadata,
        user_role=user_role,
        history=history_dump,
    )
    if fast:
        logger.info("=" * 80)
        logger.info("🔍 ROUTING DECISION | route=fast(stream) | clear_cache=%s", bool(payload.clear_cache))
        logger.info("Query: %s", (payload.message or "")[:120])
        logger.info("Fast answer: %s", fast[:120])
        logger.info("=" * 80)
        if cache.enabled and cacheable:
            await cache.set(normalized, CacheEntry(reply=fast, meta={"route": "fast"}))
        enriched = generate_response_with_actions(query=payload.message, response_text=fast, metadata={"route": "fast"})
        yield _sse("meta", {"cache": "miss", "route": "fast"})
        yield _sse("delta", enriched["response"])
        if store.enabled and conversation_id:
            await store.add_message(conversation_id=conversation_id, role="user", content=payload.message, metadata={})
            await store.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=enriched["response"],
                metadata={"route": "fast", "quick_actions": enriched["quick_actions"]},
            )
        yield _sse(
            "done",
            {"reply": enriched["response"], "quick_actions": enriched["quick_actions"], "conversation_id": conversation_id},
        )
        return

    yield _sse("meta", {"cache": "miss", "route": "full"})

    state = await prepare_state(
        message=payload.message,
        history=history_dump,
        metadata=payload.metadata or {},
    )

    logger.info("=" * 80)
    logger.info("🔍 ROUTING DECISION | route=full(stream) | clear_cache=%s", bool(payload.clear_cache))
    logger.info("Query: %s", (payload.message or "")[:120])
    logger.info("rag_used=%s tool=%s intent=%s", bool(state.get("rag_context")), state.get("tool_call"), state.get("intent"))
    logger.info("=" * 80)

    chunks: list[str] = []
    async for token in stream_synthesize(state):
        chunks.append(token)
        yield _sse("delta", token)

    reply = "".join(chunks).strip()
    tool_result, corrections, totals = _tool_to_response_extras(state)
    enriched = generate_response_with_actions(
        query=payload.message,
        response_text=reply,
        metadata={
            "route": "full",
            "rag_used": bool(state.get("rag_context")),
            "tool_used": (state.get("tool_call") or {}).get("name") if isinstance(state.get("tool_call"), dict) else None,
            "intent": state.get("intent"),
        },
    )

    if cache.enabled and cacheable and reply:
        await cache.set(normalized, CacheEntry(reply=reply, meta={"route": "full"}))

    if store.enabled and conversation_id:
        await store.add_message(conversation_id=conversation_id, role="user", content=payload.message, metadata={})
        await store.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=enriched["response"],
            metadata={
                "route": "full",
                "rag_used": bool(state.get("rag_context")),
                "tool": state.get("tool_call"),
                "quick_actions": enriched["quick_actions"],
            },
        )

    yield _sse(
        "done",
        {
            "reply": enriched["response"],
            "quick_actions": enriched["quick_actions"],
            "conversation_id": conversation_id,
            "raw_output": {
                "route": "full",
                "intent": state.get("intent"),
                "tool": state.get("tool_call"),
                "tool_result": tool_result,
                "rag_used": bool(state.get("rag_context")),
            },
            "corrections": corrections,
            "totals": totals,
            "missing_fields": [],
        },
    )


@router.post("/chat")
@limiter.limit("30/minute") if limiter else lambda f: f
async def chat(payload: ChatInput, request: Request):
    """Chat endpoint (JSON or streaming SSE).

    Rate limit: 30 requests per minute per IP address.
    """
    # Allow cache clearing via query param without changing client payload.
    if _wants_clear_cache(payload, request):
        payload.clear_cache = True
    if _wants_stream(payload, request):
        return StreamingResponse(handle_chat_stream(payload), media_type="text/event-stream")

    result = await handle_chat_non_stream(payload)
    return JSONResponse(result)
