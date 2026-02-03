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
from agent.graph import prepare_state, stream_synthesize, synthesize
from agent.logging_config import logger


router = APIRouter()


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatInput(BaseModel):
    message: str
    thread_id: Optional[str] = None
    history: Optional[List[ChatHistoryItem]] = None
    metadata: Optional[Dict[str, Any]] = None
    force_prepare: Optional[bool] = None
    stream: Optional[bool] = None
    clear_cache: Optional[bool] = None


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


async def handle_chat_non_stream(payload: ChatInput) -> dict[str, Any]:
    cache = get_chat_cache()
    normalized = normalize_question(payload.message)
    history_dump = [item.model_dump() for item in (payload.history or [])]
    cacheable = _is_cacheable(payload.metadata, history_dump)
    # Clear cache for this question when requested (dev/test).
    # Note: we still allow caching of the fresh response afterward.
    if cache.enabled and cacheable and payload.clear_cache is True:
        await cache.delete(normalized)

    if cache.enabled and cacheable:
        hit = await cache.get(normalized)
        if hit:
            meta = hit.meta or {}
            return {
                "reply": hit.reply,
                "raw_output": {"cache": "hit", "route": meta.get("route")},
                "corrections": [],
                "totals": {},
                "missing_fields": [],
            }

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
        return {
            "reply": fast,
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

    logger.info("=" * 80)
    logger.info("🔍 ROUTING DECISION | route=full | clear_cache=%s", bool(payload.clear_cache))
    logger.info("Query: %s", (payload.message or "")[:120])
    logger.info("rag_used=%s tool=%s intent=%s", bool(state.get("rag_context")), state.get("tool_call"), state.get("intent"))
    logger.info("=" * 80)

    if cache.enabled and cacheable and reply:
        await cache.set(normalized, CacheEntry(reply=reply, meta={"route": "full"}))

    return {
        "reply": reply,
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


async def handle_chat_stream(payload: ChatInput) -> AsyncIterator[str]:
    cache = get_chat_cache()
    normalized = normalize_question(payload.message)
    history_dump = [item.model_dump() for item in (payload.history or [])]
    cacheable = _is_cacheable(payload.metadata, history_dump)

    if cache.enabled and cacheable and payload.clear_cache is True:
        await cache.delete(normalized)

    if cache.enabled and cacheable:
        hit = await cache.get(normalized)
        if hit:
            yield _sse("meta", {"cache": "hit"})
            yield _sse("delta", hit.reply)
            yield _sse("done", {"reply": hit.reply})
            return

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
        yield _sse("meta", {"cache": "miss", "route": "fast"})
        yield _sse("delta", fast)
        yield _sse("done", {"reply": fast})
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

    if cache.enabled and cacheable and reply:
        await cache.set(normalized, CacheEntry(reply=reply, meta={"route": "full"}))

    yield _sse(
        "done",
        {
            "reply": reply,
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
async def chat(payload: ChatInput, request: Request):
    """Chat endpoint (JSON or streaming SSE)."""
    # Allow cache clearing via query param without changing client payload.
    if _wants_clear_cache(payload, request):
        payload.clear_cache = True
    if _wants_stream(payload, request):
        return StreamingResponse(handle_chat_stream(payload), media_type="text/event-stream")

    result = await handle_chat_non_stream(payload)
    return JSONResponse(result)
