from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

try:
    from redis.asyncio import Redis  # type: ignore
except Exception:  # pragma: no cover
    Redis = None  # type: ignore


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str
    timestamp: str
    metadata: dict[str, Any]


class ConversationStore:
    """Stocke et récupère l'historique des conversations dans Redis."""

    def __init__(
        self,
        redis: Optional["Redis"],
        *,
        key_prefix: str = "nextmind:conversation:",
        ttl_seconds: int = 3600 * 24,
    ) -> None:
        self._redis = redis
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds

    @property
    def enabled(self) -> bool:
        return self._redis is not None

    def _key(self, conversation_id: str) -> str:
        return f"{self._key_prefix}{conversation_id}:history"

    async def add_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._redis or not conversation_id or not content:
            return
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        try:
            await self._redis.rpush(self._key(conversation_id), json.dumps(message.__dict__, ensure_ascii=False))
            await self._redis.expire(self._key(conversation_id), self._ttl_seconds)
        except Exception:
            return

    async def get_history(self, *, conversation_id: str, limit: int = 10) -> list[dict[str, Any]]:
        if not self._redis or not conversation_id or limit <= 0:
            return []
        try:
            messages = await self._redis.lrange(self._key(conversation_id), -limit, -1)
        except Exception:
            return []
        out: list[dict[str, Any]] = []
        for raw in messages or []:
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("role"), str) and isinstance(parsed.get("content"), str):
                out.append(parsed)
        return out

    async def clear_history(self, *, conversation_id: str) -> None:
        if not self._redis or not conversation_id:
            return
        try:
            await self._redis.delete(self._key(conversation_id))
        except Exception:
            return


_STORE_SINGLETON: ConversationStore | None = None


def get_conversation_store() -> ConversationStore:
    """Singleton store instance configured via env.

    Env:
    - REDIS_URL: e.g. redis://localhost:6379/0
    - REDIS_CONVERSATION_TTL_SECONDS (optional): default 86400
    - REDIS_CONVERSATION_KEY_PREFIX (optional): default nextmind:conversation:
    """
    global _STORE_SINGLETON
    if _STORE_SINGLETON is not None:
        return _STORE_SINGLETON

    redis_url = os.getenv("REDIS_URL", "").strip()
    ttl = int(os.getenv("REDIS_CONVERSATION_TTL_SECONDS", "86400"))
    prefix = os.getenv("REDIS_CONVERSATION_KEY_PREFIX", "nextmind:conversation:")

    if not redis_url or Redis is None:
        _STORE_SINGLETON = ConversationStore(None, key_prefix=prefix, ttl_seconds=ttl)
        return _STORE_SINGLETON

    redis = Redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    _STORE_SINGLETON = ConversationStore(redis, key_prefix=prefix, ttl_seconds=ttl)
    return _STORE_SINGLETON

