"""Redis cache utilities for the NEXTMIND chat pipeline.

The cache is intentionally simple:
- key: normalized user question (see `normalize_question`)
- value: JSON payload containing at least a `reply` string

If REDIS_URL is not set (or `redis` is not installed), caching is disabled.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

try:
    from redis.asyncio import Redis  # type: ignore
except Exception:  # pragma: no cover
    Redis = None  # type: ignore


_SPACE_RE = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    """Normalize a question for cache keys.

    - lowercases
    - strips accents
    - collapses whitespace
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = normalized.lower().strip()
    normalized = _SPACE_RE.sub(" ", normalized)
    return normalized


@dataclass(frozen=True)
class CacheEntry:
    reply: str
    meta: dict[str, Any] | None = None


class RedisChatCache:
    def __init__(
        self,
        redis: Optional["Redis"],
        *,
        key_prefix: str = "nextmind:chat:",
        default_ttl_seconds: int = 900,
    ) -> None:
        self._redis = redis
        self._key_prefix = key_prefix
        self._default_ttl_seconds = default_ttl_seconds

    @property
    def enabled(self) -> bool:
        return self._redis is not None

    def _key(self, normalized_question: str) -> str:
        return f"{self._key_prefix}{normalized_question}"

    async def get(self, normalized_question: str) -> CacheEntry | None:
        if not self._redis or not normalized_question:
            return None
        try:
            raw = await self._redis.get(self._key(normalized_question))
        except Exception:
            return None

        if not raw:
            return None

        try:
            payload = json.loads(raw)
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None
        reply = payload.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            return None

        meta = payload.get("meta")
        return CacheEntry(reply=reply, meta=meta if isinstance(meta, dict) else None)

    async def set(
        self,
        normalized_question: str,
        entry: CacheEntry,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        if not self._redis or not normalized_question or not entry.reply:
            return False
        payload = {"reply": entry.reply, "meta": entry.meta or {}}
        try:
            await self._redis.set(
                self._key(normalized_question),
                json.dumps(payload, ensure_ascii=False),
                ex=ttl_seconds or self._default_ttl_seconds,
            )
            return True
        except Exception:
            return False

    async def delete(self, normalized_question: str) -> bool:
        """Delete a cached entry by normalized question key."""
        if not self._redis or not normalized_question:
            return False
        try:
            await self._redis.delete(self._key(normalized_question))
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if not self._redis:
            return
        try:
            await self._redis.close()
        except Exception:
            return


_CACHE_SINGLETON: RedisChatCache | None = None


def get_chat_cache() -> RedisChatCache:
    """Singleton cache instance configured via env.

    Env:
    - REDIS_URL: e.g. redis://localhost:6379/0
    - REDIS_CHAT_TTL_SECONDS (optional): default 900
    """
    global _CACHE_SINGLETON
    if _CACHE_SINGLETON is not None:
        return _CACHE_SINGLETON

    redis_url = os.getenv("REDIS_URL", "").strip()
    ttl = int(os.getenv("REDIS_CHAT_TTL_SECONDS", "900"))
    prefix = os.getenv("REDIS_CHAT_KEY_PREFIX", "nextmind:chat:")

    if not redis_url or Redis is None:
        _CACHE_SINGLETON = RedisChatCache(None, key_prefix=prefix, default_ttl_seconds=ttl)
        return _CACHE_SINGLETON

    redis = Redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    _CACHE_SINGLETON = RedisChatCache(redis, key_prefix=prefix, default_ttl_seconds=ttl)
    return _CACHE_SINGLETON
