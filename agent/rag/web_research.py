"""Optional web research + documentation updater.

This module is intentionally provider-agnostic and OFF by default.
Enable by setting:
  - WEB_RESEARCH_ENABLED=1
  - TAVILY_API_KEY=...

When enabled and local docs miss, the agent can:
  - search the web
  - summarize results
  - append a short, non-copyrighted note to the relevant local markdown doc
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

from agent.logging_config import logger
from agent.rag.local_docs import ensure_domain_doc


def web_research_enabled() -> bool:
    v = (os.getenv("WEB_RESEARCH_ENABLED") or "").strip().lower()
    return v in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class WebFinding:
    query: str
    answer: str
    sources: list[dict[str, str]]  # [{title,url}, ...]


async def _tavily_search(query: str, *, max_results: int = 3) -> WebFinding | None:
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return None
    if httpx is None:
        logger.warning("Web research disabled: httpx not installed.")
        return None

    # Tavily API: https://docs.tavily.com/
    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max(1, min(max_results, 5)),
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
    }

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.post("https://api.tavily.com/search", json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        logger.warning("Web research (Tavily) failed: %s", exc)
        return None

    answer = str(data.get("answer") or "").strip()
    results = data.get("results") if isinstance(data.get("results"), list) else []
    sources: list[dict[str, str]] = []
    for item in results[: max(1, min(max_results, 5))]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        if url:
            sources.append({"title": title or url, "url": url})

    if not answer and not sources:
        return None
    return WebFinding(query=query, answer=answer, sources=sources)


def _tavily_search_sync(query: str, *, max_results: int = 3) -> WebFinding | None:
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return None
    if httpx is None:
        logger.warning("Web research disabled: httpx not installed.")
        return None

    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max(1, min(max_results, 5)),
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
    }

    try:
        with httpx.Client(timeout=12.0) as client:
            r = client.post("https://api.tavily.com/search", json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        logger.warning("Web research (Tavily) failed: %s", exc)
        return None

    answer = str(data.get("answer") or "").strip()
    results = data.get("results") if isinstance(data.get("results"), list) else []
    sources: list[dict[str, str]] = []
    for item in results[: max(1, min(max_results, 5))]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        if url:
            sources.append({"title": title or url, "url": url})

    if not answer and not sources:
        return None
    return WebFinding(query=query, answer=answer, sources=sources)


async def web_research(query: str, *, max_results: int = 3) -> WebFinding | None:
    """Single entrypoint for web research (provider selection)."""
    if not web_research_enabled():
        return None

    provider = (os.getenv("WEB_RESEARCH_PROVIDER") or "tavily").strip().lower()
    if provider == "tavily":
        return await _tavily_search(query, max_results=max_results)

    logger.warning("Web research provider unsupported: %s", provider)
    return None


def web_research_sync(query: str, *, max_results: int = 3) -> WebFinding | None:
    """Sync entrypoint for web research (used by legacy/sync pipelines)."""
    if not web_research_enabled():
        return None

    provider = (os.getenv("WEB_RESEARCH_PROVIDER") or "tavily").strip().lower()
    if provider == "tavily":
        return _tavily_search_sync(query, max_results=max_results)

    logger.warning("Web research provider unsupported: %s", provider)
    return None

def append_finding_to_doc(*, domain: str, finding: WebFinding, note: str | None = None) -> str | None:
    """Append a short curated note to the domain doc. Returns the doc filename."""
    try:
        path = ensure_domain_doc(domain)
        now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
        lines: list[str] = []
        lines.append("\n")
        lines.append("## Ajouts (auto)\n")
        lines.append(f"- Date: {now}\n")
        lines.append(f"- Requête: {finding.query}\n")
        if note:
            lines.append(f"- Note: {note}\n")
        if finding.answer:
            lines.append("\n")
            lines.append("Synthèse:\n")
            lines.append(f"{finding.answer.strip()}\n")
        if finding.sources:
            lines.append("\n")
            lines.append("Sources:\n")
            for s in finding.sources[:5]:
                title = (s.get("title") or "").strip()
                url = (s.get("url") or "").strip()
                if not url:
                    continue
                lines.append(f"- {title or url} — {url}\n")

        with path.open("a", encoding="utf-8") as fh:
            fh.writelines(lines)
        return path.name
    except Exception as exc:
        logger.warning("Failed to append web finding to doc: %s", exc)
        return None
