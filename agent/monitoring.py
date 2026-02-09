"""Prometheus metrics for NEXTMIND agent."""
import functools
import inspect
import logging
import time
from typing import Any, Callable

try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Dummy classes for when prometheus is not installed
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def inc(self, *args): pass

    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def observe(self, *args): pass

    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def inc(self, *args): pass
        def dec(self, *args): pass

    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args): pass

logger = logging.getLogger(__name__)

if not PROMETHEUS_AVAILABLE:
    logger.warning("prometheus-client not installed - metrics disabled. Install with: pip install prometheus-client")

# Métriques principales
agent_requests_total = Counter(
    'agent_requests_total',
    'Total number of requests to agent endpoints',
    ['endpoint', 'intent', 'status']
)

agent_request_duration_seconds = Histogram(
    'agent_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint', 'intent'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

agent_llm_calls_total = Counter(
    'agent_llm_calls_total',
    'Total number of LLM API calls',
    ['model', 'endpoint']
)

agent_llm_tokens_total = Counter(
    'agent_llm_tokens_total',
    'Total number of tokens consumed',
    ['model', 'token_type']  # token_type: prompt, completion
)

agent_cache_requests_total = Counter(
    'agent_cache_requests_total',
    'Total cache requests',
    ['result']  # result: hit, miss
)

agent_errors_total = Counter(
    'agent_errors_total',
    'Total number of errors',
    ['endpoint', 'error_type']
)

agent_active_requests = Gauge(
    'agent_active_requests',
    'Number of active requests',
    ['endpoint']
)

agent_info = Info(
    'agent_info',
    'Agent version and configuration info'
)


def track_request(endpoint: str):
    """Decorator to track request metrics."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not PROMETHEUS_AVAILABLE:
                return await func(*args, **kwargs)

            agent_active_requests.labels(endpoint=endpoint).inc()
            start_time = time.time()
            status = "success"
            intent = "unknown"

            try:
                result = await func(*args, **kwargs)

                # Extract intent from result if available
                if isinstance(result, dict):
                    intent = result.get("intent") or result.get("raw_output", {}).get("intent") or "unknown"

                return result
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                agent_errors_total.labels(endpoint=endpoint, error_type=error_type).inc()
                raise
            finally:
                duration = time.time() - start_time
                agent_requests_total.labels(endpoint=endpoint, intent=intent, status=status).inc()
                agent_request_duration_seconds.labels(endpoint=endpoint, intent=intent).observe(duration)
                agent_active_requests.labels(endpoint=endpoint).dec()

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not PROMETHEUS_AVAILABLE:
                return func(*args, **kwargs)

            agent_active_requests.labels(endpoint=endpoint).inc()
            start_time = time.time()
            status = "success"
            intent = "unknown"

            try:
                result = func(*args, **kwargs)

                # Extract intent from result if available
                if isinstance(result, dict):
                    intent = result.get("intent") or "unknown"

                return result
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                agent_errors_total.labels(endpoint=endpoint, error_type=error_type).inc()
                raise
            finally:
                duration = time.time() - start_time
                agent_requests_total.labels(endpoint=endpoint, intent=intent, status=status).inc()
                agent_request_duration_seconds.labels(endpoint=endpoint, intent=intent).observe(duration)
                agent_active_requests.labels(endpoint=endpoint).dec()

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def track_llm_call(model: str, endpoint: str, prompt_tokens: int = 0, completion_tokens: int = 0):
    """Track LLM API call metrics."""
    if not PROMETHEUS_AVAILABLE:
        return

    agent_llm_calls_total.labels(model=model, endpoint=endpoint).inc()
    if prompt_tokens > 0:
        agent_llm_tokens_total.labels(model=model, token_type="prompt").inc(prompt_tokens)
    if completion_tokens > 0:
        agent_llm_tokens_total.labels(model=model, token_type="completion").inc(completion_tokens)


def track_cache_hit(hit: bool):
    """Track cache hit/miss."""
    if not PROMETHEUS_AVAILABLE:
        return

    result = "hit" if hit else "miss"
    agent_cache_requests_total.labels(result=result).inc()


# Initialize agent info
try:
    import os
    agent_info.info({
        'version': '2.0',
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'llm_model': os.getenv('LLM_MODEL', 'gpt-4o-mini'),
    })
except Exception:
    pass
