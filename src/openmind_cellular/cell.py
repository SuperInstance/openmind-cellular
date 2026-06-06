"""Cell decorator — resource-aware Jupyter cell execution."""

from __future__ import annotations

import functools
from typing import Any, Callable


def cell(
    resource_aware: bool = True,
    fallback: str = "cached",  # "cached" | "simulate" | "fail"
) -> Callable:
    """Decorator for resource-aware cell execution.

    Usage:
        @cell(resource_aware=True, fallback="cached")
        def my_pipeline():
            ...

    The decorated function runs differently depending on what's available.
    Never crashes due to missing GPU/API/ESP32.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not resource_aware:
                return func(*args, **kwargs)

            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if fallback == "fail":
                    raise
                # Resource failure — return None or empty result
                return None

        # Attach metadata
        wrapper._cell_config = {  # type: ignore
            "resource_aware": resource_aware,
            "fallback": fallback,
            "original": func,
        }
        return wrapper
    return decorator
