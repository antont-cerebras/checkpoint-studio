from typing import Any

class Config:
    """Only `max_pool_connections` is set by `dump.py`; the rest passes through."""

    def __init__(
        self, *, max_pool_connections: int = ..., **kwargs: Any
    ) -> None: ...

__all__ = ["Config"]
