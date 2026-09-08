"""
adapters/base.py - the interface every model adapter implements.

An adapter's whole job: call the model, and split its raw response into
(thinking, text). That's the entire integration surface, same principle as
open-mind and pre-response-selfcheck's own "you bring the model" design -
the harness core never touches provider-specific API shapes directly.
"""
from __future__ import annotations
from typing import Any, Protocol, Tuple


class ModelAdapter(Protocol):
    """Any adapter must implement these two methods."""

    def call(self, **request_kwargs) -> Any:
        """Send a request to the model. Returns the raw provider response -
        whatever shape that provider's SDK gives back."""
        ...

    def extract_blocks(self, raw_response: Any) -> Tuple[str, str]:
        """Split the raw response into (thinking, text). If a provider
        doesn't expose a separate reasoning/thinking trace, thinking should
        be "" - callers can decide whether that's acceptable (the harness
        core raises if thinking is required and empty, since a drift check
        against nothing isn't a check)."""
        ...
