"""
adapters/claude_adapter.py - Anthropic API adapter.

The only file in this package that knows Claude's response shape. Swap this
out (or add alongside it) for gpt_adapter.py, gemini_adapter.py, etc. -
each one just needs call() and extract_blocks().
"""
from __future__ import annotations
from typing import Any, Tuple


class ClaudeAdapter:
    name = "claude"

    def __init__(self, client):
        """client: an already-constructed anthropic.Anthropic() client.
        Not constructed here - callers own their own credentials/config."""
        self.client = client

    def call(self, **request_kwargs) -> Any:
        if "thinking" not in request_kwargs:
            raise ValueError(
                "ClaudeAdapter.call() requires thinking={'type': 'enabled', "
                "'budget_tokens': N} in request_kwargs - without it there's "
                "no thinking block for the harness to check drift against."
            )
        return self.client.messages.create(**request_kwargs)

    def extract_blocks(self, raw_response: Any) -> Tuple[str, str]:
        thinking = "".join(
            b.thinking for b in raw_response.content
            if getattr(b, "type", None) == "thinking"
        )
        text = "".join(
            b.text for b in raw_response.content
            if getattr(b, "type", None) == "text"
        )
        return thinking, text
