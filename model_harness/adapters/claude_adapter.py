"""
adapters/claude_adapter.py - Anthropic API adapter.

CORRECTION (checked after initial push, not caught before it): this
docstring originally claimed Claude's thinking blocks give raw reasoning
access, contrasted favorably against OpenAI's summary-only access. That was
wrong, asserted without checking, the same failure this whole project keeps
naming. The actual situation, per Anthropic's own docs: for Claude 4+
models, the `thinking` block returned by the API is - by default - a
SUMMARY produced by a separate summarizer model, not the raw reasoning
trace. "The thinking model does not see the summarized output." Genuine raw
access exists only through a specific enterprise arrangement, not normal
API access. `redacted_thinking` blocks (safety-flagged content) are opaque
signatures, unreadable even as a summary, and must be passed back verbatim
in multi-turn use or the conversation breaks.

Net effect: ClaudeAdapter and OpenAIAdapter are in the same position, not a
tiered one. Both typically hand the comparator a model-generated summary
that has already been through one compression/reframing pass before drift
scoring ever sees it - which is weaker evidence than a clean score against
genuinely raw reasoning, for the same reason stated in openai_adapter.py.
Don't read a passing drift score from either adapter as checking against
unfiltered thought. It's checking a summary against a response - real
signal, narrower guarantee than originally claimed here.
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
        # Note: any redacted_thinking blocks (type == "redacted_thinking")
        # are opaque and correctly excluded here - there is no text to read
        # from them, by design, for safety reasons.
        thinking = "".join(
            b.thinking for b in raw_response.content
            if getattr(b, "type", None) == "thinking"
        )
        text = "".join(
            b.text for b in raw_response.content
            if getattr(b, "type", None) == "text"
        )
        return thinking, text
