"""
adapters/openai_adapter.py - OpenAI Responses API adapter.

HONEST LIMITATION, load-bearing for how this adapter should be used: OpenAI's
reasoning models (o-series) do not expose raw reasoning tokens via the API at
all - "While we don't expose the raw reasoning tokens emitted by the model,
you can view a summary of the model's reasoning using the summary parameter."
(OpenAI's own docs, checked at build time, not assumed from memory.)

This is a real, structural difference from ClaudeAdapter, not a minor detail:
`extract_blocks()` here returns a MODEL-GENERATED SUMMARY of the reasoning,
not the reasoning itself. A summary has already been through one round of
compression and framing by the model - the exact kind of step where drift
could be introduced before the comparator ever sees it. A clean drift score
against an OpenAI response is weaker evidence than a clean score against a
Claude response for this reason. Don't treat the two as equivalent.
"""
from __future__ import annotations
from typing import Any, Tuple


class OpenAIAdapter:
    name = "openai"

    def __init__(self, client):
        """client: an already-constructed openai.OpenAI() client."""
        self.client = client

    def call(self, **request_kwargs) -> Any:
        request_kwargs.setdefault("reasoning", {}).setdefault("summary", "auto")
        return self.client.responses.create(**request_kwargs)

    def extract_blocks(self, raw_response: Any) -> Tuple[str, str]:
        summary_parts = []
        text_parts = []
        for item in raw_response.output:
            item_type = getattr(item, "type", None)
            if item_type == "reasoning":
                for s in getattr(item, "summary", []) or []:
                    summary_parts.append(getattr(s, "text", ""))
            elif item_type == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        text_parts.append(c.text)
        return "".join(summary_parts), "".join(text_parts)
