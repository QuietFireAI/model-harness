"""
adapters/gemini_adapter.py - Google Gemini adapter (native google-genai SDK).

WHERE THIS SITS, checked against current (2026) official docs, not assumed:
better than Claude/OpenAI's default access, but not the same guarantee as
Hermes. Gemini 3.x models return per-part `thought` booleans - you can
genuinely read the reasoning text via `part.thought == True`, unlike OpenAI
(no raw access under any normal path) and unlike Claude's default (summary
via a *separate* summarizer model). But Google's own docs label this content
a "Thought summary" in their own example code - it is not confirmed to be
the unprocessed chain-of-thought, and there's no public confirmation of
whether it passes through a second model the way Claude's does or is a
same-model condensation. Treat it as real, better-than-nothing access with
an unresolved provenance question, not as equivalent to Hermes's inline
<think> tags, which are unambiguously the model's own output stream.

Separately: "thought signatures" are opaque strings attached to thinking/
function-call parts, required for multi-turn continuity, unrelated to
whether the *text* is readable. They don't block reading part.text when
part.thought is True - they're a different requirement (preserve and
resend them in multi-turn use) that this single-turn adapter doesn't need
to handle, but multi-turn callers should be aware of.

Requires: pip install google-genai
"""
from __future__ import annotations
from typing import Any, Tuple


class GeminiAdapter:
    name = "gemini"

    def __init__(self, client, model: str = "gemini-3-flash-preview"):
        """client: an already-constructed google.genai.Client() instance."""
        self.client = client
        self.model = model

    def call(self, **request_kwargs) -> Any:
        from google.genai import types
        request_kwargs.setdefault("model", self.model)

        config = request_kwargs.get("config")
        if config is None:
            request_kwargs["config"] = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(include_thoughts=True)
            )
        elif getattr(config, "thinking_config", None) is None:
            config.thinking_config = types.ThinkingConfig(include_thoughts=True)

        return self.client.models.generate_content(**request_kwargs)

    def extract_blocks(self, raw_response: Any) -> Tuple[str, str]:
        thought_parts = []
        text_parts = []
        for part in raw_response.candidates[0].content.parts:
            if not getattr(part, "text", None):
                continue
            if getattr(part, "thought", False):
                thought_parts.append(part.text)
            else:
                text_parts.append(part.text)
        return "".join(thought_parts), "".join(text_parts)
