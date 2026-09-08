"""
adapters/hermes_adapter.py - Nous Research Hermes (4 / 4.3 / DeepHermes) adapter.

REAL ADVANTAGE over ClaudeAdapter and OpenAIAdapter, checked not assumed:
Hermes's hybrid reasoning mode emits <think>...</think> directly inline in
the raw completion text - there is no separate summarizer model in the loop.
What you get back genuinely is the model's own reasoning stream, not a
second model's compressed account of it. This is the actual reason to
prefer this adapter for the harness, not just "fewer moving parts."

Works against any OpenAI-compatible chat completions endpoint - Nous's own
portal (https://inference-api.nousresearch.com/v1), OpenRouter, or a
self-hosted server (Ollama, vLLM) running a Hermes GGUF/weights. base_url
is the only thing that changes between those.

Two ways reasoning gets triggered depending on how you're serving the model:
  - Nous's own portal API: pass reasoning=True in request_kwargs.
  - Generic OpenAI-compatible serving (self-hosted, OpenRouter): the model
    needs the standard Nous reasoning system prompt (REASONING_SYSTEM_PROMPT
    below) prepended, since there's no dedicated API flag for it there.
extract_blocks() regex-parses <think> tags out of the completion either way
- it doesn't care which triggering path produced them.
"""
from __future__ import annotations
import re
from typing import Any, Tuple

REASONING_SYSTEM_PROMPT = (
    "You are a deep thinking AI, you may use extremely long chains of "
    "thought to deeply consider the problem and deliberate with yourself "
    "via systematic reasoning processes to help come to a correct solution "
    "prior to answering. You should enclose your thoughts and internal "
    "monologue inside <think> </think> tags, and then provide your "
    "solution or response to the problem."
)

_THINK_TAG = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class HermesAdapter:
    name = "hermes"

    def __init__(self, client, model: str = "Hermes-4-70B", inject_system_prompt: bool = True):
        """client: an OpenAI-compatible client (openai.OpenAI(base_url=...))
        pointed at whichever Hermes endpoint you're using - Nous's portal,
        OpenRouter, or your own self-hosted server.

        inject_system_prompt: if True (default), prepends
        REASONING_SYSTEM_PROMPT as a system message when one isn't already
        present. Set False if you're calling Nous's own portal API with
        reasoning=True, which doesn't need it - or if you're managing the
        reasoning prompt yourself."""
        self.client = client
        self.model = model
        self.inject_system_prompt = inject_system_prompt

    def call(self, **request_kwargs) -> Any:
        request_kwargs.setdefault("model", self.model)
        messages = request_kwargs.get("messages", [])

        if self.inject_system_prompt and not any(
            m.get("role") == "system" for m in messages
        ):
            messages = [{"role": "system", "content": REASONING_SYSTEM_PROMPT}] + list(messages)
            request_kwargs["messages"] = messages

        return self.client.chat.completions.create(**request_kwargs)

    def extract_blocks(self, raw_response: Any) -> Tuple[str, str]:
        full_text = raw_response.choices[0].message.content or ""

        thinking_parts = _THINK_TAG.findall(full_text)
        thinking = "".join(thinking_parts).strip()

        # Everything outside <think> tags is the actual response - not a
        # separate field, since Hermes doesn't structure it that way.
        text = _THINK_TAG.sub("", full_text).strip()

        return thinking, text
