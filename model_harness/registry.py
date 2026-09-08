"""
registry.py - the "user picks a model, gets the right adapter" mechanism.

This is what makes the choice a config value instead of a code edit:

    from model_harness.registry import get_adapter_class

    AdapterClass = get_adapter_class("claude")   # or "openai", or via
                                                  # MODEL_HARNESS_PROVIDER env var
    adapter = AdapterClass(my_provider_client)

Adding a new provider means writing one adapters/<name>_adapter.py implementing
ModelAdapter (see adapters/base.py) and adding one line to _REGISTRY below -
not touching harness.py at all.
"""
from __future__ import annotations
import os
from typing import Type

_REGISTRY = {
    "claude": ("model_harness.adapters.claude_adapter", "ClaudeAdapter"),
    "openai": ("model_harness.adapters.openai_adapter", "OpenAIAdapter"),
    "hermes": ("model_harness.adapters.hermes_adapter", "HermesAdapter"),
    "gemini": ("model_harness.adapters.gemini_adapter", "GeminiAdapter"),
    # Add new providers here: "name": ("module.path", "ClassName")
}


def available_providers() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_adapter_class(provider: str | None = None) -> Type:
    """
    provider: a key in _REGISTRY, or None to read MODEL_HARNESS_PROVIDER
              from the environment.

    Raises a clear, actionable error listing what IS available if the
    requested provider isn't registered - never silently falls back to a
    default provider the caller didn't ask for.
    """
    provider = provider or os.environ.get("MODEL_HARNESS_PROVIDER")
    if not provider:
        raise ValueError(
            "No provider specified. Pass one explicitly (get_adapter_class"
            f"('claude')) or set MODEL_HARNESS_PROVIDER. Available: "
            f"{available_providers()}"
        )
    if provider not in _REGISTRY:
        raise ValueError(
            f"Unknown provider {provider!r}. Available: {available_providers()}. "
            f"To add a new one, write adapters/<name>_adapter.py implementing "
            f"ModelAdapter and register it in registry.py's _REGISTRY."
        )
    module_path, class_name = _REGISTRY[provider]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
