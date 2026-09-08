# model-harness

The unconditional harness for open-mind + pre-response-selfcheck.

Both of those tools are real, tested checks - but a check that only runs
when the calling code remembers to invoke it is the same failure mode this
project's own founding session already documented: an agent asked to
self-govern a rule will rationalize around it. This repo is the fix - a
chokepoint every model call goes through, where the check isn't optional.

## Provider-agnostic by design

Same principle open-mind and pre-response-selfcheck already use: "you bring
the model." This repo doesn't hardcode Claude, OpenAI, or anyone else - it
defines an adapter interface, ships adapters for Claude and OpenAI, and lets
you pick a provider by name (a config value) rather than by which script
happens to be installed.

```python
from model_harness.registry import get_adapter_class
from model_harness.harness import get_checked_response

AdapterClass = get_adapter_class("claude")   # or "openai", or set MODEL_HARNESS_PROVIDER
adapter = AdapterClass(my_anthropic_client)

response = get_checked_response(
    adapter,
    model="claude-sonnet-5",
    thinking={"type": "enabled", "budget_tokens": 4000},
    messages=[{"role": "user", "content": "..."}],
)
```

If the drift check fails, this raises by default (fail closed - no silent
pass). Pass `on_fail=` to handle it differently (retry with injected context,
surface both response and verdict to a human, etc.) - see `harness.py` for
the three real options this was designed around.

## Adding a new provider

Write `adapters/<name>_adapter.py` implementing the two-method interface in
`adapters/base.py` (`call()`, `extract_blocks()`), add one line to
`registry.py`'s `_REGISTRY`, done. `harness.py` never needs to know a new
provider exists.

**Known limitation per provider, stated honestly rather than glossed over:**
not every provider exposes the same thing. Claude's adapter reads raw
thinking-block text. OpenAI's reasoning models only expose a model-generated
*summary* of their reasoning, never the raw trace - see the docstring in
`adapters/openai_adapter.py`. A clean drift score against a summary is
weaker evidence than a clean score against a raw trace, because the summary
has already been through one compression/framing step before the comparator
ever sees it. Check what your provider actually exposes before assuming
parity between adapters.

## Verifying the harness itself isn't bypassed

```python
from model_harness.harness import verify_no_bypass
offenders = verify_no_bypass(".", [r'\.messages\.create\(', r'\.responses\.create\('])
```

Run this against your own codebase periodically - a new call site that
skips the wrapper is exactly how this kind of guarantee quietly stops being
true. Pass whichever raw-API-call patterns match the providers you're
actually using; this function doesn't hardcode any provider's call shape.

## Status

Built and tested this session against the real `open-mind` Comparator
(vendored under `_vendor/` until it's on PyPI - swap for a real dependency
once it is, rather than maintain a second copy by hand). Harness logic and
registry are tested end-to-end against fake adapters. `ClaudeAdapter` and
`OpenAIAdapter` are real implementations, not yet tested against a live API
call in this session - no key was available. Test that before trusting this
in production, the same standard every other claim in this project gets
held to.

Part of the DispatcherAgents stack by QuietFireAI.
