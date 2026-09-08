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

**Known limitation, checked and corrected after the initial push - not
every adapter is in the same position.** Neither Claude nor OpenAI hands
this harness genuinely raw reasoning by default. For Claude 4+ models, the
`thinking` block is a summary from a separate summarizer model unless you
have a specific enterprise arrangement with Anthropic for full access ("the
thinking model does not see the summarized output," per Anthropic's own
docs). OpenAI's reasoning models never expose raw tokens at all, only a
summary. Both summaries have already been through one compression/reframing
pass before the comparator ever sees them.

**Hermes is genuinely different here, checked not assumed:** Hermes 4 and
DeepHermes emit `<think>...</think>` reasoning directly inline in the raw
completion text - there's no separate summarizer model in the loop at all.
A drift check against Hermes's reasoning is checking the model's own
reasoning stream, not a second model's account of it. If you're running
this harness against a self-hosted or Nous-hosted Hermes model, that's the
strongest foundation of the three adapters shipped here for what this
harness is actually trying to measure. See the docstrings in
`adapters/claude_adapter.py`, `adapters/openai_adapter.py`, and
`adapters/hermes_adapter.py` for the specifics per provider.

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
