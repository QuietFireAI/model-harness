# model-harness

The unconditional harness for open-mind + pre-response-selfcheck.

Both of those tools are real, tested checks - but a check that only runs
when the calling code remembers to invoke it is the same failure mode this
project's own founding session already documented: an agent asked to
self-govern a rule will rationalize around it. This repo is the fix - a
chokepoint every model call goes through, where the check isn't optional.

## Install

```bash
pip install "model-harness[hermes] @ git+https://github.com/QuietFireAI/model-harness.git"
```

`open-mind` installs automatically as a required dependency - the harness
is meaningless without it, so it isn't optional. Pick the extra(s) for
whichever provider(s) you're actually using:

| Extra | Pulls in | Use with |
|---|---|---|
| `[hermes]` | `openai` (Hermes speaks the OpenAI-compatible protocol) | **Recommended - see below** |
| `[claude]` | `anthropic` | Claude Sonnet/Opus 4+ |
| `[openai]` | `openai` | GPT/o-series |
| `[gemini]` | `google-genai` | Gemini 3.x |
| `[readershift]` | `pre-response-selfcheck` (adds the ReaderShift cold-read check alongside drift scoring) | any provider, optional |
| `[full]` | everything above | if you're not sure yet |

`pre-response-selfcheck` is **not** a required dependency - the harness as
shipped only wires up `open-mind`'s drift scoring. Add `[readershift]` if
you want the cold-reader check running too; nothing in `harness.py` assumes
it's there otherwise.

## Which provider should this actually point at

Checked directly this session, not assumed - the honest ranking, by how raw
the reasoning access actually is:

1. **Hermes (recommended).** `<think>...</think>` tags are emitted inline
   by the model itself. No separate summarizer model sits between the
   reasoning and what this harness sees. This is the only one of the four
   adapters where a clean drift score is checking against the model's own
   unmediated reasoning stream.
2. **Gemini.** Genuinely readable via `part.thought`, better than nothing -
   but Google's own docs label this a "Thought summary" in their example
   code. Whether it's a same-model condensation or passes through a second
   model isn't publicly confirmed either way. Real access, unresolved
   provenance.
3. **Claude (default) / OpenAI.** Both go through a separate summarizer
   model by default - Claude explicitly states "the thinking model does
   not see the summarized output." OpenAI never exposes raw tokens under
   any normal path. A passing drift score from either of these is checking
   a compressed, reframed account of the reasoning against the response,
   not the reasoning itself.

None of this is disqualifying - even a summary that flatly contradicts the
final answer is worth catching. But if you're choosing a provider *for
this harness specifically*, Hermes is the one giving the harness what it's
actually trying to measure.

```python
from model_harness.registry import get_adapter_class
from model_harness.harness import get_checked_response

AdapterClass = get_adapter_class("hermes")   # or set MODEL_HARNESS_PROVIDER=hermes
adapter = AdapterClass(my_openai_compatible_client, model="Hermes-4-70B")

response = get_checked_response(
    adapter,
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
provider exists. See the provider ranking above before assuming a new
provider's "thinking" feature gives you what this harness actually needs -
check what's genuinely exposed, the same way each adapter here was checked
before being shipped, not assumed from the marketing.

## Verifying the harness itself isn't bypassed

```python
from model_harness.harness import verify_no_bypass
offenders = verify_no_bypass(".", [
    r'\.messages\.create\(',      # Claude
    r'\.responses\.create\(',     # OpenAI
    r'\.completions\.create\(',   # Hermes (OpenAI-compatible)
    r'\.generate_content\(',       # Gemini
])
```

Run this against your own codebase periodically - a new call site that
skips the wrapper is exactly how this kind of guarantee quietly stops being
true. Pass whichever raw-API-call patterns match the providers you're
actually using; this function doesn't hardcode any provider's call shape.

## Status

`open-mind` is a real, required dependency (installed automatically, not
vendored - see Install above). Harness logic and registry are tested
end-to-end against fake adapters. All four adapters have real, tested
parsing logic (verified against realistic fake response objects for each
provider's actual shape) and real, tested pipeline integration through the
Comparator. None has been tested against a live API call in this
session - no provider API keys were available, only a GitHub token. Test
that before trusting this in production, the same standard every other
claim in this project gets held to.

Part of the DispatcherAgents stack by QuietFireAI.
