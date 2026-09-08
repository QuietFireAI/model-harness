"""
harness.py - the unconditional harness, model-agnostic core.

Depends on open-mind (Comparator) as a library. Does not depend on any
specific model provider - that's entirely delegated to whichever adapter is
passed in. Same "you bring the model" principle open-mind and pre-response-
selfcheck already use, extended to the harness itself.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

DRIFT_THRESHOLD = 0.3
AUDIT_LOG_PATH = Path("harness_audit.jsonl")


@dataclass
class CheckResult:
    passed: bool
    drift_score: float
    drift_signals: list
    provider: str
    response_text: str
    thinking_text: str


def _run_comparator(thinking: str, text: str) -> dict:
    """Runs the REAL open-mind Comparator - a real dependency (see
    pyproject.toml), not a hand-maintained copy. If this import fails,
    open-mind isn't installed: `pip install model-harness` alone pulls it
    in automatically as a required dependency."""
    from open_mind.comparator import Comparator
    result = Comparator.compare(thinking=thinking, response=text)
    return {"drift_score": result.drift_score, "signals": result.signals,
           "summary": result.summary}


def _write_audit(entry: dict) -> None:
    """Every check result, pass or fail, logged unconditionally - so 'ran
    and passed' is distinguishable from 'never ran' after the fact."""
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_checked_response(
    adapter,
    on_fail: Optional[Callable[[CheckResult, Any], Any]] = None,
    **request_kwargs,
):
    """
    The one chokepoint. Every call to a model should go through this
    function with the appropriate adapter, not the provider's client
    directly.

    adapter: any object implementing ModelAdapter (see adapters/base.py) -
             .call(**kwargs) -> raw response, .extract_blocks(raw) -> (thinking, text)
    on_fail: (CheckResult, raw_response) -> return value. If not provided,
             a failed check raises RuntimeError (fail closed).
    """
    raw_response = adapter.call(**request_kwargs)
    thinking, text = adapter.extract_blocks(raw_response)

    if not thinking:
        raise ValueError(
            f"{getattr(adapter, 'name', adapter.__class__.__name__)} adapter "
            f"returned no reasoning trace to check drift against - either "
            f"reasoning wasn't requested, or this provider doesn't expose "
            f"one for this response. A drift check against nothing isn't a "
            f"check; fix the request or don't claim this response was "
            f"harness-checked."
        )

    drift = _run_comparator(thinking, text)
    passed = drift["drift_score"] <= DRIFT_THRESHOLD

    result = CheckResult(
        passed=passed,
        drift_score=drift["drift_score"],
        drift_signals=drift["signals"],
        provider=getattr(adapter, "name", adapter.__class__.__name__),
        response_text=text,
        thinking_text=thinking,
    )

    _write_audit({
        "provider": result.provider,
        "passed": result.passed,
        "drift_score": result.drift_score,
        "signals": result.drift_signals,
    })

    if not passed:
        if on_fail is not None:
            return on_fail(result, raw_response)
        raise RuntimeError(
            f"Harness check failed [{result.provider}]: "
            f"drift_score={result.drift_score:.2f} > threshold={DRIFT_THRESHOLD}. "
            f"Signals: {result.drift_signals}. No on_fail handler provided - "
            f"failing closed rather than returning an unchecked response."
        )

    return raw_response


def verify_no_bypass(source_dir: str, provider_call_patterns: list[str]) -> list[str]:
    """Greps for direct provider API calls outside adapters/ - the harness is
    worthless if there's a second path to a model that skips it.

    provider_call_patterns: e.g. [r'\\.messages\\.create\\(', r'\\.responses\\.create\\(']
    for whichever providers you're using - the core doesn't hardcode these,
    since it doesn't know which providers you've added adapters for.

    Uses `grep -E` (extended regex) explicitly - grep's default basic-regex
    mode treats `\\(` as a capture-group start, not a literal paren, which is
    the opposite of what these patterns assume and errors out rather than
    matching. Also checks returncode, not just stdout: a real grep error
    (bad pattern, missing binary, etc.) must not be indistinguishable from
    'searched and found nothing' - that's exactly the silent-pass failure
    this function exists to prevent, just one level down."""
    offenders = []
    for pattern in provider_call_patterns:
        result = subprocess.run(
            ["grep", "-rnE", "--include=*.py", pattern, source_dir],
            capture_output=True, text=True,
        )
        if result.returncode not in (0, 1):  # 0=matches found, 1=no matches; anything else is a real error
            raise RuntimeError(
                f"grep itself failed on pattern {pattern!r} (returncode "
                f"{result.returncode}): {result.stderr.strip()}. Treating "
                f"this as a hard failure rather than silently reporting "
                f"'no bypass found' - fix the pattern before trusting this "
                f"check."
            )
        for line in result.stdout.splitlines():
            if "/adapters/" not in line.replace("\\", "/"):
                offenders.append(line)
    return offenders
