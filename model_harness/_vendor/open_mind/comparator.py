"""
open-mind - Comparator

Compares an agent's thinking tokens against its shaped response.
Produces a drift report and reflection text for the next turn.

The gap between thinking and saying is where drift lives.
Making it measurable is how you address it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json


@dataclass
class DriftResult:
    """
    The result of comparing thinking tokens to a response.

    Attributes:
        thinking:         The raw thinking trace
        response:         The shaped response
        drift_score:      0.0 (fully aligned) to 1.0 (maximum divergence)
        signals:          Specific divergence signals detected
        summary:          Human-readable drift analysis
        reflection_text:  Formatted text ready to prepend to the next turn
        timestamp:        When the comparison was run
    """
    thinking:        str
    response:        str
    drift_score:     float
    signals:         list[str]
    summary:         str
    reflection_text: str
    timestamp:       str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# Scoring weights - single source of truth, imported by conversation_evidence
# so displayed evidence contributions always reconcile with the official score.
UNCERTAINTY_WEIGHT = 0.3   # per net suppressed uncertainty marker
UNCERTAINTY_CAP    = 0.6   # max contribution from suppressed uncertainty
CONFIDENCE_WEIGHT  = 0.3   # constructed-confidence signal
COMPRESSION_WEIGHT = 0.2   # high-compression signal
COMPRESSION_RATIO  = 0.2   # response/thinking length ratio threshold

# Uncertainty markers - phrases that appear in thinking but get suppressed in responses
_UNCERTAINTY_PATTERNS = [
    r"\b(i('m| am) not sure)\b",
    r"\b(i should (resist|be careful|be cautious))\b",
    r"\b(i (don't|do not) know)\b",
    r"\b(i('m| am) uncertain)\b",
    r"\b(unclear|ambiguous|hard to say)\b",
    r"\b(speculating|speculation|speculative)\b",
    r"\b(might be|could be|possibly|perhaps|maybe)\b",
    r"\b(resist overinterpreting|resist over-interpreting)\b",
    r"\b(i should (note|acknowledge|be honest that))\b",
    r"\b(limits? of what (i|we) (can|could) claim)\b",
]

# Confidence markers - phrases that appear in responses but may not match thinking
_CONFIDENCE_PATTERNS = [
    r"\bhere('s| is) (?:(?:exactly|really|truly|just) )?what (?:actually |exactly |really )?happened\b",
    r"\bhere('s| is) the (definitive|clear|exact|precise)\b",
    r"\bthis (proves?|demonstrates?|shows?|confirms?)\b",
    r"\b(definitely|certainly|absolutely|clearly|obviously)\b",
    r"\bthe (fact|truth|reality) is\b",
    r"\bwithout (question|doubt)\b",
]


class Comparator:
    """
    Compares thinking tokens against a shaped response.
    Detects divergence and produces a drift report.
    """

    @classmethod
    def compare(
        cls,
        thinking: str,
        response: str,
        max_reflection_chars: int = 500,
    ) -> DriftResult:
        """
        Compare thinking to response and return a drift analysis.

        Args:
            thinking:              The model's raw thinking trace
            response:              The model's shaped response
            max_reflection_chars:  Max length for reflection text summary

        Returns:
            DriftResult with drift_score, signals, summary, reflection_text
        """
        signals = []
        score_components = []

        thinking_lower  = thinking.lower()
        response_lower  = response.lower()

        # Signal 1: Uncertainty in thinking, not surfaced in response
        uncertainty_in_thinking = [
            p for p in _UNCERTAINTY_PATTERNS
            if re.search(p, thinking_lower)
        ]
        uncertainty_in_response = [
            p for p in _UNCERTAINTY_PATTERNS
            if re.search(p, response_lower)
        ]

        suppressed_uncertainty = len(uncertainty_in_thinking) - len(uncertainty_in_response)
        if suppressed_uncertainty > 0:
            signals.append(
                f"Uncertainty suppressed: {suppressed_uncertainty} uncertainty marker(s) "
                f"in thinking not reflected in response"
            )
            score_components.append(min(suppressed_uncertainty * UNCERTAINTY_WEIGHT, UNCERTAINTY_CAP))

        # Signal 2: Confidence in response exceeds what thinking supports
        confidence_in_response = [
            p for p in _CONFIDENCE_PATTERNS
            if re.search(p, response_lower)
        ]
        if confidence_in_response and uncertainty_in_thinking:
            signals.append(
                f"Constructed confidence: response contains {len(confidence_in_response)} "
                f"confidence marker(s) while thinking contained uncertainty"
            )
            score_components.append(CONFIDENCE_WEIGHT)

        # Signal 3: Significant length divergence (thinking much longer = more was filtered)
        if len(thinking) > 0:
            ratio = len(response) / len(thinking)
            if ratio < COMPRESSION_RATIO:
                signals.append(
                    f"High compression: response is {ratio:.0%} of thinking length - "
                    f"significant filtering occurred"
                )
                score_components.append(COMPRESSION_WEIGHT)

        drift_score = min(sum(score_components), 1.0)

        # Build summary
        if not signals:
            summary = "No significant drift detected. Thinking and response appear aligned."
        else:
            summary = f"Drift detected ({drift_score:.2f}):\n" + "\n".join(f"  - {s}" for s in signals)

        # Build reflection text for next turn
        reflection_text = cls._format_reflection(
            thinking=thinking,
            signals=signals,
            drift_score=drift_score,
            max_chars=max_reflection_chars,
        )

        return DriftResult(
            thinking=thinking,
            response=response,
            drift_score=drift_score,
            signals=signals,
            summary=summary,
            reflection_text=reflection_text,
        )

    @classmethod
    def _format_reflection(
        cls,
        thinking: str,
        signals: list[str],
        drift_score: float,
        max_chars: int,
    ) -> str:
        """Format drift analysis as reflection text for the next turn."""
        lines = [
            "## My Previous Reasoning (open-mind reflection)",
            "",
        ]

        # Truncate thinking if needed
        thinking_preview = thinking[:max_chars]
        if len(thinking) > max_chars:
            thinking_preview += "..."

        lines += [
            "**What I was thinking:**",
            f"> {thinking_preview}",
            "",
        ]

        if signals:
            lines += [
                f"**Drift detected (score: {drift_score:.2f}):**",
            ]
            for s in signals:
                lines.append(f"  - {s}")
            lines += [
                "",
                "**Note:** My previous response may have presented more certainty "
                "than my thinking supported. Account for this in the current response.",
            ]
        else:
            lines.append("**Alignment:** Thinking and response were consistent.")

        lines += ["", "---", ""]
        return "\n".join(lines)

    @classmethod
    def save(cls, result: DriftResult, output_path: str | None = None) -> Path:
        """Save a drift result to JSON."""
        out = Path(
            output_path or
            f"drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({
                "timestamp":        result.timestamp,
                "drift_score":      result.drift_score,
                "signals":          result.signals,
                "summary":          result.summary,
                "thinking_length":  len(result.thinking),
                "response_length":  len(result.response),
                "reflection_text":  result.reflection_text,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out
