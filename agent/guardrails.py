"""
agent/guardrails.py  —  Task 2.3: Hallucination Mitigation & Output Validation

Validates agent outputs before returning them to the user.

Fixes applied
-------------
1. BAD_PATTERNS now loaded from config.yaml (guardrails.bad_date_patterns) so
   year ranges can be updated without touching code.
2. validate() signature unchanged; _last_checks side-channel retained for
   display helpers (non-thread-safe — acceptable for single-process CLI use).
3. Two display modes via format_confidence_block():
   - verbose=True  → full card for demo.py  (developer / reviewer)
   - verbose=False → compact one-line bar for main.py  (chat user)

Scoring rubric
--------------
Base                                    : 0.50
+0.20  citations present
+0.10  all [Source N] indices valid
+0.10  tool results contain real data
+0.10  no hallucination signal phrases
+0.10  Check C: all numbers traceable to source
-0.20  Check C: any number not in source
-0.20  bad date / fabrication pattern
-0.10  orphan [Source N] references
+0.10  Check D: LLM judge score >= 0.8   (conditional)
 0.00  Check D: LLM judge score 0.5–0.8  (conditional)
-0.15  Check D: LLM judge score < 0.5    (conditional)
"""

from __future__ import annotations

import logging
import re

from agent.faithfulness import (
    SEMANTIC_TRIGGER_THRESHOLD,
    check_numeric,
    check_semantic,
)
from agent.cache import get_config

logger = logging.getLogger(__name__)


HALLUCINATION_PHRASES = [
    "i don't have access",
    "i cannot access",
    "i'm unable to",
    "as of my knowledge cutoff",
    "i don't know",
    "cannot be determined",
    "i'm not sure",
    "i am not able",
    "i have no information",
]

# Default patterns — overridden by config.yaml guardrails.bad_date_patterns
_DEFAULT_BAD_PATTERNS = [
    r"\b202[56]-\d{2}-\d{2}\b",
    r"i (made up|fabricated|invented)",
]


def _get_bad_patterns() -> list[str]:
    """Load bad patterns from config, falling back to defaults."""
    try:
        cfg = get_config()
        return cfg.get("guardrails", {}).get("bad_date_patterns", _DEFAULT_BAD_PATTERNS)
    except Exception:
        return _DEFAULT_BAD_PATTERNS


# ─────────────────────────────────────────────────────────────────────────────
# Main validation function
# ─────────────────────────────────────────────────────────────────────────────

def validate(
    answer: str,
    citations: list[dict],
    tool_results: list[dict],
    llm=None,
) -> float:
    """
    Score the answer on [0, 1].

    Stores per-check breakdown in validate._last_checks after every call
    so format_confidence_block() can display it without changing the signature.

    Parameters
    ----------
    answer : str
    citations : list[dict]
    tool_results : list[dict]
    llm : ChatOpenAI | None
        Provided → Check D triggered when score < SEMANTIC_TRIGGER_THRESHOLD.
        None     → Check D skipped entirely.

    Returns
    -------
    float — confidence score clamped to [0.0, 1.0]
    """
    score = 0.50
    issues: list[str] = []
    checks: dict[str, dict] = {}

    cited_indices     = _extract_cited_indices(answer)
    available_indices = {c.get("index", 0) for c in citations}

    # ── Check A: citation presence ────────────────────────────────────────
    if cited_indices:
        score += 0.20
        checks["Citations present"] = {"status": "pass", "delta": +0.20, "note": ""}
    else:
        issues.append("No [Source N] citations found in answer")
        checks["Citations present"] = {"status": "fail", "delta": 0.00, "note": "none found"}

    # ── Check A: orphan citations ─────────────────────────────────────────
    orphans = cited_indices - available_indices
    if not orphans:
        score += 0.10
        checks["Valid source indices"] = {"status": "pass", "delta": +0.10, "note": ""}
    else:
        score -= 0.10
        issues.append(f"Orphan citation references: {orphans}")
        checks["Valid source indices"] = {"status": "fail", "delta": -0.10, "note": f"orphans={orphans}"}

    # ── Check A: real data in tool results ────────────────────────────────
    has_data = any(
        isinstance(r, dict) and r.get("data") is not None and r.get("status") != "error"
        for r in tool_results
    )
    if has_data:
        score += 0.10
        checks["Real data returned"] = {"status": "pass", "delta": +0.10, "note": ""}
    else:
        checks["Real data returned"] = {"status": "fail", "delta": 0.00, "note": "all tools errored"}

    # ── Check B: hallucination signal phrases ─────────────────────────────
    lower     = answer.lower()
    triggered = [p for p in HALLUCINATION_PHRASES if p in lower]
    if not triggered:
        score += 0.10
        checks["No uncertainty phrases"] = {"status": "pass", "delta": +0.10, "note": ""}
    else:
        issues.append(f"Uncertainty signals detected: {triggered}")
        checks["No uncertainty phrases"] = {"status": "fail", "delta": 0.00, "note": "signals detected"}

    # ── Check B: bad patterns (from config) ───────────────────────────────
    bad_patterns = _get_bad_patterns()
    bad_hits: list[str] = []
    for pat in bad_patterns:
        if re.search(pat, lower):
            score -= 0.20
            bad_hits.append(pat)
            issues.append(f"Suspicious pattern detected: '{pat}'")
    if bad_hits:
        checks["No suspicious patterns"] = {"status": "fail", "delta": -0.20, "note": str(bad_hits)}
    else:
        checks["No suspicious patterns"] = {"status": "pass", "delta": 0.00, "note": ""}

    # ── Check C: numeric faithfulness (no extra LLM call) ────────────────
    delta_c = check_numeric(answer, tool_results)
    score  += delta_c
    if delta_c > 0:
        checks["Numeric faithfulness"] = {"status": "pass", "delta": delta_c, "note": "all numbers verified"}
    elif delta_c < 0:
        issues.append(
            f"Numeric faithfulness: answer contains number(s) not traceable "
            f"to source data (delta={delta_c:+.2f})"
        )
        checks["Numeric faithfulness"] = {"status": "fail", "delta": delta_c, "note": "number(s) not in source"}
    else:
        checks["Numeric faithfulness"] = {"status": "skip", "delta": 0.00, "note": "no numbers to check"}

    # ── Check D: semantic faithfulness (conditional LLM judge) ────────────
    if llm is not None and score < SEMANTIC_TRIGGER_THRESHOLD:
        logger.info(
            f"Score {score:.2f} below threshold {SEMANTIC_TRIGGER_THRESHOLD} — "
            "triggering semantic faithfulness judge (Check D)"
        )
        delta_d = check_semantic(answer, tool_results, llm)
        score  += delta_d
        if delta_d >= 0:
            checks["Semantic judge"] = {"status": "pass", "delta": delta_d, "note": "triggered · passed"}
        else:
            issues.append(f"Semantic faithfulness judge flagged issues (delta={delta_d:+.2f})")
            checks["Semantic judge"] = {"status": "fail", "delta": delta_d, "note": "triggered · issues found"}
    else:
        reason = f"score already ≥ {SEMANTIC_TRIGGER_THRESHOLD}" if llm is not None else "llm not provided"
        checks["Semantic judge"] = {"status": "skip", "delta": 0.00, "note": f"skipped · {reason}"}

    score = round(max(0.0, min(1.0, score)), 3)

    if issues:
        logger.debug(f"Guardrails issues ({score:.2f}): {issues}")
    else:
        logger.debug(f"Guardrails passed — confidence {score:.2f}")

    validate._last_checks = checks   # type: ignore[attr-defined]
    validate._last_issues = issues   # type: ignore[attr-defined]

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_confidence_block(
    confidence: float,
    citations: list[dict],
    tools_used: list[str],
    latency_s: float = 0.0,
    verbose: bool = True,
) -> str:
    level = "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.5 else "LOW"

    seen: set[str] = set()
    unique_tools: list[str] = []
    for t in tools_used:
        if t not in seen:
            seen.add(t)
            unique_tools.append(t)

    if verbose:
        return _format_verbose(confidence, level, unique_tools, citations, latency_s)
    else:
        return _format_compact(confidence, level, unique_tools, citations, latency_s)


def _format_verbose(
    confidence: float,
    level: str,
    tools: list[str],
    citations: list[dict],
    latency_s: float,
) -> str:
    SEP = "─" * 45
    lines = [
        f"\n{SEP}",
        f"  Confidence   : {confidence:.2f}  {level}",
        f"  Latency      : {latency_s}s",
        f"  Tools        : {', '.join(tools) if tools else 'none'}",
    ]

    if citations:
        lines.append("  Sources cited :")
        for c in citations:
            tool  = c.get("tool", "?")
            asset = c.get("asset_id", "")
            idx   = c.get("index", "?")
            asset_str = f"  ·  {asset}" if asset else ""
            lines.append(f"    [{idx}]  {tool}{asset_str}")

    checks: dict[str, dict] = getattr(validate, "_last_checks", {})
    if checks:
        lines.append("  Guardrail checks :")
        status_icons = {"pass": "✓", "fail": "✗", "skip": "–"}
        for label, info in checks.items():
            icon  = status_icons.get(info["status"], " ")
            delta = info["delta"]
            note  = f"  {info['note']}" if info["note"] else ""
            delta_str = f"{delta:+.2f}" if delta != 0 else " 0.00"
            lines.append(f"    {icon}  {label:<26} {delta_str}{note}")

    lines.append(SEP)
    return "\n".join(lines)


def _format_compact(
    confidence: float,
    level: str,
    tools: list[str],
    citations: list[dict],
    latency_s: float,
) -> str:
    SEP = "─" * 45
    tools_str    = ", ".join(tools) if tools else "none"
    sources_str  = f"{len(citations)} source{'s' if len(citations) != 1 else ''}"

    checks: dict[str, dict] = getattr(validate, "_last_checks", {})
    failed = [label for label, info in checks.items() if info["status"] == "fail"]
    if not failed:
        checks_str = "all checks passed"
    elif len(failed) == 1:
        checks_str = f"warning: {failed[0].lower()}"
    else:
        checks_str = f"warnings: {len(failed)} checks failed"

    line = (
        f"  {level} {confidence:.2f}  ·  {tools_str}"
        f"  ·  {sources_str}  ·  {checks_str}"
        f"  ·  {latency_s}s"
    )
    return f"\n{SEP}\n{line}\n{SEP}"


def _extract_cited_indices(text: str) -> set[int]:
    return {int(m) for m in re.findall(r"\[Source (\d+)\]", text)}
