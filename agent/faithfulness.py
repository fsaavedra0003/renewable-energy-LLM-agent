"""
agent/faithfulness.py — Numeric and semantic faithfulness checks

Two complementary checks that verify the synthesiser's answer is grounded
in what the tools actually returned.

Check C — numeric faithfulness (no extra LLM call)
    Extracts all significant numbers from the answer text and from the
    tool result payloads, then verifies every answer number is within
    NUMERIC_TOLERANCE of at least one source number.  Uses regex only.
    Adds +0.10 to confidence when all numbers match, -0.20 when any mismatch
    is detected.

Check D — semantic faithfulness (conditional LLM judge)
    A second LLM call that reads the source chunks alongside the answer and
    scores whether every factual claim is supported.  This catches wrong
    conclusions, paraphrased contradictions, and numbers-in-wrong-context
    errors that regex cannot detect.  It is ONLY called when the running
    confidence score is already below SEMANTIC_TRIGGER_THRESHOLD (default 0.6)
    to avoid paying LLM cost on high-confidence answers.
"""

from __future__ import annotations

import json
import logging
import re

from agent.prompts import SEMANTIC_FAITHFULNESS_PROMPT

logger = logging.getLogger(__name__)

NUMERIC_TOLERANCE = 0.015
SEMANTIC_TRIGGER_THRESHOLD = 0.6

_SKIP_IF_YEAR  = lambda n: 2000 <= n <= 2030
_SKIP_IF_SMALL = lambda n: n < 10
_SKIP_NUMBER   = lambda n: _SKIP_IF_YEAR(n) or _SKIP_IF_SMALL(n)


def _extract_numbers(text: str) -> set[float]:
    raw_matches = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", str(text))
    result: set[float] = set()
    for raw in raw_matches:
        try:
            val = float(raw.replace(",", ""))
            if not _SKIP_NUMBER(val):
                result.add(val)
        except ValueError:
            pass
    return result


def _numbers_from_tool_results(tool_results: list[dict]) -> set[float]:
    source_nums: set[float] = set()
    for tr in tool_results:
        data = tr.get("data")
        if data is None:
            continue
        try:
            serialised = json.dumps(data, default=str)
        except Exception:
            serialised = str(data)
        source_nums |= _extract_numbers(serialised)
    return source_nums


def _is_close(answer_num: float, source_nums: set[float]) -> bool:
    for s in source_nums:
        denom = max(abs(s), 1.0)
        if abs(answer_num - s) / denom <= NUMERIC_TOLERANCE:
            return True
    return False


def check_numeric(answer: str, tool_results: list[dict]) -> float:
    """
    Compare numbers in the answer against numbers in tool_results.

    Returns +0.10, -0.20, or 0.0.
    """
    answer_nums  = _extract_numbers(answer)
    source_nums  = _numbers_from_tool_results(tool_results)

    if not answer_nums:
        return 0.0
    if not source_nums:
        return 0.0

    mismatches = [n for n in answer_nums if not _is_close(n, source_nums)]
    if mismatches:
        logger.warning(f"Numeric faithfulness: {len(mismatches)} mismatch(es) — {mismatches}")
        return -0.20

    return 0.10


def check_semantic(
    answer: str,
    tool_results: list[dict],
    llm,
    max_source_chars: int = 6000,
) -> float:
    """
    Call a second LLM to judge whether the answer is grounded in source data.
    Only triggered when running confidence < SEMANTIC_TRIGGER_THRESHOLD.

    Returns +0.10, 0.00, -0.15, or -0.10 (on API failure).
    """
    import json as _json

    source_parts: list[str] = []
    for i, tr in enumerate(tool_results, 1):
        if tr.get("error"):
            continue
        data = tr.get("data")
        if data is None:
            continue
        try:
            serialised = _json.dumps(data, default=str, indent=2)
        except Exception:
            serialised = str(data)
        source_parts.append(f"[Source {i}] {tr.get('tool', 'tool')}:\n{serialised}")

    if not source_parts:
        return 0.0

    source_text = "\n\n".join(source_parts)
    if len(source_text) > max_source_chars:
        source_text = source_text[:max_source_chars] + "\n... [truncated for judge]"

    user_message = (
        f"SOURCE DATA:\n{source_text}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Evaluate faithfulness and return the JSON object as instructed."
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = llm.invoke([
            SystemMessage(content=SEMANTIC_FAITHFULNESS_PROMPT),
            HumanMessage(content=user_message),
        ])
        raw = response.content.strip()

        parsed = _json.loads(raw)
        judge_score: float = float(parsed.get("score", 0.5))
        issues: list[str] = parsed.get("issues", [])

        if issues:
            logger.warning(f"Semantic faithfulness issues: {issues}")

        if judge_score >= 0.8:
            return 0.10
        if judge_score >= 0.5:
            return 0.00
        return -0.15

    except Exception as exc:
        logger.error(f"Semantic faithfulness judge failed: {exc}")
        return -0.10
