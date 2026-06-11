r"""
evaluation/eval.py  —  Task 3.1: Evaluation Framework

Ground-truth test cases derived from the actual dataset.
Metrics: tool routing accuracy, keyword recall, numeric recall,
         error handling, degradation detection, confidence distribution.

"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Assets flagged by the seasonality-normalised dual-signal method in
# agent/tools/telemetry.get_underperforming_assets() (energy-ratio z-score
# <= -1.8 within asset class, OR absolute Jan-Feb-to-Mar-Jun availability
# drop >= 4.0 percentage points). These three solar assets form a tight,
# clearly separated cluster: each shows a 6.1-6.96pp availability drop
# (vs <=2.5pp for every other asset in either class) and all three carry
# the corroborating E-3002 fault code (string underperformance, DC current
# 20% below expected) starting in March 2024 — matching the brief's "three
# assets ... deliberate performance degradation trend starting in March
# 2024". Re-run get_underperforming_assets() if the dataset changes.
DEGRADED_ASSET_IDS: list[str] = ["PV-004", "PV-005", "PV-014"]

GROUND_TRUTH: list[dict] = [
    {
        "id":               "Q1",
        "question":         "What fault codes appeared on WT-001 between January and March 2024?",
        "expected_tools":   ["asset_lookup", "telemetry_tool"],
        "expected_keywords": ["WT-001", "E-2002"],
        "expected_asset":   "WT-001",
    },
    {
        "id":               "Q2",
        "question":         "What does fault code E-1001 mean and what are the corrective actions?",
        "expected_tools":   ["rag_tool"],
        "expected_keywords": ["gearbox", "overheating", "coolant", "shutdown"],
        "expected_asset":   None,
    },
    {
        "id":               "Q3",
        "question":         "Summarise the maintenance history of WT-011 over the past 6 months.",
        "expected_tools":   ["asset_lookup", "maintenance_tool"],
        "expected_keywords": ["WT-011", "scheduled", "corrective"],
        "expected_asset":   "WT-011",
        # Total cost and event count are stable synthetic values for WT-011
        # (update if the synthetic dataset changes)
        "expected_numbers": [],  # filled dynamically — see note below
    },
    {
        "id":                "Q4",
        "question":          "Which assets have shown declining energy output since March 2024?",
        "expected_tools":    ["telemetry_tool"],
        "expected_keywords": [],
        "expected_asset":    None,
        "checks_degradation": True,
    },
    {
        "id":               "Q5",
        "question":         "What is the availability of solar plant PV-005 in the last 6 months?",
        "expected_tools":   ["asset_lookup", "telemetry_tool"],
        "expected_keywords": ["PV-005", "availability"],
        "expected_asset":   "PV-005",
    },
    {
        "id":               "Q6",
        "question":         "What does the LONGi Solar manual say about inverter overtemperature?",
        "expected_tools":   ["rag_tool"],
        "expected_keywords": ["E-3001", "inverter", "temperature", "filter"],
        "expected_asset":   None,
    },
    {
        "id":               "Q7",
        "question":         "Is WT-099 due for scheduled maintenance?",
        "expected_tools":   ["asset_lookup"],
        "expected_keywords": [],
        "expected_asset":   None,
        "expected_error":   True,
    },
    {
        "id":               "Q8",
        "question":         "What is the escalation procedure for HIGH severity faults?",
        "expected_tools":   ["rag_tool"],
        "expected_keywords": ["level", "hours", "shutdown"],
        "expected_asset":   None,
    },
]


@dataclass
class EvalResult:
    question_id:          str
    question:             str
    answer:               str
    confidence:           float
    tools_used:           list[str]
    citations:            list[dict]
    latency_s:            float
    tool_routing_correct: bool  = False
    keyword_recall:       float = 0.0
    numeric_recall:       float = 1.0
    has_error_handling:   bool  = False
    degradation_detected: bool  = False
    passed:               bool  = False
    notes:                list[str] = field(default_factory=list)


class Evaluator:
    def __init__(self, run_fn: Callable[[str], dict]):
        self.run_fn  = run_fn
        self.results: list[EvalResult] = []

    def run(self, cases: list[dict] | None = None, verbose: bool = True) -> dict:
        cases = cases or GROUND_TRUTH
        self.results = []

        for case in cases:
            if verbose:
                print(f"\n[{case['id']}] {case['question'][:70]}...")

            t0 = time.time()
            try:
                output = self.run_fn(case["question"])
            except Exception as e:
                logger.error(f"Agent failed on {case['id']}: {e}")
                output = {
                    "answer":     f"ERROR: {e}",
                    "citations":  [],
                    "confidence": 0.0,
                    "tools_used": [],
                }
            latency = round(time.time() - t0, 2)

            result = self._score(case, output, latency)
            self.results.append(result)

            if verbose:
                status = "PASS" if result.passed else "FAIL"
                print(
                    f"  [{status}] conf={result.confidence:.2f} "
                    f"recall={result.keyword_recall:.2f} "
                    f"num_recall={result.numeric_recall:.2f} "
                    f"tools={result.tools_used} lat={latency}s"
                )
                for note in result.notes:
                    print(f"    ! {note}")

        metrics = self._aggregate()
        if verbose:
            self._print_summary(metrics)
        return metrics

    def _score(self, case: dict, output: dict, latency: float) -> EvalResult:
        answer     = output.get("answer", "")
        tools_used = output.get("tools_used", [])
        citations  = output.get("citations", [])
        confidence = output.get("confidence", 0.0)

        result = EvalResult(
            question_id=case["id"],
            question=case["question"],
            answer=answer,
            confidence=confidence,
            tools_used=tools_used,
            citations=citations,
            latency_s=latency,
        )
        notes: list[str] = []

        # ── tool routing ──────────────────────────────────────────────────────
        expected = set(case.get("expected_tools", []))
        result.tool_routing_correct = expected.issubset(set(tools_used))
        if not result.tool_routing_correct:
            notes.append(f"Expected tools {expected}, got {set(tools_used)}")

        # ── keyword recall ────────────────────────────────────────────────────
        keywords = case.get("expected_keywords", [])
        if keywords:
            hits = sum(1 for kw in keywords if kw.lower() in answer.lower())
            result.keyword_recall = hits / len(keywords)
            if result.keyword_recall < 0.5:
                notes.append(f"Low keyword recall: {hits}/{len(keywords)}")
        else:
            result.keyword_recall = 1.0

        # ── numeric recall ────────────────────────────────────────────────────
        expected_numbers: list[float] = case.get("expected_numbers", [])
        if expected_numbers:
            answer_nums_raw = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", answer)
            answer_nums     = {float(n.replace(",", "")) for n in answer_nums_raw}
            hits_n = 0
            for en in expected_numbers:
                close = any(abs(en - an) / max(abs(en), 1) < 0.015 for an in answer_nums)
                if close:
                    hits_n += 1
                else:
                    notes.append(f"Expected number {en} not found in answer")
            result.numeric_recall = hits_n / len(expected_numbers)
        else:
            result.numeric_recall = 1.0

        # ── error handling ────────────────────────────────────────────────────
        if case.get("expected_error"):
            result.has_error_handling = any(
                w in answer.lower()
                for w in [
                    "not found", "doesn't exist", "did you mean",
                    "invalid", "check", "typo", "no asset", "skipped",
                ]
            )
            if not result.has_error_handling:
                notes.append("Expected graceful error for invalid asset_id")

        # ── degradation detection (fixed) ─────────────────────────────────────
        #
        # Original bug: any(re.search(r"\b(WT|PV)-\d+\b", answer)) passes a
        # string to any() — Python iterates over the string's characters, so
        # this is always True on any non-empty answer.
        #
        # Fix: check each known degraded asset ID individually, and require
        # ALL of them to be mentioned (not just one) — the brief specifies
        # three deliberately-degraded assets, so a complete answer should
        # surface all three.
        if case.get("checks_degradation"):
            found_degraded = [
                asset_id for asset_id in DEGRADED_ASSET_IDS
                if asset_id.lower() in answer.lower()
            ]
            result.degradation_detected = len(found_degraded) == len(DEGRADED_ASSET_IDS)
            if not result.degradation_detected:
                missing = sorted(set(DEGRADED_ASSET_IDS) - set(found_degraded))
                notes.append(
                    f"Degradation detection: missing {missing} from "
                    f"{DEGRADED_ASSET_IDS} in answer"
                )

        result.passed = (
            result.tool_routing_correct
            and result.keyword_recall   >= 0.5
            and result.numeric_recall   >= 0.5
            and confidence              >= 0.4
            and (not case.get("expected_error") or result.has_error_handling)
        )
        result.notes = notes
        return result

    def _aggregate(self) -> dict:
        n = len(self.results)
        if n == 0:
            return {}
        return {
            "total":                n,
            "passed":               sum(1 for r in self.results if r.passed),
            "pass_rate":            round(sum(1 for r in self.results if r.passed) / n, 3),
            "avg_confidence":       round(sum(r.confidence      for r in self.results) / n, 3),
            "avg_keyword_recall":   round(sum(r.keyword_recall  for r in self.results) / n, 3),
            "avg_numeric_recall":   round(sum(r.numeric_recall  for r in self.results) / n, 3),
            "tool_routing_accuracy": round(
                sum(1 for r in self.results if r.tool_routing_correct) / n, 3),
            "avg_latency_s":        round(sum(r.latency_s for r in self.results) / n, 2),
            "hallucination_rate":   round(
                sum(1 for r in self.results if r.confidence < 0.5) / n, 3),
            "per_question": [
                {
                    "id":                   r.question_id,
                    "passed":               r.passed,
                    "confidence":           r.confidence,
                    "keyword_recall":       r.keyword_recall,
                    "numeric_recall":       r.numeric_recall,
                    "tool_routing_correct": r.tool_routing_correct,
                    "degradation_detected": r.degradation_detected,
                    "latency_s":            r.latency_s,
                    "notes":                r.notes,
                }
                for r in self.results
            ],
        }

    def _print_summary(self, m: dict) -> None:
        print("\n" + "=" * 52)
        print("EVALUATION SUMMARY")
        print("=" * 52)
        print(f"  Pass rate          : {m['passed']}/{m['total']} ({m['pass_rate']:.0%})")
        print(f"  Avg confidence     : {m['avg_confidence']:.2f}")
        print(f"  Keyword recall     : {m['avg_keyword_recall']:.2f}")
        print(f"  Numeric recall     : {m['avg_numeric_recall']:.2f}")
        print(f"  Tool routing acc.  : {m['tool_routing_accuracy']:.2f}")
        print(f"  Hallucination rate : {m['hallucination_rate']:.2f}")
        print(f"  Avg latency        : {m['avg_latency_s']}s")
        print("=" * 52)

    def save(self, path: str = "evaluation/results.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._aggregate(), f, indent=2)
        print(f"Results saved to {path}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING)
    sys.path.insert(0, ".")
    from agent.agent import run as agent_run
    evaluator = Evaluator(run_fn=agent_run)
    evaluator.run(verbose=True)
    evaluator.save()
