"""
agent/agent.py  —  Task 2.2: Tool-Augmented LangGraph Agent

LangGraph StateGraph with one node per tool.

Fixes applied
-------------
1. rag_node — fixed dead-code bug: was checking data.get("source_documents")
   which never exists.  Now reads data.get("source_scores") — the list of
   {"score": float} dicts added to run_rag()'s return value — so
   check_rag_results() actually fires when retrieval quality is low.

2. _infer_date_range_from_question removed — telemetry_node now calls
   infer_date_range from agent.tools.validators (single shared copy, also
   used by maintenance.py). The agent.py version was inconsistent with the
   tool versions on "last month".

3. router_node and synthesiser_node LLM calls wrapped with tenacity retry
   (RateLimitError, APIConnectionError) so a single transient 429 does not
   abort the entire demo run.

4. ChatOpenAI instances promoted to module-level singletons (created once,
   reused across calls) instead of being re-instantiated on every invocation.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Annotated, Literal, TypedDict
import operator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from openai import RateLimitError, APIConnectionError
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent.cache import get_config, get_pipeline_data
from agent.prompts import ROUTER_PROMPT, SYNTHESISER_PROMPT
from agent.tools.validators import (
    check_upstream_asset,
    check_nonempty_result,
    check_rag_results,
    check_date_range,
    extract_all_asset_ids,
    infer_date_range,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level LLM singletons (created once per process)
# ─────────────────────────────────────────────────────────────────────────────

_router_llm = None
_synth_llm  = None
_guard_llm  = None


def _get_router_llm():
    global _router_llm
    if _router_llm is None:
        cfg = get_config()
        _router_llm = ChatOpenAI(
            model=cfg["llm"]["model"],
            temperature=0,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        ).with_structured_output(ToolPlan)
    return _router_llm


def _get_synth_llm():
    global _synth_llm
    if _synth_llm is None:
        cfg = get_config()
        _synth_llm = ChatOpenAI(
            model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
            max_tokens=cfg["llm"]["max_tokens"],
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
    return _synth_llm


def _get_guard_llm():
    global _guard_llm
    if _guard_llm is None:
        cfg = get_config()
        _guard_llm = ChatOpenAI(
            model=cfg["llm"]["model"],
            temperature=0,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
    return _guard_llm


# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question:     str
    tool_plan:    list[str]
    tool_results: Annotated[list[dict], operator.add]
    completed:    list[str]
    answer:       str
    citations:    list[dict]
    confidence:   float


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

class ToolPlan(BaseModel):
    tools: list[Literal["asset_lookup", "telemetry_tool", "rag_tool", "maintenance_tool"]]
    reasoning: str


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=False,
)
def _invoke_router(llm, messages) -> ToolPlan | None:
    return llm.invoke(messages)


def router_node(state: AgentState) -> dict:
    llm = _get_router_llm()
    try:
        plan: ToolPlan = _invoke_router(llm, [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user",   "content": state["question"]},
        ])
        if plan is None:
            raise ValueError("Router returned None after retries")
        logger.info(f"Router → {plan.tools} | {plan.reasoning}")
        return {"tool_plan": plan.tools, "completed": []}
    except Exception as e:
        logger.error(f"Router failed: {e}. Falling back to [telemetry_tool, rag_tool].")
        return {"tool_plan": ["telemetry_tool", "rag_tool"], "completed": []}


def route_to_tools(state: AgentState) -> str:
    """Conditional edge: pick the next unfinished tool, or go to synthesiser."""
    remaining = [t for t in state["tool_plan"] if t not in state.get("completed", [])]
    return remaining[0] if remaining else "synthesiser"


# ─────────────────────────────────────────────────────────────────────────────
# Tool nodes
# ─────────────────────────────────────────────────────────────────────────────

def asset_lookup_node(state: AgentState) -> dict:
    """Validate and resolve asset IDs (supports multiple IDs per question)."""
    from ingestion.pipeline import resolve_asset_id

    try:
        _, registry, _, _ = get_pipeline_data()
    except Exception as e:
        return _err("asset_lookup", f"Pipeline load failed: {e}", state)

    asset_ids = extract_all_asset_ids(state["question"])

    if not asset_ids:
        result = {
            "tool":  "asset_lookup",
            "data":  None,
            "error": "No asset ID pattern (e.g. WT-001, PV-007) found in the question.",
        }
        return {
            "tool_results": [result],
            "completed":    state.get("completed", []) + ["asset_lookup"],
        }

    resolved: list[dict] = []
    errors:   list[str]  = []

    for raw_id in asset_ids:
        asset = resolve_asset_id(raw_id, registry)
        if asset is None:
            errors.append(
                f"Asset '{raw_id}' not found. "
                f"Valid IDs follow the format WT-001–WT-030 or PV-001–PV-020. "
                f"Check for typos."
            )
        else:
            resolved.append(asset)

    if resolved and not errors:
        data = resolved[0] if len(resolved) == 1 else {"assets": resolved}
        result = {"tool": "asset_lookup", "data": data, "status": "ok"}
    elif resolved and errors:
        result = {
            "tool":   "asset_lookup",
            "data":   {"assets": resolved},
            "status": "partial",
            "error":  "; ".join(errors),
        }
    else:
        result = {
            "tool":  "asset_lookup",
            "data":  None,
            "error": "; ".join(errors),
        }

    return {
        "tool_results": [result],
        "completed":    state.get("completed", []) + ["asset_lookup"],
    }


_KNOWN_LOCATIONS = [
    "Northern Spain", "Castile", "Galicia", "Aragon", "Navarre",
    "La Rioja", "Southern Spain", "Andalusia", "Murcia", "Extremadura",
]


def _extract_location(question_lower: str) -> str | None:
    for loc in _KNOWN_LOCATIONS:
        if loc.lower() in question_lower:
            return loc
    return None


def telemetry_node(state: AgentState) -> dict:
    """Structured pandas query on telemetry.csv."""
    from agent.tools.telemetry import query_telemetry, get_underperforming_assets

    upstream_err = check_upstream_asset(state.get("tool_results", []))
    if upstream_err:
        return _err("telemetry_tool", upstream_err, state)

    try:
        q = state["question"].lower()
        is_broad = (
            any(kw in q for kw in ["which assets", "all assets", "underperform",
                                    "degraded", "declining", "consistent"])
            and not re.search(r"\b(WT|PV)-\d+\b", q)
        )

        # Use the single authoritative date-range function from validators
        date_start, date_end = infer_date_range(state["question"])

        date_err = check_date_range(date_start, date_end)
        if date_err:
            return _err("telemetry_tool", date_err, state)

        if is_broad:
            from agent.tools.telemetry import get_low_availability_assets

            wants_availability = any(kw in q for kw in
                ["availability", "below 90", "below 80", "uptime"])
            wants_energy = any(kw in q for kw in
                ["underperform", "declining", "degraded", "output", "energy"])

            is_single_month  = (date_start[:7] == date_end[:7])
            comparison_month = date_start[:7] if is_single_month else None

            if wants_availability and not wants_energy:
                data = get_low_availability_assets(date_start=date_start, date_end=date_end)
            elif wants_availability and wants_energy:
                avail_data  = get_low_availability_assets(date_start=date_start, date_end=date_end)
                location    = _extract_location(q)
                energy_data = get_underperforming_assets(
                    location=location, comparison_month=comparison_month
                )
                data = {
                    "status":           "ok",
                    "low_availability": avail_data,
                    "declining_energy": energy_data,
                }
            else:
                location = _extract_location(q)
                data = get_underperforming_assets(
                    location=location, comparison_month=comparison_month
                )
        else:
            data = query_telemetry(state["question"], state.get("tool_results", []))

        result = {"tool": "telemetry_tool", "data": data, "status": "ok"}

    except Exception as e:
        logger.error(f"telemetry_tool: {e}")
        result = {"tool": "telemetry_tool", "data": None, "error": str(e)}

    return {
        "tool_results": [result],
        "completed":    state.get("completed", []) + ["telemetry_tool"],
    }


def maintenance_node(state: AgentState) -> dict:
    """Structured pandas query on maintenance_logs.csv."""
    from agent.tools.maintenance import query_maintenance

    upstream_err = check_upstream_asset(state.get("tool_results", []))
    if upstream_err:
        return _err("maintenance_tool", upstream_err, state)

    try:
        data   = query_maintenance(state["question"], state.get("tool_results", []))
        result = {"tool": "maintenance_tool", "data": data, "status": "ok"}
    except Exception as e:
        logger.error(f"maintenance_tool: {e}")
        result = {"tool": "maintenance_tool", "data": None, "error": str(e)}

    return {
        "tool_results": [result],
        "completed":    state.get("completed", []) + ["maintenance_tool"],
    }


def rag_node(state: AgentState) -> dict:
    """
    Semantic search on OEM manual via ChromaDB.

    Fix: run_rag() now returns 'source_scores' (list of {"score": float}).
    We pass those to check_rag_results() so the threshold check actually fires.
    Previously the code read 'source_documents' which never existed, making
    check_rag_results() always skip silently.
    """
    from agent.rag_chain import run_rag

    try:
        data = run_rag(state["question"])

        # ── per-tool output validation (FIXED) ────────────────────────────
        source_scores = data.get("source_scores", [])
        rag_err = check_rag_results(source_scores) if source_scores else None

        if rag_err:
            logger.warning(f"rag_tool: {rag_err}")
            result = {"tool": "rag_tool", "data": None, "error": rag_err}
        else:
            result = {"tool": "rag_tool", "data": data, "status": "ok"}

    except Exception as e:
        logger.error(f"rag_tool: {e}")
        result = {"tool": "rag_tool", "data": None, "error": str(e)}

    return {
        "tool_results": [result],
        "completed":    state.get("completed", []) + ["rag_tool"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Synthesiser
# ─────────────────────────────────────────────────────────────────────────────

_LIST_KEYS = (
    "underperforming_assets", "assets", "events", "fault_events",
    "monthly_trend", "fault_summary",
)
_MAX_LIST_ITEMS = 10


def _serialise_tool_data(data: dict, max_chars: int = 8000) -> str:
    import copy, json

    if not isinstance(data, dict):
        return json.dumps(data, default=str)[:max_chars]

    trimmed    = copy.deepcopy(data)
    all_notes: list[str] = []

    for sub_key in ("low_availability", "declining_energy"):
        if sub_key in trimmed and isinstance(trimmed[sub_key], dict):
            sub_trimmed, sub_notes = _trim_lists(trimmed[sub_key])
            trimmed[sub_key] = sub_trimmed
            all_notes.extend(sub_notes)

    trimmed, top_notes = _trim_lists(trimmed)
    all_notes.extend(top_notes)

    serialised = json.dumps(trimmed, indent=2, default=str)
    if len(serialised) > max_chars:
        serialised = serialised[:max_chars] + "\n... [truncated]"

    if all_notes:
        header = "\n".join(all_notes) + "\n\n"
        return header + serialised
    return serialised


def _trim_lists(d: dict) -> tuple[dict, list[str]]:
    import copy
    out = copy.copy(d)
    notes: list[str] = []
    for key in _LIST_KEYS:
        if key in out and isinstance(out[key], list):
            total = len(out[key])
            if total > _MAX_LIST_ITEMS:
                out[key] = out[key][:_MAX_LIST_ITEMS]
                out[f"{key}_showing_top"]  = _MAX_LIST_ITEMS
                out[f"{key}_total_count"]  = total
                label = key.replace("_", " ")
                notes.append(
                    f"⚠ LIST TRUNCATED: showing top {_MAX_LIST_ITEMS} of {total} "
                    f"{label}. You MUST tell the user only the top {_MAX_LIST_ITEMS} "
                    f"are shown and that there are {total} total."
                )
    return out, notes


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _invoke_synth(llm, messages) -> str:
    response = llm.invoke(messages)
    return response.content.strip()


def synthesiser_node(state: AgentState) -> dict:
    import json
    llm = _get_synth_llm()

    context_parts = []
    for i, tr in enumerate(state.get("tool_results", []), 1):
        tool_name = tr.get("tool", f"tool_{i}")
        if tr.get("error"):
            context_parts.append(f"[Source {i}] {tool_name} ERROR: {tr['error']}")
        elif tr.get("data"):
            serialised = _serialise_tool_data(tr["data"])
            context_parts.append(f"[Source {i}] {tool_name}:\n{serialised}")

    context  = "\n\n".join(context_parts)
    user_msg = (
        f"Tool results:\n\n{context}\n\n"
        f"Question: {state['question']}\n\n"
        f"Answer citing every fact with [Source N]. "
        f"Cite [Source N] on every bullet point, not just the first one."
    )

    try:
        answer = _invoke_synth(llm, [
            {"role": "system", "content": SYNTHESISER_PROMPT},
            {"role": "user",   "content": user_msg},
        ])
    except Exception as e:
        logger.error(f"Synthesiser failed: {e}")
        answer = f"Error generating final answer: {e}"

    citations = _extract_citations(answer, state.get("tool_results", []))
    return {"answer": answer, "citations": citations}


# ─────────────────────────────────────────────────────────────────────────────
# Guardrails
# ─────────────────────────────────────────────────────────────────────────────

def guardrails_node(state: AgentState) -> dict:
    """Score the answer for faithfulness and citation quality."""
    from agent.guardrails import validate

    llm = _get_guard_llm()

    confidence = validate(
        state["answer"],
        state["citations"],
        state.get("tool_results", []),
        llm=llm,
    )
    return {"confidence": confidence}


# ─────────────────────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router",           router_node)
    g.add_node("asset_lookup",     asset_lookup_node)
    g.add_node("telemetry_tool",   telemetry_node)
    g.add_node("maintenance_tool", maintenance_node)
    g.add_node("rag_tool",         rag_node)
    g.add_node("synthesiser",      synthesiser_node)
    g.add_node("guardrails",       guardrails_node)

    g.set_entry_point("router")
    g.add_conditional_edges("router", route_to_tools)

    for tool in ["asset_lookup", "telemetry_tool", "maintenance_tool", "rag_tool"]:
        g.add_conditional_edges(tool, route_to_tools)

    g.add_edge("synthesiser", "guardrails")
    g.add_edge("guardrails",  END)

    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(question: str) -> dict:
    """Run the full agent for a question. Returns answer, citations, confidence, tools_used."""
    result = get_graph().invoke({
        "question":     question,
        "tool_results": [],
        "completed":    [],
        "tool_plan":    [],
        "answer":       "",
        "citations":    [],
        "confidence":   0.0,
    })
    return {
        "answer":     result["answer"],
        "citations":  result["citations"],
        "confidence": result["confidence"],
        "tools_used": result.get("tool_plan", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _err(tool_name: str, error: str, state: AgentState) -> dict:
    return {
        "tool_results": [{"tool": tool_name, "data": None, "error": error}],
        "completed":    state.get("completed", []) + [tool_name],
    }


def _extract_citations(answer: str, tool_results: list[dict]) -> list[dict]:
    indices   = {int(m) for m in re.findall(r"\[Source (\d+)\]", answer)}
    citations = []
    for i, tr in enumerate(tool_results, 1):
        if i not in indices:
            continue
        data = tr.get("data") or {}
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        # Score reflects whether the tool actually returned data
        score = 1.0 if (tr.get("data") is not None and not tr.get("error")) else 0.0
        citations.append({
            "index":        i,
            "tool":         tr.get("tool"),
            "doc_type":     tr.get("tool"),
            "asset_id":     data.get("asset_id")     if isinstance(data, dict) else None,
            "fault_code":   data.get("fault_code")   if isinstance(data, dict) else None,
            "manufacturer": data.get("manufacturer") if isinstance(data, dict) else None,
            "score":        score,
            "snippet":      str(tr.get("data", ""))[:200],
        })
    return citations
