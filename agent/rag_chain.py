"""
agent/rag_chain.py  —  Task 2.1: LLM-Powered RAG Chain

Semantic search over the OEM manual only.
Returns a grounded answer where every factual claim cites the source chunk.

"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from agent.cache import get_config
from agent.prompts import RAG_SYSTEM_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_vs = None  # HorizonVectorStore, lazy-loaded


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Copy .env.example → .env and add your key."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def _get_vs():
    global _vs
    if _vs is None:
        from retrieval.vectorstore import load_vectorstore
        _vs = load_vectorstore()
    return _vs


# ─────────────────────────────────────────────────────────────────────────────
# LLM call with retry
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIStatusError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _chat_with_retry(client: OpenAI, model: str, temperature: float,
                     max_tokens: int, messages: list) -> str:
    """Call the OpenAI chat endpoint with exponential backoff on transient errors."""
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def run_rag(
    question: str,
    config_path: str | Path = "config.yaml",
) -> dict:
    """
    Run the RAG chain against the OEM manual.

    Optionally extracts fault_code from the question for targeted retrieval.

    Returns dict: answer, citations, chunks_used, source_scores
      source_scores is a list of {"score": float} dicts — consumed by
      rag_node in agent.py to run check_rag_results().
    """
    cfg = get_config(str(config_path))
    llm_cfg = cfg["llm"]
    k = cfg["vectorstore"]["retrieval_k"]

    vs = _get_vs()

    # try to extract fault code for targeted filter
    fc_match = re.search(r"\b(E-\d+)\b", question, re.IGNORECASE)
    fault_code = fc_match.group(1).upper() if fc_match else None

    # try to detect manufacturer from question
    manufacturer = None
    q_lower = question.lower()
    if "vestas" in q_lower:
        manufacturer = "Vestas"
    elif "siemens" in q_lower or "gamesa" in q_lower:
        manufacturer = "Siemens Gamesa"
    elif "longi" in q_lower or "hi-mo" in q_lower:
        manufacturer = "LONGi Solar"

    chunks = vs.query(
        query_text=question,
        k=k,
        fault_code=fault_code,
        manufacturer=manufacturer,
    )

    if not chunks:
        return {
            "answer": (
                "No relevant information found in the OEM manual for this query. "
                "If you are looking for telemetry or maintenance data, "
                "please use the appropriate structured data tools."
            ),
            "citations": [],
            "chunks_used": 0,
            "source_scores": [],
        }

    # ── build context ─────────────────────────────────────────────────────────
    context_lines = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        label = _source_label(i, meta)
        context_lines.append(f"[Source {i}] {label}\n{chunk['content']}\n")

    context = "\n---\n".join(context_lines)
    user_message = (
        f"Manual chunks:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the chunks above. Cite every fact with [Source N]."
    )

    # ── LLM call with retry ───────────────────────────────────────────────────
    client = _get_client()
    try:
        answer = _chat_with_retry(
            client=client,
            model=llm_cfg["model"],
            temperature=llm_cfg["temperature"],
            max_tokens=llm_cfg["max_tokens"],
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    except Exception as e:
        logger.error(f"RAG LLM call failed after retries: {e}")
        return {
            "answer": f"Failed to generate answer due to API error: {e}",
            "citations": [],
            "chunks_used": len(chunks),
            "source_scores": [{"score": c["score"]} for c in chunks],
        }

    # ── citations ─────────────────────────────────────────────────────────────
    cited_indices = {int(m) for m in re.findall(r"\[Source (\d+)\]", answer)}
    citations = []
    for i, chunk in enumerate(chunks, 1):
        if i in cited_indices:
            citations.append({
                "index": i,
                "doc_type": "manual",
                "fault_code": chunk["metadata"].get("fault_code"),
                "manufacturer": chunk["metadata"].get("manufacturer"),
                "model": chunk["metadata"].get("model"),
                "score": chunk["score"],
                "snippet": chunk["content"][:200] + "...",
            })

    return {
        "answer": answer,
        "citations": citations,
        "chunks_used": len(chunks),
        "source_scores": [{"score": c["score"]} for c in chunks],  # for rag_node validation
    }


def _source_label(index: int, meta: dict) -> str:
    fc = meta.get("fault_code") or "overview"
    mfr = meta.get("manufacturer", "unknown")
    model = meta.get("model", "")
    chunk_type = meta.get("chunk_type", "")
    return f"OEM Manual | {mfr} {model} | {fc} ({chunk_type})"
