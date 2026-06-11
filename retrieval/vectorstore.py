"""
retrieval/vectorstore.py  —  Task 1.2: Embedding & Vector Store

Only the OEM manual is embedded. One ChromaDB collection: manual_docs.

Embedding model: text-embedding-3-small
  - Best cost/quality ratio for retrieval (1536 dims)
  - Outperforms ada-002 on MTEB benchmarks at ~5× lower cost
  - Correct tool for semantic search over prose

Vector store: ChromaDB (PersistentClient)
  - Zero infra, local file persistence
  - Native metadata filtering (fault_code, manufacturer, model)
  - Runs with pip install, no external services needed

Why only one collection:
  - telemetry  → pandas (numeric, needs aggregation)
  - maintenance→ pandas (15 template descriptions, needs exact filtering)
  - assets     → registry dict (50 rows, key-value lookup)
  - manual     → ChromaDB (genuine unstructured prose, 17+ chunks)
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from agent.cache import get_config
from ingestion.pipeline import Document, run_pipeline

load_dotenv()
logger = logging.getLogger(__name__)


def _openai_ef(model: str) -> embedding_functions.OpenAIEmbeddingFunction:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example → .env and add your key."
        )
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=model,
    )


def _doc_id(doc: Document) -> str:
    """Stable unique ID for a manual Document."""
    fc = doc.metadata.get("fault_code") or "overview"
    mfr = doc.metadata.get("manufacturer", "").replace(" ", "_")
    h = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
    return f"manual_{mfr}_{fc}_{h}"


def _clean_meta(meta: dict) -> dict:
    """ChromaDB only accepts str / int / float / bool values."""
    out = {}
    for k, v in meta.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# VectorStore
# ─────────────────────────────────────────────────────────────────────────────

class HorizonVectorStore:
    """
    Thin wrapper around a single ChromaDB collection (manual_docs).
    Exposes index() and query() only.
    """

    def __init__(self, persist_dir: str, embedding_model: str, collection_name: str):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._ef = _openai_ef(embedding_model)
        self._col_name = collection_name
        self._col: chromadb.Collection | None = None

    def _collection(self) -> chromadb.Collection:
        if self._col is None:
            self._col = self._client.get_or_create_collection(
                name=self._col_name,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
        return self._col

    def index(self, docs: list[Document], batch_size: int = 64) -> None:
        """Embed and persist documents. Safe to re-run — skips already-indexed docs."""
        col = self._collection()
        existing_ids = set(col.get(include=[])["ids"])

        to_add = [d for d in docs if _doc_id(d) not in existing_ids]
        if not to_add:
            logger.info(f"'{self._col_name}': all {len(docs)} docs already indexed.")
            return

        logger.info(f"Indexing {len(to_add)} manual chunks into '{self._col_name}' ...")
        for i in range(0, len(to_add), batch_size):
            chunk = to_add[i: i + batch_size]
            col.add(
                ids=[_doc_id(d) for d in chunk],
                documents=[d.page_content for d in chunk],
                metadatas=[_clean_meta(d.metadata) for d in chunk],
            )
        logger.info(f"'{self._col_name}': {col.count()} total documents.")

    def query(
        self,
        query_text: str,
        k: int = 5,
        fault_code: str | None = None,
        manufacturer: str | None = None,
    ) -> list[dict]:
        """
        Semantic search over the manual.

        Parameters
        ----------
        fault_code   : optional exact filter (e.g. "E-1001")
        manufacturer : optional filter (e.g. "Vestas")

        Returns list of dicts: content, metadata, score
        """
        col = self._collection()
        if col.count() == 0:
            logger.warning("Collection is empty — run build_vectorstore() first.")
            return []

        where: dict | None = None
        if fault_code and manufacturer:
            where = {"$and": [
                {"fault_code": {"$eq": fault_code}},
                {"manufacturer": {"$eq": manufacturer}},
            ]}
        elif fault_code:
            where = {"fault_code": {"$eq": fault_code}}
        elif manufacturer:
            where = {"manufacturer": {"$eq": manufacturer}}

        try:
            res = col.query(
                query_texts=[query_text],
                n_results=min(k, col.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Query failed: {e}")
            return []

        results = []
        for content, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            results.append({
                "content": content,
                "metadata": meta,
                "score": round(1 - dist, 4),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Build / load helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_vectorstore(
    data_dir: str | Path = "data",
    config_path: str | Path = "config.yaml",
) -> HorizonVectorStore:
    """
    Run ingestion → embed manual → persist to ChromaDB.
    Safe to re-run: skips already-indexed documents.
    """
    cfg = get_config(str(config_path))
    vs_cfg = cfg["vectorstore"]
    emb_cfg = cfg["embeddings"]

    manual_docs, _, _, _ = run_pipeline(data_dir=data_dir, config_path=config_path)

    vs = HorizonVectorStore(
        persist_dir=vs_cfg["persist_directory"],
        embedding_model=emb_cfg["model"],
        collection_name=vs_cfg["collection"],
    )
    vs.index(manual_docs, batch_size=emb_cfg["batch_size"])
    return vs


def load_vectorstore(config_path: str | Path = "config.yaml") -> HorizonVectorStore:
    """Load an already-built vector store without re-indexing."""
    cfg = get_config(str(config_path))
    return HorizonVectorStore(
        persist_dir=cfg["vectorstore"]["persist_directory"],
        embedding_model=cfg["embeddings"]["model"],
        collection_name=cfg["vectorstore"]["collection"],
    )


def ensure_vectorstore(console=None) -> None:
    """
    Build the vector store if not already on disk (only embeds the manual — ~10s).

    Shared between demo.py and main.py — was previously duplicated verbatim
    in both files.

    Parameters
    ----------
    console : rich.console.Console | None
        If provided, status messages are printed via console.print() with
        rich markup. If None, falls back to plain print().
    """
    import os

    cfg = get_config()
    persist_dir = cfg["vectorstore"]["persist_directory"]

    def _say(msg: str, plain: str | None = None) -> None:
        if console is not None:
            console.print(msg)
        else:
            print(plain if plain is not None else msg)

    if not os.path.exists(persist_dir):
        _say(
            "[bold yellow]Building vector store (first run — embedding OEM manual, ~10s)...[/bold yellow]",
            "Building vector store (first run — embedding OEM manual, ~10s)...",
        )
        build_vectorstore()
        _say("[bold green]Vector store ready.[/bold green]\n", "Vector store ready.\n")
    else:
        _say("[dim]Vector store found — loading...[/dim]\n", "Vector store found — loading...\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    vs = build_vectorstore()
    print("\nSample query — 'gearbox overheating Vestas':")
    for r in vs.query("gearbox overheating Vestas", k=2):
        print(f"  [{r['score']:.3f}] {r['content'][:120]}...")
