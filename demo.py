"""
demo.py  —  Task 3.2: Non-interactive demo script

Runs 5 pre-set questions through the full agent pipeline.
Vector store is built on first run (~10s — only 17 manual chunks to embed).
Subsequent runs load from persisted ChromaDB.

Usage:
    python demo.py
"""

import logging
import os
import sys
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # silence chromadb posthog errors
logging.basicConfig(level=logging.WARNING)
console = Console()

DEMO_QUESTIONS = [
    {
        "id":       "Q1",
        "label":    "Cluster underperformance",
        "question": (
            "Which turbines in the Northern Spain cluster underperformed "
            "last month, and what were the main fault codes?"
        ),
    },
    {
        "id":       "Q2",
        "label":    "Asset maintenance check",
        "question": (
            "Is turbine WT-011 due for scheduled maintenance? "
            "Has it shown any anomalous readings recently?"
        ),
    },
    {
        "id":       "Q3",
        "label":    "OEM manual fault lookup",
        "question": (
            "What does the manufacturer manual say about fault code "
            "E-1001 on a Vestas V150 turbine?"
        ),
    },
    {
        "id":       "Q4",
        "label":    "Maintenance history summary",
        "question": "Summarise the maintenance history of asset PV-019 over the past 6 months.",
    },
    {
        "id":       "Q5",
        "label":    "Degradation detection",
        "question": (
            "Which assets have shown consistent availability below 90% "
            "or declining energy output since March 2024?"
        ),
    },
]


def ensure_vectorstore() -> None:
    """Build vector store if not already on disk — delegates to retrieval.vectorstore."""
    from retrieval.vectorstore import ensure_vectorstore as _ensure
    _ensure(console=console)


def run_demo() -> None:
    console.print(Rule("[bold]Horizon AI — Demo[/bold]"))
    console.print(
        "[dim]Renewable energy asset intelligence | "
        "manual → ChromaDB | telemetry + maintenance → pandas[/dim]\n"
    )

    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[bold red]ERROR: OPENAI_API_KEY not set.[/bold red]")
        console.print("Copy .env.example → .env and add your key, then re-run.")
        sys.exit(1)

    ensure_vectorstore()

    from agent.agent import run as agent_run
    from agent.guardrails import format_confidence_block

    for i, demo in enumerate(DEMO_QUESTIONS, 1):
        console.print(Rule(f"[cyan]Q{i} — {demo['label']}[/cyan]"))
        console.print(f"[bold]{demo['question']}[/bold]\n")

        t0 = time.time()
        try:
            result = agent_run(demo["question"])
        except Exception as e:
            console.print(f"[red]Agent error: {e}[/red]")
            continue
        latency = round(time.time() - t0, 2)

        console.print(Panel(result["answer"], border_style="green", padding=(1, 2)))

        # verbose=True → full card with guardrail breakdown for reviewers
        meta = format_confidence_block(
            confidence=result["confidence"],
            citations=result["citations"],
            tools_used=result["tools_used"],
            latency_s=latency,
            verbose=True,
        )
        console.print(f"[dim]{meta}[/dim]\n")

    console.print(Rule("[bold green]Demo complete[/bold green]"))


if __name__ == "__main__":
    run_demo()
