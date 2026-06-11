"""
main.py  —  Task 3.2: Interactive CLI chatbot

Usage:
    python main.py

Type 'quit' or Ctrl+C to exit.
"""

import logging
import os
import sys
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.rule import Rule

load_dotenv()
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # silence chromadb posthog errors
logging.basicConfig(level=logging.WARNING)
console = Console()


def ensure_vectorstore() -> None:
    """Build vector store if not already on disk — delegates to retrieval.vectorstore."""
    from retrieval.vectorstore import ensure_vectorstore as _ensure
    _ensure(console=console)


def main() -> None:
    console.print(Rule("[bold]Horizon AI Assistant[/bold]"))
    console.print("[dim]Ask questions about wind turbine and solar plant operations.[/dim]")
    console.print("[dim]Type 'quit' or Ctrl+C to exit.[/dim]\n")

    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[bold red]ERROR: OPENAI_API_KEY not set.[/bold red]")
        console.print("Copy .env.example → .env and add your key, then re-run.")
        sys.exit(1)

    ensure_vectorstore()

    from agent.agent import run as agent_run
    from agent.guardrails import format_confidence_block

    while True:
        try:
            question = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q", "bye"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        t0 = time.time()
        try:
            result = agent_run(question)
            latency = round(time.time() - t0, 2)

            console.print(f"\n[bold green]Assistant:[/bold green]\n{result['answer']}")

            # verbose=False → compact one-line bar for conversational use
            meta = format_confidence_block(
                confidence=result["confidence"],
                citations=result["citations"],
                tools_used=result["tools_used"],
                latency_s=latency,
                verbose=False,
            )
            console.print(f"[dim]{meta}[/dim]\n")

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")
            logging.exception("Agent error")


if __name__ == "__main__":
    main()
