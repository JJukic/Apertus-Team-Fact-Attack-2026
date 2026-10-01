"""
Command Line Interface (CLI) for Hack Apertus Track 2A (OST).
Provides claim verification, benchmark evaluation, and strategy comparison.
"""

import sys
from pathlib import Path
from typing import Optional

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.inference import ClaimVerificationEngine
from src.evaluator import BenchmarkEvaluator
from src.download_data import main as download_assets
from src import config

app = typer.Typer(help="Hack Apertus Track 2A (OST) - Voting Booklet NLI & Claim Verification")
console = Console()


@app.command()
def predict(
    claim: str = typer.Option(..., "--claim", "-c", help="Political claim to verify against the booklet"),
    booklet: Path = typer.Option(
        config.BOOKLETS_DIR / "2026-06-14_de.pdf",
        "--booklet",
        "-b",
        help="Path to the official Swiss voting booklet PDF",
    ),
    lang: str = typer.Option("de", "--lang", "-l", help="Language of claim ('de', 'fr', 'it')"),
    strategy: str = typer.Option(
        config.DEFAULT_STRATEGY,
        "--strategy",
        "-s",
        help="Strategy: 'retrieval' (selected passages) or 'full' (full document)",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of passages to retrieve when using retrieval strategy"),
):
    """
    Verify a political claim against a voting booklet (0 = Entailment, 1 = Neutral, 2 = Contradiction).
    """
    if not booklet.exists():
        console.print(f"[bold red]Error:[/bold red] Booklet PDF not found at {booklet}. Run 'download' first.")
        raise typer.Exit(code=1)

    console.print(Panel(f"[bold cyan]Verifying Claim with Apertus[/bold cyan]\n[italic]\"{claim}\"[/italic]", expand=False))
    engine = ClaimVerificationEngine(strategy=strategy)

    result = engine.verify_claim(
        claim=claim,
        booklet_pdf=booklet,
        claim_language=lang,
        strategy=strategy,
        top_k=top_k,
    )

    # Style by label
    color_map = {0: "green", 1: "yellow", 2: "red"}
    label_color = color_map.get(result.label, "white")

    table = Table(title="Verification Result", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Classification", f"[{label_color}]{result.label} — {result.label_name}[/{label_color}]")
    table.add_row("Meaning", config.LABEL_EXPLANATIONS.get(result.label, ""))
    table.add_row(
        "Calibrated Confidences",
        f"Entail (0): {result.p_entail:.2f} | Neutral (1): {result.p_neutral:.2f} | Contra (2): {result.p_contra:.2f}",
    )
    if result.fuzzy_rule:
        table.add_row("Decision Arbiter", f"[bold cyan]{result.fuzzy_rule}[/bold cyan]")
    table.add_row("Reasoning", result.reasoning)
    table.add_row("Strategy", result.strategy)
    table.add_row("Prompt Tokens", str(result.tokens_prompt))
    table.add_row("Total Tokens", str(result.tokens_total))
    table.add_row("Latency", f"{result.latency_ms:.1f} ms")

    console.print(table)

    if result.evidence:
        console.print("\n[bold underline]Supporting Evidence Passages:[/bold underline]")
        for i, ev in enumerate(result.evidence, 1):
            console.print(f"[cyan]Passage {i}:[/cyan] {ev}\n")


@app.command()
def benchmark(
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Path to JSONL benchmark dataset"),
    strategy: str = typer.Option("retrieval", "--strategy", "-s", help="Strategy: 'retrieval' or 'full'"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Limit number of evaluation samples"),
):
    """
    Run evaluation over the official or custom benchmark dataset and output Macro-F1 report.
    """
    evaluator = BenchmarkEvaluator()
    evaluator.evaluate(dataset_path=dataset, strategy=strategy, limit=limit)


@app.command()
def compare(
    limit: Optional[int] = typer.Option(5, "--limit", "-n", help="Number of benchmark samples to compare"),
):
    """
    Compare 'retrieval' vs 'full' document strategies on tokens, latency, and Macro-F1.
    """
    console.print(Panel("[bold]Comparing Strategies: 'Retrieval' vs 'Full Document'[/bold]"))
    evaluator = BenchmarkEvaluator()

    console.print("\n[bold cyan]1. Evaluating 'Retrieval' Strategy...[/bold cyan]")
    rep_ret = evaluator.evaluate(strategy="retrieval", limit=limit)

    console.print("\n[bold cyan]2. Evaluating 'Full Document' Strategy...[/bold cyan]")
    rep_full = evaluator.evaluate(strategy="full", limit=limit)

    table = Table(title="Strategy Comparison Summary", show_header=True, header_style="bold green")
    table.add_column("Metric", style="bold")
    table.add_column("Retrieval (Passages)")
    table.add_column("Full Document")

    table.add_row("Macro-F1", f"{rep_ret['macro_f1']:.4f}", f"{rep_full['macro_f1']:.4f}")
    table.add_row("Avg Input Tokens", f"{rep_ret['avg_prompt_tokens']}", f"{rep_full['avg_prompt_tokens']}")
    table.add_row("Avg Total Tokens", f"{rep_ret['avg_total_tokens']}", f"{rep_full['avg_total_tokens']}")
    table.add_row("Avg Latency", f"{rep_ret['avg_latency_ms']} ms", f"{rep_full['avg_latency_ms']} ms")

    console.print("\n")
    console.print(table)


@app.command()
def download():
    """
    Download benchmark dataset and official voting booklets from admin.ch.
    """
    download_assets()


@app.command()
def web(
    port: int = typer.Option(8501, "--port", "-p", help="Port to run Streamlit on"),
):
    """
    Launch interactive Streamlit web dashboard.
    """
    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", str(_pkg_root / "app.py"), f"--server.port={port}", "--server.address=0.0.0.0"]
    subprocess.run(cmd)


if __name__ == "__main__":
    app()
