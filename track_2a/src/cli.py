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

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
from src.inference import ClaimVerificationEngine, PredictionResult
from src.apertus_client import ApertusClient
from src import config

app = typer.Typer(help="Hack Apertus Track 2A (OST) - Voting Booklet NLI & Claim Verification")
console = Console(legacy_windows=False)


@app.command()
def predict(
    claim: str = typer.Option(..., "--claim", "-c", help="Political claim to verify against the booklet"),
    booklet: Path = typer.Option(
        config.BOOKLETS_DIR / "2026-06-14_de.pdf",
        "--booklet",
        "-b",
        help="Path to the official Swiss voting booklet PDF",
    ),
    vote: Optional[str] = typer.Option(None, "--vote", "-v", help="Specific proposal / vote title in booklet (Advanced Task)"),
    reference: Optional[str] = typer.Option(None, "--reference", "-r", help="Direct reference premise string (Beginner Task)"),
    lang: str = typer.Option("de", "--lang", "-l", help="Language of claim ('de', 'fr', 'it')"),
    strategy: str = typer.Option(
        config.DEFAULT_STRATEGY,
        "--strategy",
        "-s",
        help="Strategy: 'retrieval' (selected passages) or 'full' (full document)",
    ),
    top_k: int = typer.Option(config.DEFAULT_TOP_K, "--top-k", "-k", help="Number of passages to retrieve when using retrieval strategy"),
    json_output: bool = typer.Option(False, "--json", help="Output strictly conforming to official OST JSON format"),
    case_id: Optional[str] = typer.Option("case-0001", "--id", help="Case identifier for official JSON output"),
    mock: bool = typer.Option(False, "--mock", help="Force mock offline model mode"),
):
    """
    Verify a political claim against a voting booklet or direct premise
    (0 = Entailment, 1 = Neutral, 2 = Contradiction).
    """
    client = ApertusClient(mock=True) if mock else None
    engine = ClaimVerificationEngine(strategy=strategy, apertus_client=client)

    if reference:
        # Beginner Task: Direct premise verification
        result = engine.verify_premise(
            claim=claim,
            reference=reference,
            claim_language=lang,
            case_id=case_id,
        )
    else:
        # Advanced Task: Booklet PDF verification
        if not booklet.exists():
            # Check fallback in booklets directory
            fallback = config.BOOKLETS_DIR / booklet.name
            if fallback.exists():
                booklet = fallback
            else:
                console.print(f"[bold red]Error:[/bold red] Booklet PDF not found at {booklet}. Run 'download' first.")
                raise typer.Exit(code=1)

        result = engine.verify_claim(
            claim=claim,
            booklet_pdf=booklet,
            claim_language=lang,
            strategy=strategy,
            top_k=top_k,
            vote=vote,
            case_id=case_id,
        )

    if json_output:
        official_data = result.to_official_dict(case_id=case_id)
        print(json.dumps(official_data, indent=2, ensure_ascii=False))
        return

    console.print(Panel(f"[bold cyan]Verifying Claim with Apertus[/bold cyan]\n[italic]\"{claim}\"[/italic]", expand=False))

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
    if result.numerical_conflict:
        table.add_row("Numerical Check", f"[bold red][CONFLICT] {result.numerical_conflict}[/bold red]")
    table.add_row("Reasoning", result.reasoning)
    table.add_row("Strategy", result.strategy)
    table.add_row("Prompt Tokens", str(result.tokens_prompt))
    table.add_row("Total Tokens", str(result.tokens_total))
    table.add_row("Latency", f"{result.latency_ms:.1f} ms")

    console.print(table)

    if result.evidence_sources:
        console.print("\n[bold underline]Supporting Evidence Passages (with Page Citations):[/bold underline]")
        for i, src in enumerate(result.evidence_sources, 1):
            page_info = f" [bold yellow](Page {src.page_number})[/bold yellow]" if src.page_number else ""
            prop_info = f" [magenta](Vorlage {src.proposal_id})[/magenta]" if src.proposal_id else ""
            console.print(f"[cyan]Passage {i}{page_info}{prop_info}:[/cyan] {src.quote}\n")
    elif result.evidence:
        console.print("\n[bold underline]Supporting Evidence Passages:[/bold underline]")
        for i, ev in enumerate(result.evidence, 1):
            console.print(f"[cyan]Passage {i}:[/cyan] {ev}\n")


@app.command(name="run")
def run_batch(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to input JSON or JSONL file conforming to OST task schema"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional path to output JSON/JSONL file"),
    strategy: str = typer.Option(config.DEFAULT_STRATEGY, "--strategy", "-s", help="Strategy: 'hybrid', 'retrieval' or 'full'"),
    top_k: int = typer.Option(config.DEFAULT_TOP_K, "--top-k", "-k", help="Passages to retrieve"),
    mock: bool = typer.Option(False, "--mock", help="Force mock offline model mode"),
):
    """
    Execute verification across an official OST evaluation file (JSON or JSONL).
    Supports both Beginner Task (direct reference) and Advanced Task (booklet + vote).
    Outputs results strictly adhering to the Hack Apertus Track 2A schema.
    """
    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input_path}")
        raise typer.Exit(code=1)

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    cases = []
    if content.startswith("["):
        cases = json.loads(content)
    elif content.startswith("{") and "\n{" not in content:
        try:
            cases = [json.loads(content)]
        except Exception:
            cases = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        cases = [json.loads(line) for line in content.splitlines() if line.strip()]

    client = ApertusClient(mock=True) if mock else None
    engine = ClaimVerificationEngine(strategy=strategy, apertus_client=client)
    official_results = []

    for idx, item in enumerate(cases, 1):
        cid = item.get("id", f"case-{idx:04d}")

        claim_obj = item.get("claim", "")
        claim_text = claim_obj.get("text", "") if isinstance(claim_obj, dict) else str(claim_obj)

        ref_obj = item.get("reference")
        ref_text = (ref_obj.get("text", "") if isinstance(ref_obj, dict) else str(ref_obj)) if ref_obj else None

        if ref_text:
            res = engine.verify_premise(
                claim=claim_text,
                reference=ref_text,
                case_id=cid,
            )
        else:
            booklet_obj = item.get("booklet", "")
            booklet_path_str = booklet_obj.get("path", "") if isinstance(booklet_obj, dict) else str(booklet_obj)
            booklet_pdf = Path(booklet_path_str) if booklet_path_str else (config.BOOKLETS_DIR / "2026-06-14_de.pdf")

            if not booklet_pdf.exists():
                fallback = config.BOOKLETS_DIR / booklet_pdf.name
                if fallback.exists():
                    booklet_pdf = fallback

            vote_title = item.get("vote")
            res = engine.verify_claim(
                claim=claim_text,
                booklet_pdf=booklet_pdf,
                vote=vote_title,
                strategy=strategy,
                top_k=top_k,
                case_id=cid,
            )

        official_results.append(res.to_official_dict(case_id=cid))

    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_f:
            if str(output_path).endswith(".jsonl"):
                for r in official_results:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                out_payload = official_results if len(official_results) > 1 else official_results[0]
                out_f.write(json.dumps(out_payload, indent=2, ensure_ascii=False) + "\n")
        console.print(f"[bold green]Successfully processed {len(official_results)} case(s) -> {output_path}[/bold green]")
    else:
        out_payload = official_results if len(official_results) > 1 else official_results[0]
        print(json.dumps(out_payload, indent=2, ensure_ascii=False))


@app.command()
def benchmark(
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Path to JSONL benchmark dataset"),
    strategy: str = typer.Option(config.DEFAULT_STRATEGY, "--strategy", "-s", help="Strategy: 'hybrid', 'retrieval' or 'full'"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Label-balanced sample size"),
    task: str = typer.Option("advanced", "--task", "-t", help="'advanced' (booklet PDF) or 'beginner' (reference string)"),
    top_k: int = typer.Option(config.DEFAULT_TOP_K, "--top-k", "-k", help="Passages to retrieve when using retrieval strategy"),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel API requests"),
    seed: int = typer.Option(42, "--seed", help="Sampling seed"),
    tag: str = typer.Option("", "--tag", help="Label appended to the saved results file"),
    mock: bool = typer.Option(False, "--mock", help="Force mock offline model mode"),
    prompt_mode: str = typer.Option(config.PROMPT_MODE, "--prompt-mode", "-p", help="'ids', 'json' or 'compact'"),
):
    """
    Run evaluation over a benchmark (e.g. data/hf/dev.jsonl) and save a Macro-F1 report to results/.
    """
    engine = ClaimVerificationEngine(
        apertus_client=ApertusClient(mock=True) if mock else None,
        prompt_mode=prompt_mode,
    )
    from src.evaluator import BenchmarkEvaluator  # lazy: sklearn is slow to import

    evaluator = BenchmarkEvaluator(engine=engine)
    evaluator.evaluate(
        dataset_path=dataset, strategy=strategy, limit=limit, task=task,
        top_k=top_k, workers=workers, seed=seed, tag=tag,
    )


@app.command()
def compare(
    limit: Optional[int] = typer.Option(5, "--limit", "-n", help="Number of benchmark samples to compare"),
):
    """
    Compare 'retrieval' vs 'full' document strategies on tokens, latency, and Macro-F1.
    """
    console.print(Panel("[bold]Comparing Strategies: 'Retrieval' vs 'Full Document'[/bold]"))
    from src.evaluator import BenchmarkEvaluator  # lazy: sklearn is slow to import

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
    from src.download_data import main as download_assets

    download_assets()


@app.command(name="warm-cache")
def warm_cache(
    booklets_dir: Path = typer.Option(config.BOOKLETS_DIR, "--dir", "-d", help="Folder with booklet PDFs"),
):
    """
    Parse every booklet PDF once and store it in the on-disk cache (used by the Docker build).
    """
    from src.inference import load_parsed_booklet
    from src.pdf_parser import PDFParser

    parser = PDFParser()
    pdfs = sorted(booklets_dir.glob("*.pdf"))
    for pdf in pdfs:
        load_parsed_booklet(pdf, parser)
    console.print(f"[bold green]Cached {len(pdfs)} booklet(s) in {config.BOOKLET_CACHE_DIR}[/bold green]")


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
