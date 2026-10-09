"""
Command Line Interface (CLI) for Hack Apertus Track 2A (OST).
Provides claim verification, benchmark evaluation, and strategy comparison.
"""

import sys
import time
from pathlib import Path

_START = time.perf_counter()
from typing import Any, Dict, List, Optional

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
import re
from src.inference import ClaimVerificationEngine, PredictionResult
from src.apertus_client import ApertusClient
from src import config
from src.text_utils import guess_language
from src.hf_dataset import _download

app = typer.Typer(help="Hack Apertus Track 2A (OST) - Voting Booklet NLI & Claim Verification")
console = Console(legacy_windows=False)
_entrypoint_started = None


def _require_model(engine: ClaimVerificationEngine, mock: bool) -> None:
    """Without an API key the client silently falls back to heuristic mock answers; refuse that unless asked for."""
    if engine.client.mock and not mock and not config.MOCK_APERTUS:
        typer.echo(
            "Error: API_KEY (or LLM_API_KEY) is not set, so no Apertus model can be called. Set it in the environment "
            "(Docker: -e API_KEY) or in .env; use --mock or MOCK_APERTUS=true for an offline mock run.",
            err=True,
        )
        raise typer.Exit(code=2)


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
    _require_model(engine, mock)

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


def resolve_booklet_path(path_str: str, input_dir: Path) -> Path:
    """
    Find the booklet PDF for a case: as given, relative to the input file, or by file name in the booklets folder
    (accepting both '2024_11_24_de.pdf' and '2024-11-24_de.pdf').
    """
    if not path_str:
        raise FileNotFoundError("case has no booklet path")
    given = Path(path_str)
    name = given.name
    names = [name, name.replace("_", "-", 2), re.sub(r"^(\d{4})-(\d{2})-(\d{2})", r"\1_\2_\3", name)]
    candidates = ([given] if given.is_absolute() else [input_dir / given, given]) + [config.BOOKLETS_DIR / n for n in names]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"booklet not found: {path_str}")


def load_cases(input_path: Path) -> List[Dict[str, Any]]:
    """Read cases from JSON (object or list), JSONL, Parquet or CSV (e.g. the OST dataset as published on Hugging Face)."""
    suffix = input_path.suffix.lower()
    if suffix in (".parquet", ".csv"):
        import pandas as pd

        frame = pd.read_parquet(input_path) if suffix == ".parquet" else pd.read_csv(input_path)
        return [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in frame.astype(object).to_dict("records")]
    content = input_path.read_text(encoding="utf-8-sig").strip()  # tolerates a byte order mark
    try:
        parsed = json.loads(content)
        items = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        items = []
        for n, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                # A broken line must not cost the other cases. If its id is readable, it still gets a (neutral) answer:
                # a missing line counts as wrong anyway
                m = re.search(r'"id"\s*:\s*"((?:[^"\\]|\\.)*)"', line)
                print(f"[warning] line {n}: invalid JSON ({exc.msg})" + (", answered as neutral" if m else ", skipped"), file=sys.stderr)
                if m:
                    items.append({"id": json.loads(f'"{m.group(1)}"'), "_invalid": f"invalid JSON: {exc.msg}"})
    cases = [item for item in items if isinstance(item, dict)]
    if len(cases) < len(items):
        print(f"[warning] {len(items) - len(cases)} entries are not JSON objects and were skipped", file=sys.stderr)
    return cases


def _text(value: Any) -> str:
    """A field given as plain text or as {"text": ...}."""
    if isinstance(value, dict):
        return str(value.get("text") or "")
    return "" if value is None else str(value)


def case_booklet(item: Dict[str, Any], input_dir: Path) -> Optional[Path]:
    """
    The booklet PDF a case refers to, or None if it names none. Accepts {"booklet": {"path": ...}}, "booklet_file",
    and the Hugging Face fields "booklet_url" + "booklet_publish_date".
    Booklets must already exist locally; inference never downloads sources.
    """
    booklet = item.get("booklet")
    path_str = (booklet.get("path") or booklet.get("file")) if isinstance(booklet, dict) else booklet
    if path_str or item.get("booklet_file"):
        return resolve_booklet_path(str(path_str or item["booklet_file"]), input_dir)
    url = item.get("booklet_url")
    if not url:
        return None
    m = re.search(r"/dam/(de|fr|it)/", url)
    lang = m.group(1) if m else str(item.get("reference_language") or "de")
    # Files are named after the vote date; the publish date can differ from it, the URL usually carries the vote date
    dates = [str(item.get("booklet_publish_date") or "")[:10]] + re.findall(r"\d{4}-\d{2}-\d{2}", url)
    names = [f"{d}_{lang}.pdf" for d in dates if d]
    for name in names:
        if (config.BOOKLETS_DIR / name).exists():
            return config.BOOKLETS_DIR / name
    raise FileNotFoundError("booklet is unavailable locally; prepare sources before inference")


@app.command(name="run")
def run_batch(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to input JSON or JSONL file conforming to OST task schema"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional path to output JSON/JSONL file"),
    strategy: str = typer.Option(config.DEFAULT_STRATEGY, "--strategy", "-s", help="Strategy: 'hybrid', 'retrieval' or 'full'"),
    top_k: int = typer.Option(config.DEFAULT_TOP_K, "--top-k", "-k", help="Passages to retrieve"),
    task: str = typer.Option("auto", "--task", "-t", help="'auto' (booklet if the case names one, else reference text), 'advanced' or 'beginner'"),
    workers: int = typer.Option(config.BATCH_WORKERS, "--workers", "-w", help="Cases processed in parallel"),
    mock: bool = typer.Option(False, "--mock", help="Force mock offline model mode"),
):
    """
    Execute verification across an evaluation file: JSON, JSONL, Parquet or CSV, in the case format of the README
    or with the fields of the OST dataset on Hugging Face (claim, reference_string, booklet_url, vote).
    Supports both Beginner Task (direct reference) and Advanced Task (booklet + vote).
    Outputs results strictly adhering to the Hack Apertus Track 2A schema.
    """
    global _entrypoint_started
    # The timing below starts before the imports when called via `python -m src` (set by __main__, Josip)
    started = _entrypoint_started if _entrypoint_started is not None else _START
    _entrypoint_started = None  # embedded callers may run several batches in one interpreter
    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input_path}")
        raise typer.Exit(code=1)

    if task not in ("auto", "advanced", "beginner"):
        console.print(f"[bold red]Error:[/bold red] --task must be 'auto', 'advanced' or 'beginner', not '{task}'")
        raise typer.Exit(code=1)
    cases = load_cases(input_path)

    client = ApertusClient(mock=True) if mock else None
    engine = ClaimVerificationEngine(strategy=strategy, apertus_client=client)
    _require_model(engine, mock)

    def process(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        cid = item.get("id", f"case-{idx:04d}")
        try:
            if item.get("_invalid"):
                raise ValueError(item["_invalid"])
            claim_text = _text(item.get("claim"))
            claim_field = item.get("claim")
            given_lang = str((claim_field.get("language") if isinstance(claim_field, dict) else None)
                             or item.get("claim_language") or "").lower()
            claim_lang = given_lang if given_lang in ("de", "fr", "it") else guess_language(claim_text)
            ref_text = _text(item.get("reference")) or _text(item.get("reference_string"))

            booklet_pdf = None
            if task != "beginner":
                booklet_pdf = case_booklet(item, input_path.parent)
            if booklet_pdf is None and task == "advanced":
                raise FileNotFoundError("case has no booklet (booklet.path, booklet_file or booklet_url)")
            if booklet_pdf is None and not ref_text:
                raise ValueError("case has neither a booklet nor a reference text")

            if booklet_pdf is None:
                res = engine.verify_premise(claim=claim_text, reference=ref_text, claim_language=claim_lang, case_id=cid)
            else:
                res = engine.verify_claim(
                    claim=claim_text,
                    booklet_pdf=booklet_pdf,
                    claim_language=claim_lang,
                    booklet_language=(item["booklet"].get("language") if isinstance(item.get("booklet"), dict) else None),
                    vote=_text(item.get("vote")) or None,
                    strategy=strategy,
                    top_k=top_k,
                    case_id=cid,
                )
            return res.to_official_dict(case_id=cid)
        except Exception as exc:
            # One broken case must never cost the whole batch: emit a valid (neutral) record and continue
            print(f"[warning] {cid}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return {
                "id": cid,
                "label": 1,
                "label_name": "neutral",
                "evidence": [],
                "metrics": {"input_tokens": 0, "output_tokens": 0, "inference_time_ms": 0},
            }

    # Cases run in parallel (parsing overlaps with other cases' LLM requests); the output keeps the input order
    # and every case is predicted independently, so results do not depend on case order
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from src.apertus_client import LLM_CLOCK

    from src import page_pool
    from src.inference import is_parse_cached

    # Booklets ordered by their number of cases (then first appearance); cases are processed booklet by booklet in this
    # order, so the workers stay on booklets that are ready while the next ones are parsed. The pages of uncached
    # booklets are queued at once in a process pool (src/page_pool.py, forked here before any other thread starts).
    # Results are written in input order, and the processing order does not change any prediction.
    case_pdfs: List[Optional[Path]] = []
    for item in cases:
        try:
            case_pdfs.append(case_booklet(item, input_path.parent) if task != "beginner" else None)
        except Exception:
            case_pdfs.append(None)  # the case itself reports the problem
    first_seen = {pdf: n for n, pdf in reversed(list(enumerate(case_pdfs))) if pdf is not None}
    n_cases = {pdf: case_pdfs.count(pdf) for pdf in first_seen}
    pdfs = sorted(first_seen, key=lambda pdf: (-n_cases[pdf], first_seen[pdf]))
    rank = {pdf: r for r, pdf in enumerate(pdfs)}
    order = sorted(range(len(cases)), key=lambda i: (rank.get(case_pdfs[i], -1), i))
    page_pool.start([pdf for pdf in pdfs if not is_parse_cached(pdf)])

    def prefetch_booklets() -> None:
        # numpy (via rank_bm25) loads while the first booklet is extracted; then passages and BM25 index booklet by
        # booklet in the same order, in the background
        if pdfs:
            import rank_bm25  # noqa: F401
            if config.EVIDENCE_POLICY != "legacy":
                import pymupdf  # noqa: F401  (evidence pages; loaded here instead of in the first finished case)
        for pdf in pdfs:
            try:
                engine._get_booklet_data(pdf)
            except Exception:
                pass  # the case itself reports the problem

    threading.Thread(target=prefetch_booklets, daemon=True).start()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        done = dict(zip(order, pool.map(lambda i: process(i + 1, cases[i]), order)))
    official_results = [done[i] for i in range(len(cases))]
    page_pool.stop()
    # Our estimate of the scored processing time: wall clock (incl. start-up) minus time with an LLM request in flight
    wall_s = time.perf_counter() - started
    non_llm_ms = (wall_s - LLM_CLOCK.busy_s()) * 1000 / max(1, len(cases))
    if output_path:  # without --output, stdout carries the predictions only
        print(f"[timing] {len(cases)} cases, wall {wall_s:.1f} s, non-LLM {non_llm_ms:.1f} ms/case", file=sys.stderr)

    if output_path:
        if output_path.parent:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_f:
            if config.OFFICIAL_IO or str(output_path).endswith(".jsonl"):
                for r in official_results:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                out_payload = official_results if len(official_results) > 1 else official_results[0]
                out_f.write(json.dumps(out_payload, indent=2, ensure_ascii=False) + "\n")
        console.print(f"[bold green]Successfully processed {len(official_results)} case(s) -> {output_path}[/bold green]")
    else:
        if config.OFFICIAL_IO:
            for prediction in official_results:
                print(json.dumps(prediction, ensure_ascii=False))
        else:
            out_payload = official_results if len(official_results) > 1 else official_results[0]
            print(json.dumps(out_payload, indent=2, ensure_ascii=False))
    # Exit code 0 even if single cases failed: each of them has a valid neutral record, and a non-zero exit could
    # invalidate the whole submission


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
    _require_model(engine, mock)
    if dataset and not dataset.exists() and dataset.parent.resolve() == (config.DATA_DIR / "hf").resolve():
        # The Docker build fetches the OST dataset; if that failed (e.g. offline build), fetch it now
        from src import hf_dataset

        hf_dataset.main()
    if dataset and not dataset.exists():
        console.print(f"[bold red]Error:[/bold red] Dataset not found: {dataset}")
        raise typer.Exit(code=1)
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
    cached = 0
    for pdf in pdfs:
        try:
            load_parsed_booklet(pdf, parser)
            cached += 1
        except Exception as exc:  # a broken download must not fail the build; a code error still does (import time)
            print(f"[warning] {pdf.name}: not cached ({type(exc).__name__}: {exc})", file=sys.stderr)
    console.print(f"[bold green]Cached {cached} of {len(pdfs)} booklet(s) in {config.BOOKLET_CACHE_DIR}[/bold green]")


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
