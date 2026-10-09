"""
Parallel page extraction for booklets that are not in the parse cache.

pypdf needs ~50 ms per page (1.5-4 s per booklet), and the evaluation set uses booklets that are not baked into the
image. Parsing them one after another in a thread stalls the workers (time without a request in flight is scored)
and holds the GIL. Here the pages are extracted by a process pool: the batch CLI queues all uncached booklets at
start-up in the order of their first case. Task j of k extracts pages j, j+k, j+2k, ... (each task opens the file
once and counts the pages itself, so the main process never parses a PDF). The first booklet is split over all
workers so the first request goes out early; the others use fewer tasks, since every task pays for opening the
file. The text is exactly what PDFParser.extract_pages returns sequentially (same pypdf call per page).

The pool is forked from the main thread before any other thread exists (safe to fork, and workers start in
milliseconds without re-importing anything). Without fork (Windows), with one CPU, or outside the batch CLI,
extraction stays sequential.
"""

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

TASKS_PER_BOOKLET = int(os.getenv("PARSE_TASKS_PER_BOOKLET", "1"))  # after the first booklet
MAX_PROCESSES = 16
# A worker that dies (e.g. killed for memory) loses its task and get() would wait forever: give up and parse sequentially
TASK_TIMEOUT_S = float(os.getenv("PARSE_TASK_TIMEOUT_S", "300"))

_lock = threading.Lock()
_pool = None
_jobs: Dict[str, list] = {}


def _cpu_quota() -> Optional[int]:
    """CPUs allowed by a cgroup v2 quota (`docker run --cpus=N`), which the CPU affinity does not show."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        return None if quota == "max" else max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        return None


def processes() -> int:
    """Worker processes: the CPUs this process may use (affinity and cgroup quota), at most MAX_PROCESSES."""
    env = os.getenv("PARSE_PROCESSES", "").strip()
    if env.isdigit():
        return max(1, int(env))
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        cpus = os.cpu_count() or 1
    return max(1, min(MAX_PROCESSES, cpus, _cpu_quota() or cpus))


def _extract_stride(path: str, j: int, k: int, source: bool = False) -> List[Tuple[int, str, Any]]:
    import pypdf

    from src.pdf_parser import extract_page_text

    reader = pypdf.PdfReader(path)
    numbers = range(j, len(reader.pages), k)
    sources: Dict[int, Any] = {}
    if source:  # evidence pages as PyMuPDF reads them (src/evidence.py), in the same task
        try:
            from src.evidence import extract_source_pages

            sources = extract_source_pages(path, [i + 1 for i in numbers])
        except Exception:
            sources = {}  # the caller reads them sequentially
    return [(i, extract_page_text(reader, i, path), sources.get(i + 1)) for i in numbers]


def _key(pdf_path: Union[str, Path]) -> str:
    return str(Path(pdf_path).resolve())


def start(pdf_paths: Sequence[Union[str, Path]]) -> bool:
    """
    Fork the pool and queue the pages of these booklets in this order. Call from the main thread before starting
    other threads. Returns False (sequential extraction) if there is nothing to do or no pool can be forked.
    """
    global _pool
    if not pdf_paths or processes() < 2 or _pool is not None:
        return False
    try:
        import multiprocessing

        import pypdf  # noqa: F401  imported before forking, so the workers do not import it again

        from src import config

        source = config.EVIDENCE_POLICY != "legacy"
        if source:
            try:
                import pymupdf  # noqa: F401
            except ImportError:
                source = False

        ctx = multiprocessing.get_context("fork")  # ValueError where fork does not exist
        _pool = ctx.Pool(processes())
    except Exception:
        return False
    with _lock:
        for n, pdf in enumerate(pdf_paths):
            key = _key(pdf)
            k = processes() if n == 0 else max(1, min(TASKS_PER_BOOKLET, processes()))
            _jobs[key] = [_pool.apply_async(_extract_stride, (key, j, k, source)) for j in range(k)]
    return True


def stop() -> None:
    global _pool
    if _pool is not None:
        _pool.terminate()
        _pool = None
    _jobs.clear()


def extract_pages(pdf_path: Union[str, Path]) -> Optional[List[Dict[str, Any]]]:
    """Pages as PDFParser.extract_pages returns them, if this booklet was queued in the pool; else None."""
    with _lock:
        jobs = _jobs.get(_key(pdf_path))
    if jobs is None:
        return None
    try:
        pages = sorted(page for job in jobs for page in job.get(timeout=TASK_TIMEOUT_S))
    except Exception:  # also multiprocessing.TimeoutError
        return None  # the caller extracts sequentially
    # "source": the PyMuPDF page for the evidence (None if not read); load_parsed_booklet moves it out of the pages
    return [{"page_number": i + 1, "text": text, "source": source} for i, text, source in pages if text]
