"""Reproducible adaptive ablations; freeze validation selection before testing.

Run from track_2a (or supply absolute paths). Existing successful calls are
reused only with identical source/configuration hashes. Errors remain errors.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
from importlib.metadata import distributions
from pathlib import Path
import hashlib
import json
import random
import subprocess
import sys
import threading
import time

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src import config
from src.adaptive_calibration import search_neutral_policy
from src.adaptive_engine import AdaptiveVerificationEngine, experiment_settings
from src.adaptive_metrics import language_metrics, nli_metrics
from src.adaptive_retrieval import AdaptiveSettings
from src.embeddings import HybridSettings
from src.evidence_units import EvidenceUnit, expand_neighbors
from src.grounded_apertus import ApertusJSONTransport, NLI_SYSTEM, TRANSLATION_SYSTEM

PROFILES = ("E0", "E1", "E2", "E2r2", "E3", "E4", "E5", "E6", "E6r2", "dense", "RRF", "E8")
COMPLEXITY = dict(zip(PROFILES + ("E7",), (0, 1, 2, 2, 2, 3, 4, 5, 5, 1, 4, 6, 6)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def unit(value):
    # id and page_number are derived aliases emitted by to_dict(), not
    # constructor fields. Keep the original page and paragraph provenance.
    return EvidenceUnit(**{k:v for k,v in value.items() if k not in ("id", "page_number")})


def serialize(prepared):
    result = deepcopy(prepared)
    result["retrieval"]["evidence"] = [u.to_dict() for u in result["retrieval"]["evidence"]]
    return result


def restore(value):
    result = deepcopy(value)
    result["retrieval"]["evidence"] = [unit(u) for u in result["retrieval"]["evidence"]]
    return result


def profile(name, base):
    if name in ("dense", "RRF"):
        return replace(base, ranking="dense" if name == "dense" else "rrf",
                       translated_bm25=name == "RRF", dense=True, neighbor_radius=0)
    return experiment_settings(name.replace("r2", ""), base=base,
                               neighbor_radius=2 if name.endswith("r2") else 1)


def derive_preparation(full, name, settings, units):
    """Derive ablations from one frozen candidate pool, never from gold fields."""
    prepared = deepcopy(full)
    prepared["experiment"] = "E8" if name == "E8" else name
    prepared["settings"] = asdict(settings)
    retrieval = prepared["retrieval"]
    if name == "E0":
        generators, candidates, ranked = {}, [], [u.id for u in units]
    else:
        names = ["dense"] if name == "dense" else ["bm25_original"]
        if settings.translated_bm25:
            names.append("bm25_translated")
        if settings.dense and name != "dense":
            names.append("dense")
        generators = {n:values for n,values in retrieval["generators"].items() if n in names}
        # Preserve the original round-robin order after removing generators.
        ordered = list(dict.fromkeys(item["id"] for r in range(settings.candidate_k)
                    for values in generators.values() for item in values[r:r+1]))
        by_id = {c["unit"]["id"]:c for c in retrieval["candidates"]}
        candidates = []
        for identifier in ordered:
            candidate = deepcopy(by_id[identifier])
            candidate["ranks"] = {n:r for n,r in candidate["ranks"].items() if n in generators}
            candidate["scores"] = {n:s for n,s in candidate["scores"].items() if n in generators}
            candidates.append(candidate)
        if settings.ranking == "nli":
            ranked = [c["unit"]["id"] for c in sorted(candidates,
                      key=lambda c:(-c["reranker_score"], c["unit"]["order"]))]
        elif settings.ranking == "rrf":
            ranked = [c["unit"]["id"] for c in sorted(candidates,
                      key=lambda c:(-sum(1/(60+r) for r in c["ranks"].values()), c["unit"]["order"]))]
        elif settings.ranking in ("bm25", "dense"):
            generator = "dense" if settings.ranking == "dense" else "bm25_original"
            ranked = [item["id"] for item in generators[generator]]
        else:
            ranked = ordered
    by_id = {u.id:u for u in units}
    selected = ranked if name == "E0" else ranked[:settings.evidence_k]
    expanded = expand_neighbors([by_id[i] for i in selected], units, settings.neighbor_radius)
    best = next((c for c in candidates if ranked and c["unit"]["id"] == ranked[0]), None)
    second = next((c for c in candidates if len(ranked)>1 and c["unit"]["id"] == ranked[1]), None)
    retrieval.update(generators=generators, candidates=candidates, ranked_ids=ranked,
        selected_ids=selected, expanded_ids=[u.id for u in expanded], evidence=expanded,
        signals={"best_reranker_score":best["reranker_score"] if best else 0,
                 "reranker_margin":best["reranker_score"]-second["reranker_score"] if best and second else 0,
                 "retrieval_agreement":len(best["ranks"]) if best else 0},
        reranking_ms=retrieval["reranking_ms"] if settings.ranking == "nli" else 0)
    return prepared


def summaries(rows, expected):
    import numpy as np
    metric = language_metrics(rows)
    metric["overall"]["eligible_for_selection"] &= len(rows) == expected
    costs = [r["cost"] for r in rows if r.get("label") is not None]
    def statistics(key):
        values = [c[key] for c in costs if c.get(key) is not None]
        return {"measured_cases":len(values), "mean":float(np.mean(values)) if values else None,
                "p50":float(np.percentile(values,50)) if values else None,
                "p95":float(np.percentile(values,95)) if values else None}
    return {"nli":metric, "efficiency":{"input_tokens":statistics("input_tokens"),
            "latency_ms":statistics("total_latency_ms")}, "expected_cases":expected}


def calibrate(rows):
    if not nli_metrics(rows)["eligible_for_selection"]:
        raise ValueError("Calibration requires complete successful validation predictions")
    policy, grid = search_neutral_policy([{"split":"validation", "gold":r["entailment_label"],
        "label":r["label"], "probabilities":r["decision"]["probabilities"],
        "signals":r["retrieval"]["signals"]} for r in rows])
    calibrated = deepcopy(rows)
    for row in calibrated:
        row["experiment"] = "E7"
        row["neutral_policy"] = policy.to_dict()
        row["label"] = policy.apply(row["raw_label"], row["decision"]["probabilities"], row["retrieval"]["signals"])
        row["decision"]["label"] = row["label"]
        row["official"]["label"] = row["label"]
        row["label_name"] = config.LABEL_MAPPING[row["label"]]
        row["official"]["label_name"] = row["label_name"].lower()
        if row["label"] == 1:
            row["decision"]["evidence_ids"] = row["decision"]["evidence"] = []
            row["official"]["evidence"] = []
    return policy, grid, calibrated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "predict", "run", "summarize", "freeze"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--source-commit", help="Base Git commit for a source snapshot transferred to a cluster")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / "selection.json"
    if args.split == "test" and not lock.exists():
        raise ValueError("Freeze validation selection before accessing held-out evaluation")
    source = BASE / f"data/evaluation/adaptive/{args.split}.jsonl"
    split_manifest = read(source.parent / "manifest.json")
    if digest(source) != split_manifest["splits"][args.split]["sha256"]:
        raise ValueError("Dataset split changed")
    cases = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    folder = args.output / args.split
    folder.mkdir(exist_ok=True)
    sources = list((BASE / "src").glob("adaptive*.py")) + [BASE/"src"/n for n in
        ("evidence_units.py", "grounded_apertus.py", "embeddings.py", "config.py", "apertus_client.py")]
    sources.append(Path(__file__))
    settings = AdaptiveSettings(device=args.device)
    dense = HybridSettings(device=args.device, local_files_only=True)
    signature = {"sources":{str(p.relative_to(BASE)):digest(p) for p in sources},
        "split_sha256":digest(source), "model":config.LLM_NAME,
        "settings":asdict(settings), "dense":json.loads(json.dumps(asdict(dense), default=str)),
        "profiles":args.profiles, "seed":42, "output_token_caps":{}}
    if args.split == "test":
        selection = read(lock)
        if signature["sources"] != selection["sources"] or signature["model"] != selection["model"]:
            raise ValueError("Sources/model differ from frozen validation selection")
        args.profiles = list(dict.fromkeys([selection["profile"].replace("E7","E6"), "E1"]))
        signature["profiles"] = args.profiles
        if signature["settings"] != selection["settings"] or signature["dense"] != selection["dense"]:
            raise ValueError("Runtime configuration differs from frozen selection")
    manifest_path = folder / "run.json"
    if manifest_path.exists():
        if read(manifest_path)["signature"] != signature:
            raise ValueError("Cannot resume with changed sources/configuration")
    else:
        write(manifest_path, {"signature":signature,
            "git_commit":args.source_commit or subprocess.check_output(["git","rev-parse","HEAD"],cwd=BASE,text=True).strip(),
            "dependencies":{d.metadata["Name"]:d.version for d in distributions()},
            "selection_rule":"maximum validation macro_f1; ties: fewer components, then profile name",
            "nli_prompt":NLI_SYSTEM, "translation_prompt":TRANSLATION_SYSTEM})
        for p in sources:
            destination = folder / "frozen_sources" / p.relative_to(BASE)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(p.read_bytes())
    journal_lock = threading.Lock()
    def transport(case_id, variant):
        def journal(event):
            with journal_lock, (folder / "requests.api.jsonl").open("a") as handle:
                handle.write(json.dumps({**event,"case_id":case_id,"profile":variant},ensure_ascii=False)+"\n")
                handle.flush()
        return ApertusJSONTransport(journal=journal)
    random.seed(42)
    def predict(case, name):
        path = folder / "predictions" / name / f"{case['id']}.json"
        if path.exists() and (read(path).get("label") is not None or not args.retry_errors):
            return case["id"],name,"cached"
        full = restore(read(folder / "prepared" / f"{case['id']}.json"))
        units = [unit(u) for u in read(folder/"documents"/f"{full['parsing']['document_sha256']}.json")["units"]]
        prepared = derive_preparation(full,name,profile(name,settings),units)
        engine = AdaptiveVerificationEngine(transport=transport(case["id"],name),dense_settings=dense)
        start = time.perf_counter()
        try:
            result = engine.verify(case["claim"],full["booklet"],case_id=case["id"],
                experiment=prepared["experiment"],prepared=prepared)
            result["experiment"] = name
        except Exception as error:
            result = {"id":case["id"],"experiment":name,"label":None,
                "error_type":type(error).__name__,"cost":{"api_requests":engine.transport.requests,
                "input_tokens":engine.transport.input_tokens,"output_tokens":engine.transport.output_tokens,
                "total_latency_ms":(time.perf_counter()-start)*1000}}
        # Scoring metadata is attached only after prediction.
        result.update(split=args.split,entailment_label=case["entailment_label"],
            claim_language=case["claim_language"],booklet_language=case["booklet_language"],claim=case["claim"])
        write(path,result)
        return case["id"],name,result.get("label")

    if args.stage in ("prepare", "run"):
        import torch
        torch.manual_seed(42)
        engine = AdaptiveVerificationEngine(transport=transport("preparation","union"),dense_settings=dense)
        def preparations():
            for index, case in enumerate(cases):
                path = folder / "prepared" / f"{case['id']}.json"
                booklet = BASE / f"data/booklets/{case['booklet_date']}_{case['booklet_language']}.pdf"
                if path.exists():
                    if read(path)["parsing"]["document_sha256"] != digest(booklet):
                        raise ValueError("Prepared PDF changed")
                    yield case
                    continue
                with redirect_stdout(sys.stderr):
                    full = engine.prepare(case["claim"], booklet, claim_language=case["claim_language"],
                        booklet_language=case["booklet_language"],experiment="E5",settings=profile("E5",settings))
                    units, parsing = engine.document(booklet,case["booklet_language"])
                document_path = folder / "documents" / f"{parsing['document_sha256']}.json"
                if not document_path.exists():
                    write(document_path,{"parsing":parsing,"units":[u.to_dict() for u in units]})
                write(path,serialize(full))
                print(f"prepared {index+1}/{len(cases)} {case['id']}",flush=True)
                yield case
        if args.stage == "prepare":
            for _ in preparations():
                pass
        else:
            # API inference starts as soon as a case is prepared. Measure the
            # adaptive pipeline and BM25 first, then the remaining ablations.
            priority = [n for n in ("E6", "E1") if n in args.profiles]
            jobs = []
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for case in preparations():
                    for name in priority:
                        jobs.append(executor.submit(predict, case, name))
                for name in args.profiles:
                    if name not in priority:
                        jobs.extend(executor.submit(predict, c, name) for c in cases)
                for index, future in enumerate(as_completed(jobs)):
                    print(f"prediction {index+1}/{len(jobs)} {future.result()}", flush=True)
        return
    if args.stage == "predict":
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            jobs = [executor.submit(predict,c,n) for c in cases for n in args.profiles]
            for index, future in enumerate(as_completed(jobs)):
                print(f"prediction {index+1}/{len(jobs)} {future.result()}",flush=True)
        return
    rows = {}
    for name in args.profiles:
        rows[name] = [read(folder/"predictions"/name/f"{c['id']}.json") if
            (folder/"predictions"/name/f"{c['id']}.json").exists() else
            {**{k:c[k] for k in ("id","claim_language","booklet_language","entailment_label")},"label":None}
            for c in cases]
    if "E6" in rows and nli_metrics(rows["E6"])["eligible_for_selection"]:
        if args.split == "validation":
            policy, grid, rows["E7"] = calibrate(rows["E6"])
            write(folder/"neutral_policy.json",policy.to_dict())
            write(folder/"threshold_search.json",grid)
        elif selection["profile"] == "E7":
            from src.adaptive_calibration import NeutralPolicy
            policy = NeutralPolicy(**selection["neutral_policy"])
            rows["E7"] = deepcopy(rows["E6"])
            for row in rows["E7"]:
                row["experiment"] = "E7"
                row["neutral_policy"] = policy.to_dict()
                row["label"] = policy.apply(row["raw_label"],row["decision"]["probabilities"],row["retrieval"]["signals"])
                row["decision"]["label"] = row["official"]["label"] = row["label"]
                row["label_name"] = config.LABEL_MAPPING[row["label"]]
                row["official"]["label_name"] = row["label_name"].lower()
                if row["label"] == 1:
                    row["decision"]["evidence_ids"] = row["decision"]["evidence"] = []
                    row["official"]["evidence"] = []
    summary = {name:summaries(values,len(cases)) for name,values in rows.items()}
    write(folder/"results.json",summary)
    for name, values in rows.items():
        (folder/f"{name}.cases.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in values))
    lines = [f"# Adaptive comparison — {args.split}","", "| Profile | Macro-F1 | Coverage |", "|---|---:|---:|"]
    for name, metrics in summary.items():
        overall = metrics["nli"]["overall"]
        score = f"{overall['macro_f1']:.4f}" if overall["macro_f1"] is not None else "pending"
        lines.append(f"| {name} | {score} | {overall['coverage']:.1%} |")
    lines += ["", "Scores with incomplete coverage cannot select the final configuration.",
              "Validation scores select configurations; only the frozen test estimates held-out quality."]
    (folder/"results.md").write_text("\n".join(lines)+"\n")
    if args.stage == "freeze":
        if args.split != "validation" or set(args.profiles) != set(PROFILES):
            raise ValueError("Selection requires the complete validation ablation suite")
        if not all(m["nli"]["overall"]["eligible_for_selection"] for m in summary.values()):
            raise ValueError("Complete all validation predictions before selection")
        best = min(summary,key=lambda n:(-summary[n]["nli"]["overall"]["macro_f1"],COMPLEXITY[n],n))
        frozen = {**signature,"profile":best,"validation_macro_f1":summary[best]["nli"]["overall"]["macro_f1"],
            "results_sha256":digest(folder/"results.json"),
            "neutral_policy":read(folder/"neutral_policy.json") if best == "E7" else None}
        if lock.exists() and read(lock) != frozen:
            raise ValueError("Selection already frozen; do not adapt it to test results")
        write(lock,frozen)
        print(json.dumps({"selected":best,"validation_macro_f1":frozen["validation_macro_f1"]}))
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
