"""Offline stress test of the task A part selection with rephrased claim openings (no Apertus calls).

For every task A case the opening that names the source (up to the first clause break after the cue) is replaced:
  SYN  - the same source named with words outside the cue list
  LATE - no opening, the source named at the end ("…, so das Komitee.")
  NONE - no source at all
Each variant is routed again (claim_part + part_pages); outcomes: same part, no part (fallback retrieval), other part.
For the fallback the hybrid top-8 context is rebuilt without the claim's translation (conservative) and checked for
the gold page; for another part, whether that part holds a gold page.

usage: python stress_routing.py <track_2a> <gen|ho> <rows.jsonl (retrieval_miss.py)> <out.json> [cue-module]
"""
import collections
import importlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
SET, ROWS, OUT = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
sys.path.insert(0, str(ROOT))
from tests.test_submission_config import submission_settings  # noqa: E402

DATA = ROOT / ".cache" / ("gen/data" if SET == "gen" else "holdout_2026-09-27")
os.environ.update(submission_settings())
os.environ.update(MOCK_APERTUS="true", BOOKLET_CACHE_DIR=str(ROOT / ".cache" / "gen" / "booklets"),
                  BOOKLET_CACHE_PREBUILT=str(ROOT / ".cache" / "gen" / "none"),
                  QUERY_TRANSLATION_CACHE_DIR=str(ROOT / ".cache" / "gen" / "queries" / "stress_none"))

from src import config  # noqa: E402
from src import part_context  # noqa: E402
from src.inference import ClaimVerificationEngine, attributed_section  # noqa: E402

if len(sys.argv) > 5:  # an alternative cue list to compare: path of a part_context.py copy
    import importlib.util
    _spec = importlib.util.spec_from_file_location("cues_alt", sys.argv[5])
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    claim_part, part_pages = _mod.claim_part, _mod.part_pages
else:
    claim_part, part_pages = part_context.claim_part, part_context.part_pages

SYN = {  # part -> language -> opening that names the source without any listed cue
    "committee": {"de": "Die Befürworter der Vorlage sind der Meinung, dass", "fr": "Les partisans du projet sont d'avis que",
                  "it": "I sostenitori del progetto sono dell'avviso che"},
    "federal_council": {"de": "Die Landesregierung ist der Meinung, dass", "fr": "Le gouvernement est d'avis que",
                        "it": "Il governo è dell'avviso che"},
    "summary": {"de": "Gemäss der Kurzübersicht gilt:", "fr": "D'après l'aperçu,", "it": "Stando alla sintesi,"},
    "legal_text": {"de": "Laut dem Wortlaut der Vorlage gilt:", "fr": "Selon le libellé du projet,",
                   "it": "Secondo il testo del progetto,"},
    "detail": {"de": "Bei einem Ja gilt:", "fr": "En cas de oui,", "it": "Con un sì,"},
}
SYN_B = {  # evaluation only: other wordings, none of them used to extend the cue list
    "committee": {"de": "Die Unterstützer der Initiative finden, dass", "fr": "Les promoteurs du projet jugent que",
                  "it": "I fautori del progetto giudicano che"},
    "federal_council": {"de": "Die Exekutive des Bundes findet, dass", "fr": "L'exécutif fédéral juge que",
                        "it": "L'esecutivo federale giudica che"},
    "summary": {"de": "In den Kernpunkten heisst es:", "fr": "Selon les points essentiels,", "it": "Secondo i punti essenziali,"},
    "legal_text": {"de": "Im Erlasstext steht:", "fr": "Selon la disposition proposée,", "it": "Secondo la disposizione proposta,"},
    "detail": {"de": "Sagt das Volk Ja, dann gilt:", "fr": "Si le peuple dit oui,", "it": "Se il popolo dice sì,"},
}
LATE = {
    "committee": {"de": ", so das Komitee.", "fr": ", selon le comité.", "it": ", secondo il comitato."},
    "federal_council": {"de": ", so der Bundesrat.", "fr": ", selon le Conseil fédéral.", "it": ", secondo il Consiglio federale."},
    "summary": {"de": ", so die Zusammenfassung.", "fr": ", selon le résumé.", "it": ", secondo il riassunto."},
    "legal_text": {"de": ", so der Abstimmungstext.", "fr": ", selon le texte soumis au vote.", "it": ", secondo il testo in votazione."},
    "detail": {"de": ", falls die Vorlage angenommen wird.", "fr": ", si le projet est accepté.", "it": ", se il progetto viene accettato."},
}
_BREAK = re.compile(r"(?:,\s*|:\s*|\s+)(?:dass|que|che|di)\s+|\s+qu[’']|,\s*qu[’']|,\s*|:\s*")
# "Laut der Zusammenfassung hat …", "Selon le texte soumis au vote, …": a prepositional opening ends with the cue
_PREP = re.compile(r"^\s*(?:laut|gem[aä]ss|gem[aä]ß|nach|selon|d.apr[eè]s|secondo|stando)\b", re.I)


def split_opening(claim, part):
    """(opening, body) at the first clause break after the part's cue, or None."""
    pattern = dict(part_context._CLAIM_CUES)[part]
    m = pattern.search(claim[:part_context._CUE_WINDOW])
    if not m:
        return None
    b = _BREAK.search(claim, m.end())
    if _PREP.match(claim) and (not b or b.start() > m.end() + 12):
        end = re.match(r"\w*", claim[m.end():]).end() + m.end()  # finish the cue word ("Zusammenfassung")
        body = claim[end:].lstrip(" ,:")
        return (claim[:end], body) if len(body) > 15 else None
    if not b or b.start() > 140:
        return None
    body = claim[b.end():].strip()
    return (claim[:b.end()], body) if len(body) > 15 else None


def variants(claim, part, lang):
    sp = split_opening(claim, part)
    if not sp:
        return {}
    body = sp[1].rstrip(" .")
    cap = body[0].upper() + body[1:]
    return {"SYN": f"{SYN[part][lang]} {body}.", "SYN_B": f"{SYN_B[part][lang]} {body}.", "LATE": f"{cap}{LATE[part][lang]}", "NONE": f"{cap}."}


def jl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


rows = {r["id"]: r for r in jl(ROWS)}
cases = {c["id"]: c for c in jl(DATA / "cases.jsonl") if "booklet" in c}
engine = ClaimVerificationEngine(strategy="hybrid", prompt_mode="ids")
res, skipped = [], collections.Counter()
for cid in sorted(cases, key=lambda i: (cases[i]["booklet"]["path"], i)):
    c, r = cases[cid], rows[cid]
    claim, vote, lang = c["claim"]["text"], c.get("vote"), c["claim"]["language"]
    part = part_context.claim_part(claim)
    if part is None:
        skipped["no part originally"] += 1
        continue
    vs = variants(claim, part, lang)
    if not vs:
        skipped[f"no clause break ({part})"] += 1
        continue
    data = engine._get_booklet_data(DATA / c["booklet"]["path"])
    parts = engine._booklet_parts(data)
    gold_pages = set(r.get("gold_pages") or [])
    out = {"id": cid, "part": part, "lang": lang, "gold": r["gold"], "orig_gold_in_ctx": r["gold_in_ctx"], "v": {}}
    for name, text in vs.items():
        new = claim_part(text)
        if new == part:
            outcome, gold_ok = "same", r["gold_in_ctx"]
        else:
            found = part_pages(text, vote, parts) if new else None
            if found:
                outcome = "other"
                gold_ok = bool(gold_pages & set(found[1])) if gold_pages else None
            else:
                outcome = "fallback"
                side = attributed_section(text)
                ctx = data["retriever"].retrieve_hybrid(text, top_k=config.DEFAULT_TOP_K, target_vote=vote,
                                                        ensure_sections={side: config.SPEAKER_BOOST} if side else None)
                gold_ok = bool(gold_pages & {p["page_number"] for p in ctx}) if gold_pages else None
        out["v"][name] = {"text": text, "routed": new, "outcome": outcome, "gold_in_ctx": gold_ok}
    res.append(out)
OUT.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")

print(f"{SET}: {len(res)} cases perturbed, skipped {dict(skipped)}")
for name in ("SYN", "SYN_B", "LATE", "NONE"):
    oc = collections.Counter(x["v"][name]["outcome"] for x in res)
    g02 = [x for x in res if x["gold"] in (0, 2)]
    gold = sum(bool(x["v"][name]["gold_in_ctx"]) for x in g02)
    by = collections.defaultdict(collections.Counter)
    for x in res:
        by[x["part"]][x["v"][name]["outcome"]] += 1
    print(f"  {name:4s} same {oc['same']:4d}  fallback {oc['fallback']:4d}  other part {oc['other']:4d}"
          f"  | gold page in context (E/C) {gold}/{len(g02)}  (original {sum(bool(x['orig_gold_in_ctx']) for x in g02)})")
    print("       by part:", {p: dict(v) for p, v in sorted(by.items())})
