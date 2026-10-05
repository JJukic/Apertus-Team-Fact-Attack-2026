# Optionales Hybrid Retrieval mit BGE-M3

## Anpassung des Auftrags an diesen Checkout

Vor dieser Änderung existierten nur `retrieval` und `full`, **keine** BM25-Strategie
`hybrid` mit Claim-/Titel-Gewichtung. Diese wird als lexikalische Vergleichsbasis
ergänzt. Der Apertus-Client verwendet einen JSON-Prompt mit wörtlichen Zitaten,
keinen separaten `ids`-Prompt. Dieser Prompt bleibt unverändert: `[Page N]` bezeichnet
die physische, 1-basierte PDF-Seite, niemals die Position in der Trefferliste.

Der Parser liefert Absatz-ID, `page_number`, `proposal_id` und `section_type`.
Abschnittslabels sind heuristisch; verlässliche Sprecherlabels fehlen derzeit.
Vorhandene Sprecher-Metadaten werden übernommen, fehlende nicht erfunden.
Beginner Task (`--reference`), bisherige Methoden und offizielles JSON bleiben erhalten.
Standard bleibt `NLI_STRATEGY=retrieval`, solange kein Qualitätsvorteil gemessen ist.

## Methoden und Installation

| Strategie | Kontext für Apertus | Suche |
|---|---|---|
| `retrieval` | bisherige Top-k-Absätze, Standard 5 | bestehendes Proposal-Aware BM25 und Empfehlung-Anker |
| `full` | vollständiger extrahierter Text | keine Passagensuche |
| `hybrid` | bis zu 10 Originalseiten | normalisiertes BM25(Claim) + 2 × normalisiertes BM25(Vorlagentitel) |
| `hybrid_dense` | bis zu 10 Originalseiten | dieselbe BM25-Liste + BGE-M3-Cosinus-Liste, fusioniert mit RRF |

Die neuen Methoden suchen ausschließlich im übergebenen Büchlein, ohne harte
Vorlagen-/Sprecherfilter. `--vote` ist ein optionaler echter Eingabe-Vorlagentitel.
Ohne Titel entfällt dessen Gewicht. Gold-Referenzen und Labels werden nicht verwendet.
`hybrid` und `hybrid_dense` teilen Kontextgröße und Quellenprüfung, damit ihr Vergleich
die Dense-Ergänzung untersucht. Ein Vergleich gegen `retrieval` ändert außerdem
Vorlagenfilterung und Kontextgröße und ist daher keine isolierte Dense-Ablation.

Im Repository-Hauptverzeichnis:

```bash
source .venv/bin/activate
make install
make install-hybrid
make web
```

Die optionale `requirements-hybrid.txt` installiert Sentence Transformers und dessen
Abhängigkeiten, darunter PyTorch, Transformers und Hugging Face Hub. BM25 und `full`
brauchen diese Installation **nicht**. NumPy ist eine explizite Basisabhängigkeit.
Das Modell wird nur von `hybrid_dense` oder dem Indexierungsbefehl geladen.

Im Dashboard unter **Erweiterte Kontext-Einstellungen** die Hybrid-Vektorsuche
auswählen und optional einen Vorlagentitel eingeben. Die Auswahl allein lädt keine
Gewichte; die Prüfung startet den Suchpfad. Beim Methoden-/Büchleinwechsel wird das
vorige Ergebnis gelöscht. Konfigurationsänderungen erfordern einen App-Neustart.

CLI-Befehle aus `track_2a/`:

```bash
cd track_2a

# Modell und Index aufwärmen: kein Apertus-Aufruf.
python -m src index --booklet data/booklets/2026-06-14_de.pdf
python -m src index --all

# Trefferseiten/Zeiten ansehen: kein Apertus-Aufruf.
python -m src search \
  --claim "L'initiative demande de limiter la population résidante permanente." \
  --booklet data/booklets/2026-06-14_de.pdf \
  --strategy hybrid_dense --show-context

# Einzelprüfung mit bestehendem Apertus-Client.
python -m src predict \
  --claim "Die Initiative verlangt eine Begrenzung der Wohnbevölkerung." \
  --vote "Nachhaltigkeitsinitiative" --strategy hybrid_dense --json
```

`make index INDEX_ARGS="--all"` funktioniert auch vom Hauptverzeichnis.
`--top-k` behält seine Bedeutung für `retrieval`; Hybrid verwendet die separate
Seitenzahl unten. `--mock` ersetzt nur Apertus, **nicht** das Embedding-Modell.

## Konfiguration

Die erste vorhandene Datei wird geladen: `track_2a/.env.local`, `track_2a/.env`,
`.env.local`, `.env`. Bereits gesetzte Shell-Variablen haben Vorrang.
Beide `.env.example` enthalten diese Optionen:

| Variable | Standard | Bedeutung |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | lokales Embedding-Modell |
| `EMBEDDING_REVISION` | `5617a9f61b028005a4858fdac845db406aefb181` | festgelegter Hugging-Face-Commit |
| `EMBEDDING_DEVICE` | `cpu` | alternativ passende `cuda`-/`mps`-Installation |
| `EMBEDDING_BATCH_SIZE` | `4` | Absätze je Batch |
| `EMBEDDING_MAX_LENGTH` | `2048` | Tokenlimit je Embedding-Eingabe |
| `EMBEDDING_LOCAL_FILES_ONLY` | `false` | `true` verhindert Modelldownloads |
| `EMBEDDING_CACHE_DIR` | `track_2a/.cache/embeddings` | Dokumentvektoren; absoluter Pfad empfohlen |
| `HYBRID_CANDIDATES` | `40` | maximale eindeutige Seiten je Trefferliste |
| `HYBRID_FINAL_PAGES` | `10` | maximale Originalseiten für Apertus |
| `HYBRID_RRF_K` | `60` | positive RRF-Konstante |
| `HYBRID_DENSE_FALLBACK` | `false` | ausdrücklich erlaubter Fallback auf `hybrid` |

Kandidatenzahl muss mindestens der finalen Seitenzahl entsprechen. Modell und
Revision müssen zusammenpassen. Für reproduzierbare Experimente einen unveränderlichen
40-stelligen Commit statt eines beweglichen Branches wie `main` verwenden; der reale Backend-Loader erzwingt dies.
BGE-M3 unterstützt längere Eingaben; das CPU-Startlimit von 2048 reduziert den Aufwand.
Längere Absätze werden **für das Embedding** abgeschnitten; Originalseiten für Apertus
bleiben vollständig. Wir verwenden nur Dense-Vektoren, keine BGE-Sparse-/ColBERT-Funktion.

## Ranking und Quellen

1. BM25 arbeitet über alle Parser-Absätze des einen Büchleins. Claim- und Titelscores
   werden separat min-max auf `[0, 1]` normalisiert; konstante Listen werden null.
   Absatzscore: `claim_score + 2 * title_score`.
2. Absatz- und Claim-Vektoren werden L2-normalisiert. Exakte lokale Vektorsuche:
   `cosine = document_vectors @ query_vector`. Bei diesen kleinen Dokumenten ist
   kein externer Vektorserver oder approximativer ANN-Index erforderlich.
3. **Vor** Kandidatenbegrenzung und Fusion erhält jede Seite je Kanal den maximalen
   Absatzscore. Scores werden nicht summiert: viele Absätze erzeugen keine mehrfachen
   Stimmen. Mehr Absätze können dennoch die Chance auf einen hohen Maximalwert erhöhen.
4. Pro Kanal werden bis zu 40 eindeutige Seiten ausgewählt. Eine vollständig
   signalfreie lexikalische Liste bleibt leer, statt beliebige frühe Seiten einzufügen.
5. RRF mit 1-basierten Rängen: `RRF(page) = Σ 1 / (RRF_K + rank_channel(page))`.
   Eine Seite trägt pro Kanal höchstens einmal bei. Gleichstände werden aufsteigend
   nach physischer Seitennummer aufgelöst. Quellenidentität: `PDF-SHA256:page:N`.
6. Bis zu zehn Seiten folgen der Fusionsrangfolge. Unveränderter Originaltext und
   vorhandene Abschnitts-/Vorlagen-/Sprecherlabels stehen neben `[Page N]`.
   Originalseitenobjekte werden kopiert, nicht verändert.

Keine Zusammenfassungen, kein MMR, kein LLM-Reranking, kein GraphRAG, keine Agentenschleife.
Pro Vorhersage gibt es genau einen `ApertusClient.infer()`-Aufruf; die bisherigen
Transport-Retries bleiben erhalten. Lokale Embeddings erzeugen keinen generativen API-Aufruf.

Für beide neuen Methoden werden Zitate whitespace-normalisiert gegen ausgewählte
Originalseiten geprüft. Falsche Seitentags werden anhand des Zitats korrigiert;
unauffindbare Zitate werden nicht zu erfundenen Seite-1-Belegen. Entailment ohne
zuordenbares Zitat wird Neutral. Diese zusätzliche Quellenprüfung ist ebenfalls ein
Unterschied zum alten `retrieval`. Offizielles JSON bleibt bei `id`, `label`,
`label_name`, `evidence` und `metrics` mit unveränderter Metrikbedeutung.

## Cache, Startzeiten und Fehler

Der SHA256-Key enthält PDF-Inhaltshash, `PDFParser.VERSION`, Hash der Absätze samt
Metadaten, Modellname/-revision, Tokenlimit und Index-/Normalisierungsversion.
Eine ersetzte PDF invalidiert auch den Parsercache am gleichen Dateipfad.
Vektoren liegen als komprimiertes `.npz` ohne Pickle vor. Shape, Metadaten,
SHA256-Prüfsumme, endliche Werte und Normalisierung werden geprüft. Beschädigte
Caches werden protokolliert und neu erstellt; Schreiben erfolgt atomar. Ein geteilter
Index serialisiert Erstellung und Suche innerhalb des Prozesses. Alte Keys werden
nicht automatisch gelöscht und können bei Bedarf entfernt werden.

Gewichte liegen separat im Hugging-Face-Cache, normalerweise `~/.cache/huggingface`;
`HF_HOME` kann den Pfad ändern. Fehlende Abhängigkeiten/Gewichte, leere/nicht lesbare
PDFs, ungültige Vektoren und nicht beschreibbare Caches führen zu erklärten Fehlern.
BM25 braucht keinen Modell-/Embedding-Cache. `.cache/` und Gewichte werden nicht eingecheckt.

Bei explizitem `HYBRID_DENSE_FALLBACK=true` wird ein Wechsel auf `hybrid` protokolliert:
`PredictionResult.strategy` und `actual_strategy` zeigen die tatsächliche Methode,
`requested_strategy` den Wunsch und `fallback_reason` den Grund. Solche Fälle dürfen
beim Benchmark nicht als erfolgreicher Dense-Lauf interpretiert werden.

Die bisherige `latency_ms`, `avg_latency_ms` und offizielle `inference_time_ms`
bleiben Modell-/API-Inferenzzeit. Neue `total_latency_ms`/`avg_total_latency_ms`
umfassen die ganze Prüfung. Zusätzlich werden erfasst:

| Feld | Bedeutung |
|---|---|
| `document_preparation_ms` | PDF-Hash und Parserarbeit bzw. Parsercache-Zugriff |
| `retrieval_ms` | ganzer Suchpfad einschließlich erforderlichem Index-/Modellstart |
| `model_load_ms` | Import, ggf. Download, Laden/Initialisieren des Modells |
| `index_build_ms` | Dokumentindex-Erstellung einschließlich Modellstart, Encoding, Schreiben |
| `index_load_ms` | Disk-Lookup und ggf. Lesen/Validieren |
| `document_embedding_ms` | Dokument-Encoding ohne Modellstart |
| `query_embedding_ms` | Claim-Encoding ohne Modellstart |
| `embedding_ms` | Summe der beiden Encoding-Zeiten |

Zeiten sind **verschachtelt**, nicht alle additiv. Disk-/Memory-Hits kodieren die
Dokumentabsätze nicht erneut; Claims werden pro Anfrage kodiert. `index` lädt den
Query-Encoder auch bei vorhandenem Disk-Index und macht damit einen echten Warm-up.

## Vergleich auf Dev-Daten

Die gebündelten Datensätze enthalten nur DE→DE, FR→FR und IT→IT sowie keine verlässlichen
Gold-Seiten. Für die Parameterwahl braucht es unabhängig annotierte Dev-/Validation-Fälle
mit Sprachpaaren. Den offiziellen Testbenchmark nicht als Dev-Daten umetikettieren.

JSONL-Eingabe: `claim`, `entailment_label`, `claim_language`, `booklet_language`,
optional `booklet_date`, `vote`, `reference_string`, `split`, `gold_evidence_pages`.
Gold-Referenz, Label und Gold-Seiten werden ausschließlich ausgewertet und nie an
Retrieval oder Vorhersage übergeben. `vote` ist ein tatsächlicher Eingabe-Vorlagentitel.

```bash
# Technischer Vergleich ohne Apertus-Kosten; NLI-Metriken bleiben Mock-Werte.
python -m src compare-hybrid --dataset /path/to/independent_dev.jsonl \
  --split dev --mock --output /tmp/hybrid_mock.json

# Erst nach Autorisierung des konkreten API-Budgets ausführen.
# --live führt vier NLI-Evaluierungen je ausgewähltem Fall aus.
python -m src compare-hybrid --dataset /path/to/independent_dev.jsonl \
  --split dev --live --limit 5 --output /tmp/hybrid_live.json

python -m src benchmark --dataset /path/to/independent_dev.jsonl \
  --strategy hybrid_dense --mock --report /tmp/dense_report.json
```

`compare-hybrid` verlangt `--mock` oder `--live`, akzeptiert `dev`/`validation` und
lehnt abweichend markierte Daten ab. Ohne Split-Metadaten muss der Anwender die Herkunft
korrekt deklarieren; der Code kann sie nicht beweisen. Beide Methoden erhalten dieselben
Fälle für einen ersten Durchlauf und eine warme Wiederholung. Ein erster Durchlauf ist
**nicht automatisch kalt**: vorhandene Disk-/Modellcaches bleiben bestehen. Cache-Status
und Modell-/Indexzeiten zeigen den tatsächlich beobachteten Zustand.

Zusätzliche Reports: Macro-F1 pro Claim-Sprache, tatsächlichem Sprachpaar und
sprachübergreifender Teilmenge, Output-Tokens, Gesamtzeiten, Cache-/Strategiezähler.
Sprachübergreifendes F1 ist `null`, wenn entsprechende Fälle fehlen. Die alte
Evidence-Alignment-Rate bleibt eine **lexikalische Näherung** (Wortüberlappung ≥ 0,30
oder Teilstring), keine semantische Belegqualitätsmessung. Seiten-Recall wird nur für
explizit annotierte physische 1-basierte `gold_evidence_pages` des tatsächlich verwendeten Büchleins berechnet, sonst `null`. Bei Sprach-Fallback werden diese Gold-Seiten nicht verglichen.
Die alte Rate 1,0 bei fehlenden Referenzen bleibt aus Kompatibilitätsgründen erhalten;
`reference_string_count=0` bedeutet dort **nicht gemessen**.

## Docker und Ressourcen

Beide Dockerfiles behalten die leichte Basisinstallation. `INSTALL_HYBRID=true`
ergänzt CPU-PyTorch und Sentence Transformers; der Build lädt keine Embedding-Gewichte.
`.dockerignore` schließt lokale Zugangsdaten, virtuelle Umgebungen und Indizes aus.

```bash
docker build --build-arg INSTALL_HYBRID=true -t fact-attack-dense .

docker run --rm \
  -v fact-attack-models:/root/.cache/huggingface \
  -v fact-attack-indices:/app/.cache/embeddings \
  fact-attack-dense index --booklet data/booklets/2026-06-14_de.pdf

docker run --rm \
  -v fact-attack-models:/root/.cache/huggingface \
  -v fact-attack-indices:/app/.cache/embeddings \
  -e EMBEDDING_LOCAL_FILES_ONLY=true \
  fact-attack-dense search --strategy hybrid_dense \
  --claim "Die Initiative verlangt eine Begrenzung der Wohnbevölkerung."
```

Gewichte und PyTorch erhöhen Downloadvolumen, Imagegröße und Speicherbedarf deutlich.
Für CPU-Starts mehrere GB RAM und Platz für die Gewichte einplanen; Batchgröße und
Absatzlänge beeinflussen den Bedarf. Begrenzte Render-Instanzen sollten die alte Methode
verwenden, solange Ressourcen und Qualität nicht geprüft sind. GPU-Nutzung ist optional
und braucht eine passende PyTorch-/Containerinstallation.

## Tatsächliche Messungen und Grenzen

Rohdaten: [`hybrid_dense_cpu_measurement.json`](hybrid_dense_cpu_measurement.json).
Ein CPU-Smoke-Test auf macOS/arm64 mit vier Threads, Batchgröße 4, 2048 Tokenlimit
und 35 Absätzen des deutschen Juni-2026-Büchleins:

| Zustand | Messung |
|---|---|
| Initialer Warm-up einschließlich Modelldownload | 227,6 s gesamt |
| Modellimport/-download/-start dabei | 168,0 s |
| Dokument-Encoding ohne Modellstart | 58,0 s |
| Erster Suchlauf in neuem Prozess mit Disk-Index | 15,8 s gesamt; kein Dokument-Encoding |
| Weitere DE-/FR-/IT-Claims gegen deutsches Büchlein | etwa 3,3–4,0 s je Anfrage, Memory-Hit |

Die drei Beispielclaims lieferten jeweils Seite 11 zuerst; Wiederholungen je Claim
hatten identische Rangfolgen. Das ist ein Funktionsbeispiel, **kein annotierter
Retrieval-Qualitätsbenchmark**. Dieser CPU-Smoke-Test enthielt keine Apertus-Aufrufe
und keine NLI-F1-Messung. Die Zeiten gelten für diese Maschine.

### Abgeschlossener Live-Qualitätsvergleich vom 5. Oktober 2026

Nach Freigabe wurden genau 200 erfolgreiche Apertus-Anfragen ausgeführt: dieselben
50 historischen Validierungsfälle je Methode, jeweils erster Lauf und warme
Wiederholung. 18 Fälle sind sprachübergreifend; insgesamt liegen 32 verschiedene
Claims vor. Der offizielle Testsplit und die Retrieval-Konfiguration blieben unverändert.

| Methode | Macro-F1 erster Lauf / warm | Sprachübergreifendes F1 erster Lauf / warm |
|---|---:|---:|
| `hybrid` (BM25) | 0.6016 / 0.5593 | 0.5333 / 0.4524 |
| `hybrid_dense` (BM25 + Vektoren) | 0.5865 / 0.6106 | 0.6639 / 0.6639 |

Vektorsuche verbessert die sprachübergreifende Teilmenge dieser Probe; beim
gesamten Pipeline-F1 ist der Vorteil noch nicht stabil. Drei fragliche übernommene
Labels, abgeleitete Sprachpaar-Fälle und nachgelagerte Beleg-/Konfidenzregeln
begrenzen die Aussagekraft. Warm benötigt die Vektormethode auf CPU im Mittel
5.744 s, BM25 1.421 s pro Fall. Der bisherige Standard bleibt erhalten.

Der [vollständige Qualitätsbericht](evaluation/hybrid_quality_2026-10-05.md)
enthält Sprachpaare, Belege, Labelprüfung, Rohlabels, Tokens, Index-/Cachezeiten,
Originalantworten und die nachvollziehbaren Ergebnisse beider Wiederholungen.

Tests nutzen kontrollierte Fake-Vektoren ohne API/Modelldownload. Sie prüfen Fusion,
Cosinus-Normalisierung, Titelgewichtung, Seitenaggregation und Quellen, Cache-/PDF-
Invalidierung, korrupte Caches, leere Eingaben, fehlende Vektoren, expliziten Fallback,
einen Inferenzaufruf, alte Methoden und offizielles JSON. AppTest prüft die verzögerte
Modellauswahl und Fehleranzeige. Fake-Vektoren belegen keine echte Mehrsprachigkeitsqualität.

### Ausgeführte technische Validierung

- 65 Tests lokal unter Python 3.12: bestanden, einschließlich Streamlit AppTest.
- Root-Dockerfile mit Basisinstallation und Track-Dockerfile mit
  `INSTALL_HYBRID=true`: erfolgreich gebaut auf Linux/arm64.
- Dieselben 65 Tests in beiden Images mit aktuellem Source-Code als read-only Bind-Mount:
  bestanden, mit Offline-Umgebungsvariablen und ohne Modell-/API-Aufrufe in den Tests.
- `pip check`, Python-Kompilierung und `git diff --check`: erfolgreich.
- Echter CLI-Suchaufruf mit vorhandenen Gewichten und Disk-Index: erfolgreich.
- Echter Offline-Suchaufruf im CPU-Dockerimage mit read-only Gewichten/Index:
  30,7 s inklusive Prozess-/Modellstart, dieselbe Rangfolge wie der DE-Beispielaufruf.
  Der Containerlauf ist eine einzelne Funktionsmessung, kein warmer Leistungsbenchmark.
- Im Basisimage ohne Sentence Transformers stoppt `hybrid_dense` ausdrücklich mit
  Installationshinweis; kein stiller BM25-Fallback und kein Modelldownload.

Die lokalen Abhängigkeiten und BGE-M3-Gewichte sind für die Entwicklung installiert;
das deutsche Juni-2026-Büchlein sowie die November-2024-Büchlein in DE/FR/IT sind
indexiert. GPU-/MPS-Laufzeiten wurden nicht gemessen. Die sprachübergreifende
NLI-Probe ist oben beschrieben. Docker-/Modellcaches sind
lokale Entwicklungsartefakte und werden nicht eingecheckt.

## Quellen

- [Offizielle BGE-M3-Modellbeschreibung](https://huggingface.co/BAAI/bge-m3):
  mehrsprachige Dense-Embeddings, keine notwendigen Query-Instruktionen.
- [Sentence Transformers API](https://sbert.net/docs/package_reference/sentence_transformer/model.html):
  Modellrevision, `local_files_only` und normalisiertes Encoding.

Vor einer Änderung des Standards steht ein budgetierter Vergleich auf unabhängig
annotierten Dev-Sprachpaaren aus. Installation und CPU-Funktion sind geprüft;
ein Qualitätsvorteil wird nicht behauptet.
