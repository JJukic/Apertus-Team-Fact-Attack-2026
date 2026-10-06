# Independent Claude Opus 5.5 review and Codex assessment

Model confirmed by CLI response metadata: `claude-opus-5-5`. Review was performed from supplied source files, without execution or edits. Claude Code was updated from 2.1.267 to 2.1.292 to support the model.

## Codex assessment

- Confirmed: judge parse failures discard that response's usage; the experiment then records only extraction tokens.
- Confirmed, conditional on SPEAKER_HINT=true: original passage ids in the appended claim hint become stale after the selected context is renumbered. Default SPEAKER_HINT=false avoids this path today.
- Confirmed: extraction checks decorated passage strings, so metadata-only snippets can pass substring validation. Validation should use original source text.
- Confirmed design limitation: extracted statements are logged, but only their source ids affect judgment. This is a learned passage filter, not atomic claim verification.
- Correction to Claude review: SPEAKER_AWARE is false by default. Opposing sections are not normally removed; speaker boost is enabled by default.
- A baseline fallback protects against technical extraction failures, not against an incorrect second judgment. A second pass can still worsen accuracy. Measure both fixed and newly broken cases.
- Proposed next experiment: sentence ids with original page/speaker provenance; baseline and second-pass review over the same full context, and compare that with filtering selected passages. Conditional triggering is a hypothesis to validate on dev, not a proven improvement.
- Use matched dev cases with both failures and correct controls. If oversampling language groups, report subgroup metrics and avoid equating that sample's aggregate with the population benchmark.

No new live benchmark was performed. Existing tests do not establish model accuracy. No implementation changes were made during this review.

## Original Claude review

# Review: Extract-then-Verify (`experiment/two-stage-fact-check`)

**Kurzfazit:** Die Verdrahtung ist sauber getestet. Inhaltlich ist die erste Stufe aber nur ein **LLM-Passagenfilter**. Die extrahierten Zitate erreichen den Judge nie, nur ihre `passage_id`s bestimmen, welche vollständigen Passagen übrig bleiben. Der Ansatz kann also höchstens durch das Entfernen von Distraktoren helfen. Gleichzeitig öffnet er neue Wege zu falschem Neutral, und das ist schon heute die grösste Fehlerklasse der Baseline (0→1: 5, 2→1: 7, also 12 von 24). Ich habe nichts ausgeführt. Alle Aussagen unten stammen aus dem Lesen des Codes.

---

## Befunde

### 1. Bug: Speaker-Hint zeigt nach dem Umnummerieren auf falsche Passagen
**Ort:** `inference.py::_compact_predict` → `two_stage.py::infer_two_stage`

`_compact_predict` hängt bei `config.SPEAKER_HINT` an `model_claim` den Satz `(Passages written by the … itself: P2, P5)` an. Die Nummern beziehen sich auf die **ursprüngliche** Nummerierung. `infer_two_stage` übergibt diesen `claim` unverändert an `client.infer(..., passages=selected)`. Dort werden die ausgewählten Passagen aber neu als P1…Pk nummeriert.

Beispiel: Ausgewählt werden die Originalpassagen [5, 7]. Der Judge sieht sie als P1 und P2, liest aber den Hinweis „P2, P5“. P5 existiert nicht, und P2 ist die Originalpassage 7, möglicherweise von der Gegenseite.

Das trifft genau die Fehlerklasse, gegen die der Hint gebaut wurde (sprecherattribuierte Claims). In Stufe 1 ist der Hint noch korrekt, weil dort die Nummerierung übereinstimmt.

**Fix:** Entweder den Hint nach der Auswahl neu aufbauen (`two_stage` braucht dafür die strukturierten Passagen samt `section`), oder im Judge die Original-IDs beibehalten (`[P5]`, `[P7]`) und die `evidence_ids` gegen diese Menge validieren. Dazu gehört ein Test mit aktivem `SPEAKER_HINT` und nicht-trivialer Auswahl.

### 2. Bug: Verbrauchte Tokens gehen bei Judge-Fehlern verloren
**Ort:** `apertus_client.py::infer` (except-Zweig), betrifft auch `infer_compact` bei Parse-Fehlern

Wenn die Antwort zwar ankommt, aber nicht parsebar ist, gibt `infer` `tokens_prompt=0, tokens_completion=0` zurück. Die Usage aus `response.usage` und gegebenenfalls `forced_prompt/forced_completion` des Thinking-Nachschubs werden verworfen. `infer_two_stage` addiert danach nur die Extraktionstokens.

Konkret erwartet `test_two_stage_pipeline.py::test_invalid_judgment` keine Tokenprüfung. Nach meiner Lesart würde das Ergebnis 120 statt 240 melden.

Die Folge ist eine systematische Unterschätzung der Kosten, genau bei der Methode mit mehr Calls und mehr Fehlerstellen. Das verzerrt den Kostenvergleich und die offiziellen `metrics`.

**Fix:** Usage direkt nach Erhalt der Response auslesen, also vor dem `try`, und auch im Fehlerpfad zurückgeben. Danach `tokens_total == 240` im Test prüfen.

### 3. Bug: Die Grounding-Prüfung ist schwächer, als die Doku suggeriert
**Ort:** `two_stage.py::infer_two_stage` (Validierungsschleife), Eingabe aus `_compact_predict.label()`

- Validiert wird gegen `texts`, also gegen Text **mit** Metadaten-Präfix `(p. 16; Arguments of the Federal Council and Parliament) …`. Ein „Zitat“ wie `Arguments of the Federal Council and Parliament` besteht die Prüfung, obwohl es kein Booklet-Text ist.
- Beliebig kurze Substrings wie `"Die"` oder `"de"` bestehen ebenfalls. „Exaktes Zitat“ beweist hier weder Relevanz noch Grounding.
- Die exportierte Evidence ist die ganze (per `clip_to_query` gekürzte) Passage, nicht das extrahierte Zitat. Ausserdem müssen die vom Judge zitierten IDs nicht mit den Passagen der Zitate übereinstimmen. Die Zitate sind reine Logs.

**Fix:** Gegen `p["text"]` ohne Präfix validieren, eine Mindestlänge oder Mindest-Tokenzahl verlangen und Whitespace normalisiert vergleichen (siehe Punkt 4).

### 4. Design-Risiko, Mechanik belegt, Häufigkeit Hypothese: Alles-oder-nichts-Validierung und Kurzschluss auf Neutral
**Ort:** `two_stage.py::infer_two_stage`

Belegt ist der Mechanismus:
- **Ein** ungültiges Statement von bis zu 8 verwirft die ganze Extraktion. Das Ergebnis ist `label=1` mit Error, ohne Judge-Call.
- `statements: []` liefert Neutral ohne Judge.
- `passage_id: "2"` oder `"P2"` scheitert an `type(i) is not int`.
- Bei `max_tokens=1200` und 8 längeren deutschen Zitaten ist abgeschnittenes JSON möglich. Das führt zu einem Parse-Fehler und damit zu Neutral.
- Anders als `infer` ruft `infer_two_stage` kein `_strip_thinking` auf. Das ist relevant, falls die Thinking-Variante auch ohne Prefill Reasoning-Tokens ausgibt.

Hypothesen, die zu messen sind:
- Bei cross-lingualen Paaren übersetzt oder normalisiert das Modell Zitate: typografische Apostrophe ’ statt ', Silbentrennung und Zeilenumbrüche aus dem PDF, «» statt "". Dadurch scheitert der exakte Substring-Check. fr→de ist mit 6 von 36 Fehlern (16,7 %) schon das schwächste Paar der Baseline.
- Der Extraktor übersieht Gegenbelege, obwohl der Prompt sie anfordert. Dann sieht der Judge die widersprechende Passage gar nicht, und es entsteht 2→1.

Beides würde gezielt die bestehende Hauptfehlerklasse verstärken.

**Fix:** Ungültige Statements einzeln verwerfen statt die ganze Extraktion. Bei leerer oder komplett ungültiger Extraktion **auf den Baseline-Kontext zurückfallen**, nicht auf Neutral.

### 5. Vergleichbarkeit: Der geplante Vergleich ist so nicht fair
**Ort:** `docs/two_stage_experiment.md`

- Die 0.9402 stammen aus Testdaten, `hybrid_ids`, `70B-thinking`, Commit `e9f6cd7` und liegen vor der strengeren Validierung. Ein Dev-Lauf ist damit nicht direkt vergleichbar, was die Doku teilweise selbst sagt.
- Die Kommandos pinnen weder Modell noch die Flags `THINKING`, `FEW_SHOT`, `SPEAKER_AWARE/BOOST/HINT`, `TRANSLATE_CLAIM`, `PAGE_MAX_CHARS` und `top_k`. Zwei Läufe zu unterschiedlichen Zeiten oder mit anderer `.env` vergleichen damit mehr als nur die Methode.
- Grobe Rechnung, keine Messung: Bei etwa 6 % Fehlerrate entstehen auf 150 Fällen rund 9 Fehler. Unterschiede von 2 bis 4 Fällen liegen im Rauschen, besonders weil nicht bekannt ist, wie stark der Endpoint bei `temperature=0` zwischen Läufen schwankt.
- Technische Fehler werden als Label 1 ausgegeben. In der Macro-F1 erscheinen sie damit als Modellentscheidungen. Two-stage hat mehr Fehlerstellen, das muss getrennt ausgewiesen werden.

---

## Bringt Extract-then-Verify überhaupt etwas?

So wie implementiert, ist es funktional ein **Reranker mit Recall-Risiko**. Der Judge bekommt dieselbe Art Input wie in der Baseline (`infer` im ids-Modus), nur weniger Passagen. Das hilft nur, wenn die Fehler der Baseline durch Distraktoren entstehen. `SPEAKER_AWARE` entfernt die Gegenseite aber schon heute. Bei kleinem `top_k` bleibt wenig zu filtern, und die Passagen werden dafür zweimal bezahlt.

Ob die Fehler von Distraktoren, fehlendem Kontext, Sprache oder Gold-Labels kommen, ist laut `offline_baseline_analysis.md` noch nicht geklärt. Der Nutzen ist damit eine **unbelegte Hypothese**.

## Empfohlene Alternative

**Baseline zuerst, zweiter Durchgang nur gezielt, Satz-IDs statt Freitext-Zitate:**

1. **Baseline-`ids`-Call wie bisher.** Er liefert das Label für alle Fälle.
2. **Risiko-Trigger** bestimmen, wann ein zweiter Durchgang läuft. Kandidaten:
   - cross-linguale Paare, besonders →de,
   - sprecherattribuierte Claims (`attributed_section`),
   - Label 1 mit `p_entail`/`p_contra` > 0,
   - Konflikt zwischen Arbiter-Regel und Modelllabel (Rule 0/0b),
   - numerischer Konflikt.
3. **Zweiter Durchgang nur für getriggerte Fälle.** Die Passagen werden in nummerierte Sätze S1…Sn zerlegt, jeder mit Seite und Sprecher-Label. Gefragt wird:
   - nach Satz-IDs für Stützung und Widerspruch, getrennt nach Sprecher,
   - nach einem Urteil, das diese Sätze nutzt, mit den vollen Passagen als Kontext und mit **Original-IDs**.

   IDs statt Freitext eliminieren die Substring-Fehler aus Punkt 4. Die gewählten Sätze ergeben direkt präzise, seitengenaue Evidence für den Export.
4. **Kein Pfad endet ohne Judge in Neutral.** Ein Fehler in Stufe 2 bedeutet, dass das Baseline-Label bleibt.
5. Optional für 0→2 und 2→1: **Claim-Dekomposition** in atomare Aussagen, die einzeln geprüft und dann aggregiert werden. Die Doku nennt das als fehlend. Ob zusammengesetzte Claims Fehler verursachen, wäre vorher an den Dev-Fehlern zu prüfen.

Vorteile: Die Kosten steigen nur für den Bruchteil getriggerter Fälle. Das Risiko ist nach unten begrenzt, weil die Baseline der Fallback ist. Und die Wirkung lässt sich pro Trigger auswerten.

---

## Experimentplan

### Offline (ohne API)
1. Bugs 1–3 fixen und Tests ergänzen:
   - Speaker-Hint nach der Auswahl korrekt,
   - `tokens_total == 240` bei Judge-Parse-Fehler,
   - Metadaten-Präfix als Zitat wird abgelehnt,
   - Whitespace- und Apostroph-normalisierter Match,
   - Fallback statt Neutral bei leerer oder ungültiger Extraktion.
2. **Retrieval-Obergrenze bestimmen:** Für Dev-Fälle mit Gold-Evidence (falls der Datensatz Seiten oder Text enthält) Recall@k der Hybrid-Retrieval messen. Was nicht im Kontext steht, kann kein Filter retten. Dieser Wert begrenzt den möglichen Gewinn.
3. Tokenkosten pro Methode aus den realen Retrieval-Kontexten mit `_estimate_tokens` abschätzen und klar als Schätzung kennzeichnen.
4. Die Trigger-Abdeckung offline auf dem gespeicherten Baseline-Report prüfen: Wie viele der 24 Fehler und wie viele der korrekten Fälle würden getriggert? Nur zur Diagnose, **nicht** zum Tunen. Das sind exponierte Testfälle.

### Live
1. **Config-Snapshot pro Lauf loggen:** Modell, alle Flags, `top_k`, Commit. Rohantworten **beider** Stufen speichern.
2. **Rauschboden messen:** Baseline `ids` zweimal auf denselben Dev-Fällen laufen lassen. Die Label-Flip-Rate zwischen den beiden Läufen ist das Rauschen, gegen das jeder Effekt bestehen muss.
3. **Stratifizierte Dev-Stichprobe** statt einer zufälligen mit 150 Fällen. fr→de, sprecherattribuierte Claims und Gold-Neutral überrepräsentieren, dazu korrekte Kontrollfälle.
4. Auf **identischen Fällen im selben Zeitfenster** laufen lassen: Baseline `ids`, `two_stage` mit Fixes, Alternative mit Triggern. Als Ablation zusätzlich „alle Passagen plus markierte Zitate“ statt Filter. Damit trennt sich der Nutzen der Extraktion vom Effekt des Wegfilterns.
5. **Metriken:**
   - Macro-F1 einmal mit technischen Fehlern als falsch und einmal mit ausgeschlossenen Fehlern,
   - Fehlerrate pro Stufe,
   - Rate leerer Extraktionen und deren Präzision gegen Gold-Neutral,
   - Extraktions-Recall (enthält die Auswahl die Gold-Evidence?),
   - Konfusionsübergänge,
   - Tokens und Latenz pro Fall inklusive Fehlerfälle.
6. **Statistik:** gepaarter McNemar-Test auf den diskordanten Fällen und Bootstrap-Konfidenzintervall für ΔF1. Das **Entscheidungskriterium wird vorher festgelegt**: CI schliesst 0 aus, keine höhere technische Fehlerrate, Tokenbudget eingehalten.
7. Erst danach **einmal** auf dem Test-Split (402 Fälle) mit der aktuellen Baseline als Gegenlauf. Die 24 Fehlerfälle werden nicht als Tuning-Grundlage verwendet.

---

**Nicht gemessen:** Ich habe weder Modellqualität noch Fehlerraten, Tokenverbrauch oder Latenz erhoben. Die Zahlen oben stammen aus den mitgelieferten Dokumenten oder sind ausdrücklich als Überschlag markiert.