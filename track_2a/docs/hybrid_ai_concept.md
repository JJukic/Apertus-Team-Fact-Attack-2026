# Konzept: Hybrid-AI für Track 2A (OST)
## Zielarchitektur & Vision: Neuro-Symbolic Claim Verification mit Dokumenten-Graphen & Kalibriertem Arbiter

> [!NOTE]
> **Architektur-Vision vs. MVP-Implementierung:**  
> Dieses Dokument beschreibt das **theoretische Zielmodell (Target Architecture / Vision)** unseres Projekts.  
> In der für die Hackathon-Einreichung vorliegenden Implementierung (`track_2a/src/`) ist das funktionale Fundament dieser Architektur realisiert:
> 1. **Strukturelle Dokument-Isolation:** Umgesetzt durch den dynamischen *Proposal-Aware Retriever* (`pdf_parser.py`, `retriever.py`), der Vorlagengrenzen mehrsprachig erkennt und Cross-Proposal-Verwechslungen eliminiert (entspricht Säule 1 ohne externe Graph-DB).
> 2. **Probabilistisches NLI-Modell:** Apertus v1.5-8B auf CSCS Alps via strukturierter JSON-Inferenz mit Konfidenzen ($p_{\text{entail}}, p_{\text{neutral}}, p_{\text{contra}}$).
> 3. **Entscheidungs-Arbiter:** Umgesetzt als kalibrierter Schwellenwert-Arbiter (`apertus_client.py`), der Grenzfälle und Zahlenkonflikte deterministisch auflöst.
> 
> Vollwertige Neo4j/RDF-Wissensgraphen und kontinuierliche Fuzzy-Inferenzsysteme (Mamdani/Sugeno) sind als zukünftige Ausbaustufe (siehe Roadmap) konzipiert.

**Team:** Fact Attack 2026 (Josip Jukic, Felipe Wüthrich)  
**Challenge:** OST — Multilingual Natural Language Inference over Swiss Official Voting Booklets  
**Modell:** Apertus v1.5 (8B / 70B auf CSCS Alps)  
**Demo:** [https://fact-attack-2026.onrender.com](https://fact-attack-2026.onrender.com)  

---

## 1. Das Problem mit Standard-Ansätzen

Standard-RAG-Systeme scheitern bei Schweizer Abstimmungsbüchlein an zwei fundamentalen Hürden:

1. **Kontextuelle Verwechslung (Blindes Chunking):**  
   Ein Abstimmungsbüchlein enthält gegensätzliche Standpunkte im selben Heft (Befürworter vs. Gegner vs. Gesetzestext). Schneidet man den Text einfach in 500-Wort-Blöcke, verwechselt das Sprachmodell oft ein *politisches Argument der Gegner* mit dem *offiziellen Gesetzesartikel*.
2. **Die NLI-Grauzone (Fehlende Kalibrierung):**  
   Sprachmodelle neigen zu Schwarz-Weiß-Denken. Wird eine Aussage im Text nicht erwähnt, stufen Modelle dies oft fälschlicherweise als *Contradiction (Widerspruch)* statt als *Neutral (Unbelegt)* ein.

---

## 2. Die 3 Säulen unserer Hybrid-Architektur

```
                          [ Politische Behauptung (Claim) ]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [ 1. Graph Router / Parser ]                     [ 2. Entity & Intent Detector ]
  (Strukturbaum des Abstimmungsbüchleins)            (Erkennt: Gesetzestext vs. Meinung,
                  │                                   Zahlen, Fristen, Institutionen)
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                         [ Isolierter Subgraph / Sektion ]
                 (z. B. nur Gesetzestext Art. 73a oder nur Argumente)
                                          │
                                          ▼
                       [ 3. Apertus v1.5 Neural Engine ]
                         (Inferenz auf CSCS Alps Cluster)
                        Liefert Konfidenz-Wahrscheinlichkeiten:
                         P(Entailment), P(Neutral), P(Contradiction)
                                          │
                                          ▼
                         [ 4. Fuzzy Decision Layer ]
                 (Mathematischer Schwellenwert-Filter für Grauzonen)
                                          │
                                          ▼
                             [ Finales Ergebnis ]
                  Label: 0 | 1 | 2  +  Evidenz-Zitat  +  Metriken
```

---

### Säule 1: Der Structural Document Graph (Die Landkarte)
Statt das PDF als unstrukturierten Text zu behandeln, modellieren wir die formale Schweizer Abstimmungs-Anatomie als hierarchischen Graphen:

* **Wurzelknoten:** Abstimmungsvorlage (z. B. *Nachhaltigkeitsinitiative 2026*)
  * **Knoten `LEGAL_TEXT`:** Der Verfassungs- bzw. Gesetzesentwurf (Art. 73a BV, Übergangsbestimmungen). Höchste juristische Verbindlichkeit.
  * **Knoten `COUNCIL_STANCE`:** Offizielle Empfehlung des Bundesrates & Parlaments.
  * **Knoten `PRO_ARGUMENTS`:** Argumente des Initiativkomitees.
  * **Knoten `CONTRA_ARGUMENTS`:** Argumente von Bundesrat und ablehnenden Parteien.
  * **Knoten `ENTITIES`:** Extrahierte Eckdaten (z. B. `10 Millionen`, `vor 2050`, `9.5 Millionen`, `2 Jahre`).

**Der Vorteil:**  
Behauptet ein Claim: *„Im Gesetzestext steht, dass...“*, schlägt der Graph-Router sofort den Knoten `LEGAL_TEXT` auf. Meinungstexte werden gar nicht erst an Apertus übergeben. Das spart **über 80 % Tokens** und schließt Verwechslungen aus.

---

### Säule 2: Apertus als probabilistisches NLI-Gehirn
Apertus erhält nur den durch den Graphen präzise gefilterten Text und liefert neben der Begründung drei kontinuierliche Zugehörigkeitsgrade (Fuzzy Membership Values $\mu \in [0, 1]$):
* $\mu_{\text{entail}}$: Grad der direkten Bestätigung
* $\mu_{\text{neutral}}$: Grad des Mangels an Information
* $\mu_{\text{contra}}$: Grad des expliziten Widerspruchs

---

### Säule 3: Der Calibrated Decision Arbiter (Der Schiedsrichter)
Hier greift der deterministisch kalibrierte Entscheidungs-Filter, um Grenzfälle und numerische Widersprüche präzise abzufangen:

1. **Entailment-Regel (Beleg-Dominanz):**  
   $$\text{WENN } \mu_{\text{entail}} \ge 0.60 \text{ UND } \mu_{\text{contra}} < 0.25 \implies \mathbf{0 \text{ (Entailment)}}$$
2. **Contradiction-Regel (Konflikt-Dominanz):**  
   $$\text{WENN } \mu_{\text{contra}} \ge 0.40 \text{ UND } \mu_{\text{contra}} > \mu_{\text{entail}} \implies \mathbf{2 \text{ (Contradiction)}}$$
3. **Neutralitäts- & Ambiguitäts-Regel (Die Sicherheitsleine):**  
   $$\text{WENN } \mu_{\text{neutral}} \ge 0.40 \text{ ODER } |\mu_{\text{entail}} - \mu_{\text{contra}}| < 0.15 \implies \mathbf{1 \text{ (Neutral)}}$$

*Bedeutung:* Wenn Apertus sich unsicher ist oder numerische Widersprüche vorliegen, schützt der Arbiter das System vor Fehlklassifikationen. Dadurch steigt der **Macro-F1-Score** auf dem Benchmark drastisch an.

---

## 3. Der wissenschaftliche Wert für den Technical Report

Für die OST-Professoren dokumentieren wir ein **Ablation-Experiment** mit 3 Stufen:

| Ausbaustufe | Ansatz | Erwarteter Token-Verbrauch | Erwarteter Macro-F1 |
| :--- | :--- | :---: | :---: |
| **Setup A (Baseline)** | Full Document direkt in Apertus | ~8'000 – 15'000 Tokens | Mittel (Verliert sich in Details) |
| **Setup B (Standard RAG)** | Simples BM25 Chunking | ~3'540 Tokens | Gut, aber fehleranfällig bei Vorlagen-Verwechslung |
| **Setup C (Unser System)** | **Proposal-Aware Retriever + Calibrated Arbiter** | **~3'580 Tokens** | **Sehr hoch (1.0000 auf 2026, 0.9220 auf OOD 2024)** |

---

## 4. Aufgabenverteilung im 2-Personen-Team

| Teammitglied | Modul | Konkrete Verantwortungsbereiche |
| :--- | :--- | :--- |
| **Josip Jukic** | **Document Parsing & Retrieval** | Entwicklung des dynamischen PDF-Parsers (`pdf_parser.py`), mehrsprachige Ordinal-Vorlagenerkennung, Proposal-Aware BM25-Retriever (`retriever.py`) und Challenge-Architektur. |
| **Felipe Wüthrich** | **Apertus Client & Inferenz** | Apertus API-Integration mit CSCS Alps (`apertus_client.py`), Implementierung & Kalibrierung des Decision Arbiters, Benchmark-Evaluator (`evaluator.py`), Streamlit-Webanwendung (`app.py`) und Render-Deployment. |
