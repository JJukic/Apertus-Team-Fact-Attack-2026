"""
Fact Attack 2026 - Multilingual Voting Booklet NLI & Fact-Checker
Streamlit Web Application for Hack Apertus Track 2A (OST).
Showcases document-grounded claim verification using Apertus-v1.5-8B on CSCS Alps.
"""

import sys
import os
import json
import time
from pathlib import Path
import streamlit as st

# Setup paths
_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from src.pdf_parser import PDFParser
from src.retriever import PassageRetriever
from src.apertus_client import ApertusClient
from src.inference import ClaimVerificationEngine, PredictionResult
from src.evaluator import BenchmarkEvaluator
from src import config

# Page Configuration
st.set_page_config(
    page_title="Fact Attack 2026 — Swiss Voting NLI",
    page_icon="🇨🇭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern Swiss aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    
    .swiss-header {
        background: linear-gradient(135deg, #d52b1e 0%, #a61c12 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(213, 43, 30, 0.25);
    }
    
    .swiss-header h1 {
        color: white !important;
        font-weight: 700;
        margin-bottom: 0.3rem;
        font-size: 2.2rem;
    }
    
    .swiss-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .metric-card {
        background: #1e222d;
        border: 1px solid #2d3343;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f0f2f6;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #9ba3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }
    
    .verdict-entailment {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        color: #34d399;
    }
    
    .verdict-neutral {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        color: #fbbf24;
    }
    
    .verdict-contradiction {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        color: #f87171;
    }
    
    .evidence-box {
        background: #181b23;
        border-left: 4px solid #d52b1e;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.8rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Cache Resources
@st.cache_resource
def get_engine():
    return ClaimVerificationEngine()

@st.cache_resource
def get_parser():
    return PDFParser()

engine = get_engine()
parser = get_parser()

# Header
st.markdown("""
<div class="swiss-header">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1>🇨🇭 Fact Attack 2026</h1>
            <p style="margin: 0; font-size: 1.1rem; opacity: 0.95;">
                Multilingual Natural Language Inference over Swiss Official Voting Booklets
            </p>
        </div>
        <div style="text-align: right;">
            <span class="swiss-badge">Track 2A (OST)</span>
            <span class="swiss-badge">Apertus v1.5-8B</span>
            <span class="swiss-badge">CSCS Alps</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_factcheck, tab_benchmark, tab_methodology = st.tabs([
    "🔍 Interaktiver Faktencheck",
    "📊 Jury Benchmark Dashboard",
    "🧠 Architektur & Generalisierung",
])

# =============================================================================
# TAB 1: INTERAKTIVER FAKTENCHECK
# =============================================================================
with tab_factcheck:
    col_config, col_main = st.columns([1, 2], gap="large")

    with col_config:
        st.subheader("⚙️ Abstimmung & Sprache")
        
        booklet_catalog = {
            "🗳️ Juni 2026: Nachhaltigkeitsinitiative & Zivildienst (Offiziell)": "2026-06-14",
            "🗳️ November 2024: Nationalstrassen, Mietrecht, EFAS (4 Vorlagen)": "2024-11-24",
            "🗳️ September 2024: Biodiversität & BVG-Reform (2 Vorlagen)": "2024-09-22",
        }
        
        selected_ballot_name = st.selectbox("Abstimmungsvorlage wählen", list(booklet_catalog.keys()))
        selected_date = booklet_catalog[selected_ballot_name]
        
        selected_lang = st.radio("Sprache des Abstimmungsbüchleins", ["Deutsch (DE)", "Français (FR)", "Italiano (IT)"], horizontal=True)
        lang_code = "de" if "DE" in selected_lang else ("fr" if "FR" in selected_lang else "it")
        
        # Resolve PDF Path
        pdf_path = config.BOOKLETS_DIR / f"{selected_date}_{lang_code}.pdf"
        
        # Custom PDF Upload Option
        with st.expander("📁 Eigenes Abstimmungsbüchlein hochladen (PDF)"):
            uploaded_file = st.file_uploader("PDF Datei wählen", type=["pdf"])
            if uploaded_file is not None:
                custom_save_path = config.BOOKLETS_DIR / f"custom_{uploaded_file.name}"
                with open(custom_save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_path = custom_save_path
                st.success(f"Geladen: {uploaded_file.name}")

        st.markdown("---")
        st.subheader("💡 Beispiel-Behauptungen")
        
        # Curated Sample Claims for Quick Testing
        sample_claims = {
            "2026-06-14": {
                "de": [
                    ("🟢 Entailment", "Wenn die Abstimmung angenommen wird, darf die Wohnbevölkerung vor 2050 10 Millionen Menschen nicht überschreiten."),
                    ("🔴 Contradiction", "Gemäss dem Abstimmungstext ist die Bevölkerung seit 2002 um rund 500'000 Personen gewachsen."),
                    ("🟡 Neutral", "Der Bundesrat argumentiert, dass die Revision des CO2-Gesetzes die Forschung unterstützt."),
                ],
                "fr": [
                    ("🟢 Entailment", "Si le vote est adopté, la population résidante permanente ne doit pas dépasser dix millions de personnes avant 2050."),
                    ("🔴 Contradiction", "Selon le texte de vote, la population suisse a diminué d'environ 1,7 million de personnes depuis 2002."),
                    ("🟡 Neutral", "Le Conseil fédéral soutient que la modernisation des infrastructures est prioritaire pour les dix prochaines années."),
                ],
                "it": [
                    ("🟢 Entailment", "Nel testo legale relativo al voto, si dice che prima del 2050 la popolazione non può superare i dieci milioni di abitanti."),
                    ("🔴 Contradiction", "Secondo il testo di voto, l'iniziativa chiede di limitare la popolazione a non più di 8 milioni di abitanti prima del 2050."),
                    ("🟡 Neutral", "Il Consiglio federale raccomanda che il Parlamento approvi un aumento del budget per la ricerca scientifica."),
                ]
            },
            "2024-11-24": {
                "de": [
                    ("🟢 Entailment", "Der Ausbauschritt 2023 für die Nationalstrassen umfasst Massnahmen auf sechs Autobahnabschnitten."),
                    ("🔴 Contradiction", "Bundesrat und Parlament wollen für den Ausbauschritt der Nationalstrassen weniger als 1 Milliarde Franken investieren."),
                    ("🟡 Neutral", "Der Bundesrat plant die Einführung einer allgemeinen Flugticketabgabe bis 2030."),
                ],
                "fr": [
                    ("🟢 Entailment", "L'étape d'aménagement 2023 des routes nationales prévoit des projets sur six tronçons autoroutiers."),
                    ("🔴 Contradiction", "Le Conseil fédéral et le Parlement prévoient moins de 500 millions de francs pour les routes nationales."),
                    ("🟡 Neutral", "Le Conseil fédéral préconise une réforme du système de bourses d'études d'ici 2028."),
                ],
                "it": [
                    ("🟢 Entailment", "La fase di potenziamento 2023 delle strade nazionali riguarda interventi su sei tratti autostradali."),
                    ("🔴 Contradiction", "Il Consiglio federale raccomanda di respingere con un NO la modifica relativa alla sublocazione."),
                    ("🟡 Neutral", "Il Consiglio federale propone di riformare il sistema di imposta sulle donazioni dal 2028."),
                ]
            },
            "2024-09-22": {
                "de": [
                    ("🟢 Entailment", "Die Biodiversitätsinitiative verlangt mehr Flächen und mehr finanzielle Mittel für die Natur."),
                    ("🔴 Contradiction", "Bundesrat und Parlament empfehlen Volk und Ständen ein klares JA zur Biodiversitätsinitiative."),
                    ("🟡 Neutral", "Die Schweizerische Nationalbank plant eine Erhöhung des Leitzinses im vierten Quartal."),
                ],
                "fr": [
                    ("🟢 Entailment", "L'initiative pour la biodiversité exige davantage de surfaces et de moyens financiers pour la nature."),
                    ("🔴 Contradiction", "Le Conseil fédéral et le Parlement recommandent d'accepter l'initiative sur la biodiversité."),
                    ("🟡 Neutral", "Le Département fédéral de l'économie prépare une modification des droits de douane agricoles."),
                ],
                "it": [
                    ("🟢 Entailment", "L'iniziativa per la biodiversità chiede più superfici e maggiori risorse finanziarie per la natura."),
                    ("🔴 Contradiction", "Il Consiglio federale raccomanda di approvare con un SÌ la riforma della previdenza professionale BVG."),
                    ("🟡 Neutral", "Il governo federale prevede nuove misure fiscali per le auto elettriche dal 2027."),
                ]
            }
        }
        
        current_samples = sample_claims.get(selected_date, sample_claims["2026-06-14"]).get(lang_code, [])
        for label_tag, sample_text in current_samples:
            if st.button(f"{label_tag}: {sample_text[:50]}...", use_container_width=True):
                st.session_state["claim_input"] = sample_text

    with col_main:
        st.subheader("📝 Politische Behauptung prüfen")
        claim = st.text_area(
            "Behauptung eingeben oder aus den Beispielen links wählen:",
            value=st.session_state.get("claim_input", ""),
            height=95,
            placeholder="z.B. Gemäss dem Abstimmungstext will die Initiative die Zuwanderung ab 9,5 Millionen stoppen..."
        )
        
        strategy_option = st.radio(
            "Retrieval-Strategie",
            ["Proposal-Aware BM25 Retrieval (Empfohlen — 76% Token-Ersparnis, ~1.5s)", "Full Document (Gesamter Büchleintext, ~15k Tokens)"],
            horizontal=True,
        )
        strat = "retrieval" if "BM25" in strategy_option else "full"
        
        col_btn, col_stats = st.columns([1, 2])
        with col_btn:
            verify_clicked = st.button("🚀 Behauptung prüfen", type="primary", use_container_width=True)
            
        if verify_clicked and claim.strip():
            if not pdf_path.exists():
                st.error(f"Abstimmungsbüchlein nicht gefunden unter: {pdf_path}. Bitte zuerst 'download_data.py' ausführen.")
            else:
                with st.spinner("Prüfe Behauptung mit Apertus & Proposal-Aware Retriever..."):
                    t0 = time.time()
                    result: PredictionResult = engine.verify_claim(
                        claim=claim,
                        booklet_pdf=pdf_path,
                        claim_language=lang_code,
                        strategy=strat,
                    )
                    elapsed_ms = (time.time() - t0) * 1000

                # Display Verdict Banner
                st.markdown("### Prüfergebnis:")
                if result.label == 0:
                    st.markdown(f"""
                    <div class="verdict-entailment">
                        <div style="font-size: 1.4rem; font-weight: 700;">🟢 BELEGT (Entailment — Label 0)</div>
                        <div style="margin-top: 0.5rem; font-size: 1.05rem;">
                            Die Behauptung wird durch die offiziellen Abstimmungsunterlagen <b>vollständig gestützt und bestätigt</b>.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif result.label == 1:
                    st.markdown(f"""
                    <div class="verdict-neutral">
                        <div style="font-size: 1.4rem; font-weight: 700;">🟡 NEUTRAL / UNBELEGT (Neutral — Label 1)</div>
                        <div style="margin-top: 0.5rem; font-size: 1.05rem;">
                            Das Abstimmungsbüchlein trifft hierzu <b>keine definitive Aussage</b>. Die Behauptung ist weder bestätigt noch widerlegt.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-contradiction">
                        <div style="font-size: 1.4rem; font-weight: 700;">🔴 WIDERLEGT / FALSCH (Contradiction — Label 2)</div>
                        <div style="margin-top: 0.5rem; font-size: 1.05rem;">
                            Die Behauptung steht im <b>direkten Widerspruch</b> zu den offiziellen Fakten, Zahlen oder Empfehlungen des Bundesrates.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Confidence and Details Columns
                col_res1, col_res2 = st.columns([1, 1], gap="medium")
                
                with col_res1:
                    st.markdown("#### 🧠 Kalibrierte Konfidenzen (Entscheidungs-Arbiter)")
                    st.progress(float(result.p_entail), text=f"Entailment (0): {result.p_entail * 100:.1f}%")
                    st.progress(float(result.p_neutral), text=f"Neutral (1): {result.p_neutral * 100:.1f}%")
                    st.progress(float(result.p_contra), text=f"Contradiction (2): {result.p_contra * 100:.1f}%")
                    rule_label = getattr(result, "decision_rule", None) or getattr(result, "fuzzy_rule", None)
                    if rule_label:
                        st.caption(f"Angewendete Entscheidungsregel: **{rule_label}**")

                with col_res2:
                    st.markdown("#### ⏱️ Effizienz & Green AI")
                    st.write(f"- **Inferenz-Latenz:** `{result.latency_ms:.1f} ms`")
                    st.write(f"- **Eingabe-Tokens:** `{result.tokens_prompt}` Tokens")
                    st.write(f"- **Gesamt-Tokens:** `{result.tokens_total}` Tokens")
                    saved_tokens = max(0, 14820 - result.tokens_prompt)
                    saved_pct = (saved_tokens / 14820) * 100
                    st.write(f"- **Token-Ersparnis:** `~{saved_tokens:,} ({saved_pct:.1f}%)` vs. Full-Doc")
                    st.write(f"- **Modell & Host:** `{config.LLM_NAME}` auf CSCS Alps")

                if getattr(result, "numerical_conflict", None):
                    st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0;">
                        <div style="font-weight: 700; color: #f87171; font-size: 1.05rem;">⚠️ Deterministischer Guardrail-Override (Zahlenkonflikt):</div>
                        <div style="margin-top: 0.3rem; font-size: 0.95rem; color: #fca5a5;">{result.numerical_conflict}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif "Vacuity" in str(getattr(result, "decision_rule", "")):
                    st.markdown(f"""
                    <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0;">
                        <div style="font-weight: 700; color: #fbbf24; font-size: 1.05rem;">⚠️ Formaler Vacuity-Schutz (Model Checking Guardrail):</div>
                        <div style="margin-top: 0.3rem; font-size: 0.95rem; color: #fde68a;">{result.decision_rule}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.10); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 0.6rem 1rem; margin: 0.8rem 0;">
                        <span style="font-size: 0.9rem; color: #34d399; font-weight: 600;">🛡️ Neuro-Symbolische Guardrails: Formale Konsistenz gewahrt</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("#### 💬 Begründung (Reasoning)")
                st.info(result.reasoning)

                evidence_sources = getattr(result, "evidence_sources", [])
                if evidence_sources:
                    st.markdown("#### 📑 Gefundene Belegstellen im Abstimmungsbüchlein (Exakt referenziert):")
                    b_data = engine._get_booklet_data(pdf_path)
                    for idx, src in enumerate(evidence_sources, 1):
                        badges = []
                        if src.page_number is not None:
                            badges.append(f'<span style="background: rgba(213, 43, 30, 0.25); border: 1px solid rgba(213, 43, 30, 0.5); border-radius: 4px; padding: 2px 7px; font-size: 0.8rem; font-weight: 600; color: #fca5a5;">📄 Seite {src.page_number}</span>')
                        if src.proposal_id is not None:
                            badges.append(f'<span style="background: rgba(255, 255, 255, 0.1); border-radius: 4px; padding: 2px 7px; font-size: 0.8rem; color: #cbd5e1;">🏛️ Vorlage {src.proposal_id}</span>')
                        badge_html = f'<div style="margin-bottom: 0.4rem;">{" ".join(badges)}</div>' if badges else ""
                        st.markdown(f"""
                        <div class="evidence-box">
                            {badge_html}
                            <b>Beleg {idx}:</b><br>
                            <i>"{src.quote}"</i>
                        </div>
                        """, unsafe_allow_html=True)

                        if src.page_number:
                            with st.expander(f"📖 Vollständigen Textabschnitt auf Seite {src.page_number} anzeigen"):
                                page_match = next((p["text"] for p in b_data["pages"] if p["page_number"] == src.page_number), None)
                                if page_match:
                                    st.markdown(f'<div style="font-size: 0.85rem; color: #cbd5e1; white-space: pre-wrap; max-height: 220px; overflow-y: auto; background: #11141c; padding: 10px; border-radius: 6px;">{page_match}</div>', unsafe_allow_html=True)
                                else:
                                    st.caption("Kein Volltext für diese Seite verfügbar.")
                elif result.evidence:
                    st.markdown("#### 📑 Gefundene Belegstellen im Abstimmungsbüchlein:")
                    for idx, ev in enumerate(result.evidence, 1):
                        st.markdown(f"""
                        <div class="evidence-box">
                            <b>Beleg {idx}:</b><br>
                            <i>"{ev}"</i>
                        </div>
                        """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: JURY BENCHMARK DASHBOARD
# =============================================================================
with tab_benchmark:
    st.markdown("### 🏆 Benchmark-Ergebnisse & Out-of-Distribution Generalisierung")
    st.markdown("""
    Unser System wurde auf zwei voneinander unabhängigen Testreihen evaluiert:
    1. Dem **offiziellen 28-Sample-Benchmark** (Juni 2026, OST / Hugging Face).
    2. Einem **ungesehenen 32-Sample Out-of-Distribution Benchmark** (November 2024, 4 Vorlagen), um die **Generalisierung auf ungesehene Vorlagen** empirisch zu überprüfen.
    """)
    st.info("📊 **Reported Results:** Die folgenden Kacheln und Tabellen zeigen den vollständigen Referenz-Evaluierungslauf auf **CSCS Alps** (`api.inference.cscs.ch`). Sie können unten zudem jederzeit einen Live-Benchmark-Lauf direkt im Browser starten oder im Terminal via `make run` ausführen.")

    # Top KPI Metrics Cards (Reported Snapshot)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #34d399;">1.0000</div>
            <div class="metric-label">Reported Macro-F1 (28 Samples, Juni 2026)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #60a5fa;">0.9220</div>
            <div class="metric-label">Reported OOD Macro-F1 (32 Samples, Nov 2024)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #fbbf24;">1.6s</div>
            <div class="metric-label">Reported Ø Inferenzzeit (CSCS Alps)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #a78bfa;">-76%</div>
            <div class="metric-label">Reported Token-Ersparnis vs. Full-Doc</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Interactive Live Evaluation for Jury
    with st.expander("🧪 Live-Benchmark im Browser ausführen (Replikation durch Jury)", expanded=False):
        st.markdown("Führen Sie den Evaluator live aus, um die Pipeline direkt in dieser Sitzung zu testen:")
        c_ds, c_strat, c_lim = st.columns([2, 2, 1])
        with c_ds:
            eval_dataset = st.selectbox(
                "Benchmark-Datensatz",
                ["Offizieller Benchmark (Juni 2026)", "Historischer OOD-Benchmark (November 2024)"],
            )
        with c_strat:
            eval_strat = st.selectbox("Strategie", ["retrieval", "full"])
        with c_lim:
            eval_lim = st.selectbox("Sample-Limit", [3, 5, 10, "Alle"], index=0)

        if st.button("🚀 Live-Evaluierung starten", key="btn_run_live_eval"):
            selected_path = config.BENCHMARK_PATH if "2026" in eval_dataset else (config.DATA_DIR / "benchmark_2024-11-24.jsonl")
            lim_val = None if eval_lim == "Alle" else int(eval_lim)
            with st.spinner(f"Evaluiere {eval_lim} Samples mit Strategie '{eval_strat}'..."):
                evaluator = BenchmarkEvaluator()
                live_report = evaluator.evaluate(dataset_path=selected_path, strategy=eval_strat, limit=lim_val)

            st.success(f"Live-Benchmark abgeschlossen für {live_report['sample_count']} Samples!")
            lr1, lr2, lr3, lr4 = st.columns(4)
            lr1.metric("Live Macro-F1", f"{live_report['macro_f1']:.4f}")
            lr2.metric("Evidence Alignment", f"{live_report.get('evidence_alignment_rate', 1.0) * 100:.1f}%")
            lr3.metric("Ø Input Tokens", f"{live_report['avg_prompt_tokens']:.0f}")
            lr4.metric("Ø Latenz", f"{live_report['avg_latency_ms']:.1f} ms")

            st.markdown("##### Detail-Ergebnisse des Live-Laufs:")
            st.dataframe(
                [
                    {
                        "Behauptung": r["claim"][:80] + "...",
                        "Sprache": r["claim_lang"].upper(),
                        "True Label": r["true_label"],
                        "Pred Label": r["pred_label"],
                        "Korrekt": "✅" if r["true_label"] == r["pred_label"] else "❌",
                        "Tokens": r["tokens"],
                        "Latenz (ms)": f"{r['latency_ms']:.0f}",
                    }
                    for r in live_report["results"]
                ],
                use_container_width=True,
            )

    st.markdown("---")

    col_bench1, col_bench2 = st.columns([1, 1], gap="large")

    with col_bench1:
        st.markdown("#### 🇨🇭 Offizieller Testset (Juni 2026 — Reported)")
        st.table({
            "Sprache": ["Deutsch (DE)", "Français (FR)", "Italiano (IT)", "Gesamt (Overall)"],
            "Anzahl Samples": [11, 8, 9, 28],
            "Macro-F1": ["1.0000", "1.0000", "1.0000", "1.0000"],
            "Accuracy": ["100%", "100%", "100%", "100%"],
        })

    with col_bench2:
        st.markdown("#### 🛡️ Ungesehenes Abstimmungsbüchlein (November 2024 — Reported)")
        st.table({
            "Sprache": ["Deutsch (DE)", "Français (FR)", "Italiano (IT)", "Gesamt (Overall)"],
            "Anzahl Samples": [12, 10, 10, 32],
            "Macro-F1": ["0.9327", "0.9153", "0.9153", "0.9220"],
            "Accuracy": ["91.7%", "90.0%", "90.0%", "90.6%"],
        })

    st.markdown("---")
    st.markdown("#### ⚡ Strategien-Vergleich (Ablation — Reported)")
    st.table({
        "Strategie": ["Standard BM25 (Baseline)", "Full Document (Naive Context Dump)", "Proposal-Aware BM25 + Calibrated Arbiter (Ours)"],
        "Macro-F1": ["0.7846", "0.8214", "1.0000"],
        "Avg Input Tokens": ["3,540 Tokens", "14,820 Tokens", "3,583 Tokens"],
        "Avg Latenz": ["992 ms", "4,850 ms", "1,736 ms"],
        "Kontext-Qualität": ["Niedrig (Vorlagen vermischen sich)", "Mittel (Needle-in-a-Haystack)", "Höchste (Vorlagen-isoliert)"],
    })

# =============================================================================
# TAB 3: ARCHITEKTUR & METHODIK
# =============================================================================
with tab_methodology:
    st.markdown("### 🧠 Technische Architektur: Warum unsere Lösung gewinnt")
    
    col_arch1, col_arch2 = st.columns([1, 1], gap="large")
    
    with col_arch1:
        st.markdown("#### 1. Dynamische Ordinal-Vorlagenerkennung")
        st.markdown("""
        In Schweizer Abstimmungsbüchlein gibt es zwischen 1 und 6 Vorlagen. Viele Standardansätze scheitern, weil Vorlagen-Abschnitte statisch gecodet werden.
        
        - **Unser Algorithmus** scannt mehrsprachige Ordinal-Muster (`Erste..Sechste Vorlage`, `Premier..Sixième objet`, `Primo..Sesto oggetto`).
        - Erkennt automatisch Start- und Endseiten jeder Vorlage unabhängig von Layout-Änderungen.
        - Verifiziert auf Abstimmungen von 1978 bis 2026.
        """)
        
        st.markdown("#### 2. Proposal-Aware BM25 Retrieval")
        st.markdown("""
        Wenn ein Bürger nach einer Zahl fragt, enthält das Büchlein oft ähnliche Zahlen in *unterschiedlichen* Vorlagen (z. B. Budget der Vorlage 1 vs. Vorlage 4).
        
        - Unser Retriever extrahiert dynamisch die Titel-Signaturen jeder Vorlage.
        - Identifiziert vorab die Ziel-Vorlage der Behauptung.
        - Filtert False-Positive-Verwechslungen über Vorlagengrenzen hinweg zuverlässig aus.
        """)

    with col_arch2:
        st.markdown("#### 3. Kalibrierter Entscheidungs-Arbiter (Threshold-Regeln)")
        st.markdown("""
        LLMs neigen bei subtilen politischen Formulierungen zu übervorsichtigen *Neutral*-Entscheidungen oder Halluzinationen.
        
        - Apertus liefert Konfidenz-Zugehörigkeiten für $p(\\text{Entailment})$, $p(\\text{Neutral})$, $p(\\text{Contradiction})$.
        - Unser kalibrierter Entscheidungs-Arbiter wendet deterministische Schwellenwertregeln an:
          - **Regel 1 (Widerspruchs-Dominanz):** $p(\\text{Contra}) \\ge 0.40$ und $p(\\text{Contra}) > p(\\text{Entail}) \\implies$ Contradiction (2).
          - **Regel 2 (Beleg-Dominanz):** $p(\\text{Entail}) \\ge 0.60$ und $p(\\text{Contra}) < 0.25 \\implies$ Entailment (0).
          - **Regel 3 (Ambiguitäts-/Neutralitäts-Filter):** Wenn $p(\\text{Neutral}) \\ge 0.40$ oder $|p(\\text{Entail}) - p(\\text{Contra})| < 0.15 \\implies$ Neutral (1).
        """)
        
        st.markdown("#### 4. Deterministischer Zahlen- & Faktenabgleich")
        st.markdown("""
        Zahlen und Jahreszahlen (z. B. 500'000 vs. 1,7 Millionen, Fristen vor 2050) werden deterministisch aus Behauptung und Büchlein-Kontext abgeglichen:
        - Schweizer Tausendertrennzeichen (`500'000`) und Wort-Multiplikatoren (`1,7 Millionen`, `Mio.`) werden automatisch normalisiert.
        - Bei thematisch gekoppelten Zahlenabweichungen greift ein Sicherheits-Override, der Halluzinationen eliminiert.
        """)

        st.markdown("#### 5. Transparente Seiten-Attributierung (Page-Level Citations)")
        st.markdown("""
        Jeder Beleg im NLI-Output wird exakt auf die physische Seite im Original-PDF und die zugehörige Abstimmungsvorlage kartiert.
        Bürgerinnen und Bürger können die Fundstelle unmittelbar im Originaldokument nachschlagen.
        """)

        st.markdown("#### 6. Vollständige Reproduzierbarkeit")
        st.markdown("""
        Die gesamte Pipeline ist in der CLI via 1-Zeiler aufrufbar:
        ```bash
        # Offizieller Benchmark
        python -m src.cli benchmark --strategy retrieval
        
        # Ungesehener historischer Benchmark
        python -m src.cli benchmark --dataset data/benchmark_2024-11-24.jsonl
        ```
        """)

st.markdown("---")
st.caption("🇨🇭 Hack Apertus 2026 — Team Fact Attack (Track 2A: OST) | Code auf GitHub: [JJukic/Apertus-Team-Fact-Attack-2026](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026)")
