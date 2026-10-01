import streamlit as st
import sys
import os
from pathlib import Path

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pdf_parser import PDFParser
from src.retriever import PassageRetriever
from src.apertus_client import ApertusClient
from src import config

st.set_page_config(page_title="Fact Attack - Apertus Demo", page_icon="🇨🇭", layout="wide")

st.title("🇨🇭 Fact Attack 2026 - Voting Booklet NLI")
st.markdown("""
**Track 2A: OST** — Multilingual Natural Language Inference over Swiss Official Voting Booklets.
Verify claims directly against the official *Abstimmungsbüchlein* using **Apertus-v1.5-8B** on CSCS Alps.
""")

@st.cache_resource
def get_client():
    return ApertusClient()

@st.cache_resource
def get_parser():
    return PDFParser()

@st.cache_data
def get_booklet_data(pdf_path: str):
    parser = get_parser()
    paras = parser.extract_paragraphs(pdf_path)
    return paras

client = get_client()

booklet_options = {
    "German (2026-06-14_de.pdf)": "data/booklets/2026-06-14_de.pdf",
    "French (2026-06-14_fr.pdf)": "data/booklets/2026-06-14_fr.pdf",
    "Italian (2026-06-14_it.pdf)": "data/booklets/2026-06-14_it.pdf",
}

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Config")
    selected_booklet = st.selectbox("Select Booklet", options=list(booklet_options.keys()))
    booklet_path = booklet_options[selected_booklet]
    lang = "de" if "_de" in booklet_path else ("fr" if "_fr" in booklet_path else "it")
    
    st.info(f"Loaded {len(get_booklet_data(booklet_path))} paragraphs.")
    
    st.markdown("### Predefined Samples")
    samples = [
        "Der Bundesrat empfiehlt, dass die Initiative abgelehnt wird.",
        "Il Consiglio federale raccomanda che l'iniziativa venga approvata e che si voti sì.",
        "Le Conseil fédéral recommande de rejeter l'initiative."
    ]
    for s in samples:
        if st.button(s):
            st.session_state.claim_input = s

with col2:
    st.subheader("Inference")
    claim = st.text_area("Enter your claim:", value=st.session_state.get("claim_input", ""), height=100)
    
    if st.button("Verify Claim", type="primary"):
        if claim:
            with st.spinner("Analyzing with Apertus & Proposal-Aware Retriever..."):
                # 1. Retrieve
                paras = get_booklet_data(booklet_path)
                retriever = PassageRetriever(paras)
                top_paras = retriever.retrieve(claim, top_k=5)
                context = "\n\n".join([f"[Page {p['page_number']}] {p['text']}" for p in top_paras])
                
                # 2. Infer
                out = client.infer(context=context, claim=claim, claim_language=lang)
                
            st.success(f"Inference complete in {out.latency_ms:.0f} ms")
            
            # Display results
            label_name = {0: "✅ Entailment", 1: "⚪ Neutral", 2: "❌ Contradiction"}[out.label]
            st.metric(label="Prediction", value=label_name)
            
            st.write(f"**Fuzzy Logic Confidence:** Entail: {out.p_entail:.2f} | Neutral: {out.p_neutral:.2f} | Contra: {out.p_contra:.2f}")
            if out.fuzzy_rule_applied:
                st.caption(f"Applied Rule: {out.fuzzy_rule_applied}")
            
            st.markdown("### Reasoning")
            st.write(out.reasoning)
            
            st.markdown("### Extracted Evidence")
            for ev in out.evidence:
                st.info(f'"{ev}"')
                
            with st.expander("Show Retrieved Context"):
                for p in top_paras:
                    st.markdown(f"**Page {p['page_number']}** (Score: {p.get('retrieval_score', 'N/A')})")
                    st.write(p['text'])
                    st.divider()
        else:
            st.warning("Please enter a claim.")
