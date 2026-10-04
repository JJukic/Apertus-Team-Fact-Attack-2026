"""
Fact Attack 2026 - Multilingual Voting Booklet NLI & Fact-Checker
Streamlit Web Application for Hack Apertus Track 2A (OST).
Showcases document-grounded claim verification using Apertus-v1.5-8B on CSCS Alps.
"""

import sys
import os
import json
import time
import re
import html
from pathlib import Path
from typing import Optional, List, Dict, Any
import streamlit as st

# Setup paths
_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from src.pdf_parser import PDFParser
from src.retriever import PassageRetriever
from src.apertus_client import ApertusClient
from src.inference import ClaimVerificationEngine, PredictionResult, EvidenceSource
from src.evaluator import BenchmarkEvaluator
from src.results_summary import build_summary
from src.text_utils import best_snippet
from src import config

# Page Configuration
st.set_page_config(
    page_title="Fact Attack 2026",
    page_icon="🇨🇭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# DESIGN SYSTEM — Premium minimal Swiss-inspired dark UI
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ══════════════════════════════════════════════════════════════
       GLOBAL DARK THEME — Forces dark on EVERY Streamlit widget
       ══════════════════════════════════════════════════════════════ */

    /* ── Reset & Base ──────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e4e8ef;
    }

    .stApp {
        background: #08090c !important;
        color: #e4e8ef !important;
    }

    /* Force dark on all Streamlit internal containers */
    .stApp > div,
    .stApp > div > div,
    section[data-testid="stSidebar"],
    div[data-testid="stAppViewBlockContainer"] {
        background-color: transparent !important;
    }

    /* Hide Streamlit header chrome & footer */
    header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
    }
    footer, #MainMenu { display: none !important; }

    .block-container {
        padding-top: 1.6rem !important;
        padding-bottom: 2rem !important;
        max-width: 1240px !important;
    }

    /* ── Ambient glow (top-of-page accent) ─────────────────────── */
    .ambient-glow {
        position: fixed;
        top: -120px;
        left: 50%;
        transform: translateX(-50%);
        width: 600px;
        height: 320px;
        background: radial-gradient(ellipse, rgba(213,43,30,0.08) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }

    /* ══════════════════════════════════════════════════════════════
       STREAMLIT WIDGET OVERRIDES — Comprehensive dark theme
       ══════════════════════════════════════════════════════════════ */

    /* ── ALL Text Inputs (input, textarea) & BaseWeb Wrappers ───── */
    div[data-baseweb="base-input"],
    div[data-baseweb="input"],
    div[data-baseweb="textarea"],
    div[data-baseweb="base-input"] > div,
    div[data-baseweb="textarea"] > div,
    [data-testid="stTextInputRootElement"],
    [data-testid="stTextArea"] > div,
    [data-testid="stTextInput"] > div {
        background-color: #12151c !important;
        background: #12151c !important;
        border-color: rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
    }

    input, textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #12151c !important;
        background-color: #12151c !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
        -webkit-text-fill-color: #f1f5f9 !important;
        caret-color: #d52b1e !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }
    input:focus, textarea:focus,
    div[data-baseweb="base-input"]:focus-within,
    div[data-baseweb="textarea"]:focus-within {
        border-color: rgba(213,43,30,0.6) !important;
        box-shadow: 0 0 0 2px rgba(213,43,30,0.15) !important;
        outline: none !important;
        background-color: #141822 !important;
    }
    input::placeholder, textarea::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
    }
    textarea {
        border-radius: 10px !important;
    }

    /* ── Selectbox (dropdown trigger) ──────────────────────────── */
    [data-testid="stSelectbox"] > div > div {
        background: #12151c !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stSelectbox"] > div > div:hover {
        border-color: rgba(255,255,255,0.18) !important;
    }
    [data-testid="stSelectbox"] svg {
        fill: #94a3b8 !important;
    }
    [data-testid="stSelectbox"] label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    /* ── Selectbox Dropdown Popup (listbox) ────────────────────── */
    div[data-baseweb="popover"] {
        background: #12151c !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
    }
    div[data-baseweb="popover"] ul {
        background: #12151c !important;
    }
    div[data-baseweb="popover"] li {
        background: transparent !important;
        color: #cbd5e1 !important;
    }
    div[data-baseweb="popover"] li:hover {
        background: rgba(255,255,255,0.06) !important;
        color: #f1f5f9 !important;
    }
    div[data-baseweb="popover"] li[aria-selected="true"] {
        background: rgba(213,43,30,0.15) !important;
        color: #fca5a5 !important;
    }

    /* ── BaseWeb Select internals ─────────────────────────────── */
    div[data-baseweb="select"] > div {
        background: #12151c !important;
        border-color: rgba(255,255,255,0.1) !important;
        color: #e2e8f0 !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(255,255,255,0.18) !important;
    }
    div[data-baseweb="select"] span {
        color: #e2e8f0 !important;
    }
    div[data-baseweb="select"] svg {
        fill: #94a3b8 !important;
        color: #94a3b8 !important;
    }
    div[data-baseweb="select"] input {
        color: #e2e8f0 !important;
    }
    /* Dropdown menu container */
    div[data-baseweb="menu"] {
        background: #12151c !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    div[data-baseweb="menu"] li {
        background: transparent !important;
        color: #cbd5e1 !important;
    }
    div[data-baseweb="menu"] li:hover {
        background: rgba(255,255,255,0.06) !important;
    }

    /* ── Multiselect ──────────────────────────────────────────── */
    [data-testid="stMultiSelect"] > div > div {
        background: #12151c !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background: rgba(213,43,30,0.15) !important;
        color: #fca5a5 !important;
        border: 1px solid rgba(213,43,30,0.3) !important;
    }

    /* ── Radio Buttons ────────────────────────────────────────── */
    .stRadio label, .stSelectbox label, .stTextArea label,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    .stRadio div[role="radiogroup"] label {
        color: #cbd5e1 !important;
        font-size: 0.85rem !important;
    }
    .stRadio div[role="radiogroup"] label:hover {
        color: #f1f5f9 !important;
    }
    /* Radio button circle */
    .stRadio div[role="radiogroup"] label > div:first-child {
        border-color: rgba(255,255,255,0.2) !important;
    }

    /* ── Checkbox ─────────────────────────────────────────────── */
    .stCheckbox label {
        color: #cbd5e1 !important;
    }
    .stCheckbox label > span:first-child {
        border-color: rgba(255,255,255,0.2) !important;
        background: transparent !important;
    }

    /* ── Slider ───────────────────────────────────────────────── */
    [data-testid="stSlider"] label {
        color: #94a3b8 !important;
    }
    [data-testid="stSlider"] div[data-baseweb="slider"] div {
        background: rgba(255,255,255,0.08) !important;
    }

    /* ── Number Input ─────────────────────────────────────────── */
    [data-testid="stNumberInput"] input {
        background: #12151c !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stNumberInput"] button {
        background: rgba(255,255,255,0.04) !important;
        color: #94a3b8 !important;
        border-color: rgba(255,255,255,0.1) !important;
    }

    /* ── Tabs ─────────────────────────────────────────────────── */
    [data-testid="stTabs"] {
        background: transparent !important;
    }
    [data-testid="stTabs"] button {
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: #64748b !important;
        border-bottom-color: transparent !important;
        padding: 0.6rem 1rem !important;
        background: transparent !important;
    }
    [data-testid="stTabs"] button:hover {
        color: #94a3b8 !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #f1f5f9 !important;
        border-bottom-color: #d52b1e !important;
        background: transparent !important;
    }

    /* ── Expander ─────────────────────────────────────────────── */
    [data-testid="stExpander"],
    details[data-testid="stExpander"] {
        background: #0f1219 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        margin-bottom: 0.8rem !important;
        box-shadow: none !important;
        overflow: hidden !important;
    }
    [data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary,
    [data-testid="stExpanderToggleIcon"] {
        background-color: #0f1219 !important;
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stExpander"] summary:hover,
    details[data-testid="stExpander"] summary:hover {
        background-color: #141822 !important;
        color: #ffffff !important;
    }
    [data-testid="stExpander"] summary svg,
    details[data-testid="stExpander"] summary svg {
        color: #94a3b8 !important;
        fill: #94a3b8 !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: #0a0d13 !important;
        border-top: 1px solid rgba(255,255,255,0.05) !important;
        padding: 1.2rem !important;
    }
    [data-testid="stExpander"] > div > div {
        background: transparent !important;
    }

    /* ── File Uploader ────────────────────────────────────────── */
    [data-testid="stFileUploader"] { background: transparent !important; }
    [data-testid="stFileUploaderDropzone"] {
        background: #12151c !important;
        border: 1px dashed rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] p {
        color: #64748b !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: rgba(255,255,255,0.04) !important;
        color: #94a3b8 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 6px !important;
    }

    /* ── Primary Button ───────────────────────────────────────── */
    button[kind="primary"] {
        background: #d52b1e !important;
        color: #fff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(213,43,30,0.3), 0 0 0 1px rgba(213,43,30,0.5) !important;
        transition: all 0.15s ease !important;
        letter-spacing: 0.01em !important;
    }
    button[kind="primary"] p, button[kind="primary"] div, button[kind="primary"] span {
        color: #fff !important;
        -webkit-text-fill-color: #fff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    button[kind="primary"]:hover {
        background: #b91c1c !important;
        box-shadow: 0 6px 20px rgba(213,43,30,0.45), 0 0 0 1px rgba(213,43,30,0.6) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Secondary / All Other Buttons ─────────────────────────── */
    button[kind="secondary"],
    .stButton > button:not([kind="primary"]),
    .stDownloadButton > button {
        background: rgba(255,255,255,0.04) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        transition: all 0.15s ease !important;
    }
    button[kind="secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover,
    .stDownloadButton > button:hover {
        background: rgba(255,255,255,0.08) !important;
        color: #f1f5f9 !important;
        border-color: rgba(255,255,255,0.16) !important;
    }

    /* ── Alert Boxes (st.success, st.info, st.warning, st.error) ─ */
    [data-testid="stAlert"],
    .stAlert,
    div[role="alert"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #cbd5e1 !important;
        border-radius: 8px !important;
    }
    [data-testid="stAlert"] p,
    .stAlert p,
    div[role="alert"] p {
        color: #cbd5e1 !important;
    }
    /* Success variant */
    div[data-testid="stAlert"][data-baseweb*="positive"],
    .element-container .stSuccess {
        border-left: 3px solid #10b981 !important;
    }
    /* Info variant */
    .stInfo, [data-testid="stNotification"] {
        background: rgba(59,130,246,0.06) !important;
        border: 1px solid rgba(59,130,246,0.15) !important;
        color: #93c5fd !important;
        border-radius: 8px !important;
    }
    .stInfo p, [data-testid="stNotification"] p {
        color: #93c5fd !important;
    }

    /* ── Metric Widget ────────────────────────────────────────── */
    [data-testid="stMetric"],
    [data-testid="stMetricValue"],
    [data-testid="metric-container"] {
        background: transparent !important;
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #94a3b8 !important;
    }

    /* ── Spinner ──────────────────────────────────────────────── */
    .stSpinner > div {
        background: transparent !important;
        color: #94a3b8 !important;
    }
    .stSpinner > div > div {
        border-top-color: #d52b1e !important;
    }

    /* ── Caption ──────────────────────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #475569 !important;
    }
    .stCaption p, [data-testid="stCaptionContainer"] p {
        color: #475569 !important;
    }

    /* ── Markdown Text ────────────────────────────────────────── */
    .stMarkdown p, .stMarkdown li, .stMarkdown h1,
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    .stMarkdown h5, .stMarkdown h6 {
        color: #e2e8f0 !important;
    }
    .stMarkdown a {
        color: #60a5fa !important;
    }
    .stMarkdown code {
        background: rgba(255,255,255,0.06) !important;
        color: #f0abfc !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
    .stMarkdown pre {
        background: #0c0e14 !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 8px !important;
    }
    .stMarkdown pre code {
        background: transparent !important;
        color: #cbd5e1 !important;
    }
    /* LaTeX / KaTeX */
    .katex, .katex .mord, .katex .mbin, .katex .mrel,
    .katex .mopen, .katex .mclose, .katex .mpunct, .katex .minner {
        color: #e2e8f0 !important;
    }

    /* ── Table ────────────────────────────────────────────────── */
    [data-testid="stTable"] {
        border-radius: 10px;
        overflow: hidden;
    }
    [data-testid="stTable"] table {
        background: transparent !important;
    }
    [data-testid="stTable"] th {
        background: rgba(255,255,255,0.04) !important;
        color: #64748b !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        border-color: rgba(255,255,255,0.05) !important;
    }
    [data-testid="stTable"] td {
        background: rgba(255,255,255,0.015) !important;
        color: #cbd5e1 !important;
        font-size: 0.85rem !important;
        border-color: rgba(255,255,255,0.04) !important;
    }

    /* ── DataFrame / DataEditor ───────────────────────────────── */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    .stDataFrame {
        background: transparent !important;
    }
    [data-testid="stDataFrame"] [data-testid="glideDataEditor"] {
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 8px !important;
    }

    /* ── Progress bar ─────────────────────────────────────────── */
    [data-testid="stProgressBar"] > div > div {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 3px !important;
    }

    /* ── Code Block ───────────────────────────────────────────── */
    [data-testid="stCode"],
    .stCodeBlock {
        background: #0c0e14 !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 8px !important;
    }
    [data-testid="stCode"] pre,
    .stCodeBlock pre {
        background: transparent !important;
        color: #cbd5e1 !important;
    }
    /* Copy button in code blocks */
    [data-testid="stCode"] button,
    .stCodeBlock button {
        color: #64748b !important;
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    /* ── Tooltip ──────────────────────────────────────────────── */
    [data-testid="stTooltipIcon"] {
        color: #475569 !important;
    }

    /* ── Divider (st.divider / hr) ────────────────────────────── */
    hr {
        border-color: rgba(255,255,255,0.06) !important;
    }

    /* ── Scrollbar ────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }


    /* ══════════════════════════════════════════════════════════════
       CUSTOM COMPONENT STYLES
       ══════════════════════════════════════════════════════════════ */

    /* ── Top Bar ───────────────────────────────────────────────── */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.4rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .topbar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .topbar-brand .logo-mark {
        width: 28px;
        height: 28px;
        background: #d52b1e;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: 800;
        color: #fff;
        box-shadow: 0 0 16px rgba(213,43,30,0.35);
    }
    .topbar-brand .brand-text {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #f1f5f9;
    }
    .topbar-meta {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .topbar-tag {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748b;
        padding: 3px 10px;
        border: 1px solid rgba(100,116,139,0.25);
        border-radius: 4px;
    }
    .topbar-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px rgba(16,185,129,0.6);
    }

    /* ── Hero Title ────────────────────────────────────────────── */
    .hero-section {
        text-align: center;
        margin: 0.2rem 0 1rem 0;
    }
    .hero-section h1 {
        font-size: 1.7rem;
        font-weight: 900;
        letter-spacing: -0.035em;
        line-height: 1.15;
        margin: 0 0 0.6rem 0;
        background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-section p {
        font-size: 1.05rem;
        color: #64748b !important;
        font-weight: 400;
        line-height: 1.6;
        max-width: 560px;
        margin: 0 auto;
        -webkit-text-fill-color: #64748b !important;
    }
    .hero-section p strong {
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
        font-weight: 600;
    }

    /* ── Claim Input Label ─────────────────────────────────────── */
    .claim-stage-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.7rem;
    }

    /* ── Verdict Display ───────────────────────────────────────── */
    .verdict-container {
        border-radius: 16px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
    }
    .verdict-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .verdict-entailment {
        background: rgba(16, 185, 129, 0.06);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .verdict-entailment::before { background: #10b981; }
    .verdict-neutral {
        background: rgba(245, 158, 11, 0.06);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    .verdict-neutral::before { background: #f59e0b; }
    .verdict-contradiction {
        background: rgba(239, 68, 68, 0.06);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    .verdict-contradiction::before { background: #ef4444; }

    .verdict-label {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .verdict-label-entailment { color: #34d399 !important; -webkit-text-fill-color: #34d399 !important; }
    .verdict-label-neutral    { color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; }
    .verdict-label-contradiction { color: #f87171 !important; -webkit-text-fill-color: #f87171 !important; }

    .verdict-icon {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 800;
    }
    .icon-entailment { background: rgba(16,185,129,0.2); color: #34d399; }
    .icon-neutral    { background: rgba(245,158,11,0.2); color: #fbbf24; }
    .icon-contradiction { background: rgba(239,68,68,0.2); color: #f87171; }

    .verdict-title {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.3;
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        margin: 0 0 0.4rem 0;
    }
    .verdict-reasoning {
        font-size: 0.92rem;
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
        line-height: 1.6;
        margin: 0;
    }

    /* ── Guardrail Banner ──────────────────────────────────────── */
    .guardrail-banner {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        font-size: 0.82rem;
    }
    .guardrail-ok {
        background: rgba(16,185,129,0.06);
        border: 1px solid rgba(16,185,129,0.15);
        color: #6ee7b7;
    }
    .guardrail-num {
        background: rgba(239,68,68,0.06);
        border: 1px solid rgba(239,68,68,0.15);
        color: #fca5a5;
    }
    .guardrail-vac {
        background: rgba(245,158,11,0.06);
        border: 1px solid rgba(245,158,11,0.15);
        color: #fde68a;
    }
    .guardrail-icon { font-size: 1.1rem; flex-shrink: 0; }
    .guardrail-text strong { font-weight: 700; }

    /* ── Evidence Block ────────────────────────────────────────── */
    .evidence-block {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .evidence-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.8rem;
    }
    .evidence-tag {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .tag-page {
        background: rgba(213,43,30,0.12);
        color: #fca5a5;
        border: 1px solid rgba(213,43,30,0.3);
    }
    .tag-proposal {
        background: rgba(59,130,246,0.12);
        color: #93c5fd;
        border: 1px solid rgba(59,130,246,0.3);
    }
    .evidence-quote {
        font-size: 1rem;
        line-height: 1.65;
        color: #e2e8f0;
        border-left: 3px solid #d52b1e;
        padding-left: 1rem;
        margin: 0;
        font-style: italic;
    }

    /* ── Comparison bars (benchmark) ───────────────────────────── */
    .bar-chart { padding: 1rem 0 0.4rem 0; }
    .bar-title { font-size: 0.8rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.7rem; }
    .bar-row { display: grid; grid-template-columns: 150px 1fr 70px; align-items: center; gap: 10px; margin: 8px 0; cursor: default; }
    .bar-name { font-size: 0.82rem; color: #94a3b8; }
    .bar-name.bar-ours { color: #f8fafc; font-weight: 700; }
    .bar-track { height: 18px; background: rgba(255,255,255,0.03); border-radius: 4px; }
    .bar-fill { height: 100%; background: #64748b; border-radius: 0 4px 4px 0; }
    .bar-fill-ours { background: #3987e5; }
    .bar-row:hover .bar-fill { filter: brightness(1.25); }
    .bar-value { font-size: 0.85rem; font-weight: 700; color: #e2e8f0; text-align: right; font-variant-numeric: tabular-nums; }
    .bar-note { font-size: 0.72rem; color: #64748b; margin-top: 0.4rem; }

    /* ── Language-pair matrix ──────────────────────────────────── */
    .mx-grid { display: grid; grid-template-columns: 170px repeat(3, 1fr); gap: 2px; max-width: 640px; margin: 0.6rem 0; }
    .mx-corner { font-size: 0.7rem; color: #64748b; display: flex; align-items: center; }
    .mx-head { font-size: 0.8rem; font-weight: 700; color: #cbd5e1; display: flex; align-items: center; justify-content: center; padding: 6px; }
    .mx-cell { border-radius: 4px; padding: 10px 6px; text-align: center; display: flex; flex-direction: column; gap: 2px; cursor: default; }
    .mx-cell:hover { outline: 2px solid rgba(255,255,255,0.5); }
    .mx-val { font-size: 1.05rem; font-weight: 800; color: #ffffff; font-variant-numeric: tabular-nums; }
    .mx-n { font-size: 0.68rem; color: #e2e8f0; opacity: 0.8; }

    /* ── Architecture flow ─────────────────────────────────────── */
    .flow { display: flex; align-items: stretch; gap: 10px; margin: 1rem 0 0.5rem 0; }
    .flow-step { flex: 1; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1rem 1.1rem; background: rgba(255,255,255,0.02); }
    .flow-step-apertus { border-color: rgba(213,43,30,0.5); background: rgba(213,43,30,0.06); }
    .flow-num { width: 24px; height: 24px; border-radius: 50%; background: #d52b1e; color: #fff; font-size: 0.75rem; font-weight: 800; display: flex; align-items: center; justify-content: center; margin-bottom: 0.6rem; }
    .flow-title { font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.35rem; }
    .flow-text { font-size: 0.8rem; color: #94a3b8; line-height: 1.5; }
    .flow-arrow { display: flex; align-items: center; color: #475569; font-size: 1.3rem; }
    .tag-section {
        background: rgba(148,163,184,0.10);
        color: #cbd5e1;
        border: 1px solid rgba(148,163,184,0.25);
    }

    /* ── Efficiency line, page chips, story, notices ───────────── */
    .eff-line {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem 1.4rem;
        font-size: 0.85rem;
        color: #94a3b8;
        margin: -0.4rem 0 1.2rem 0.2rem;
    }
    .eff-line strong { color: #e2e8f0; font-weight: 700; }
    .page-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.6rem; }
    .page-chip {
        font-size: 0.75rem;
        padding: 3px 9px;
        border-radius: 999px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        color: #cbd5e1;
    }
    .story-desc {
        font-size: 0.88rem;
        line-height: 1.5;
        color: #cbd5e1;
        margin: 0.3rem 0 1rem 0;
        padding: 0.6rem 0.8rem;
        border-left: 3px solid #d52b1e;
        background: rgba(255,255,255,0.02);
        border-radius: 0 6px 6px 0;
    }
    .story-desc strong { color: #f8fafc; }

    /* ── Verdict stats (efficiency per check) ──────────────────── */
    .verdict-stats {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.8rem;
        margin-top: 1.3rem;
        padding-top: 1.1rem;
        border-top: 1px solid rgba(255,255,255,0.08);
    }
    .verdict-stats > div { display: flex; flex-direction: column; gap: 2px; }
    .vs-num { font-size: 1.15rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em; }
    .vs-lbl { font-size: 0.72rem; color: #94a3b8; line-height: 1.3; }

    /* ── Result placeholder ────────────────────────────────────── */
    .result-placeholder {
        border: 1px dashed rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-top: 1.7rem;
    }
    .outcome-row { display: flex; align-items: center; gap: 10px; margin: 0.7rem 0; color: #cbd5e1; font-size: 0.92rem; }
    .outcome-row strong { color: #f8fafc; }
    .outcome-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    .dot-entailment { background: #10b981; }
    .dot-neutral { background: #f59e0b; }
    .dot-contradiction { background: #ef4444; }
    .notice-box {
        padding: 0.7rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        background: rgba(59,130,246,0.07);
        border: 1px solid rgba(59,130,246,0.2);
        color: #bfdbfe;
    }
    .disclaimer {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.4rem;
    }

    /* ── Document Facsimile ────────────────────────────────────── */
    .doc-facsimile {
        background: #0c0e14;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        max-height: 340px;
        overflow-y: auto;
        font-size: 0.88rem;
        line-height: 1.65;
        color: #94a3b8;
    }
    .doc-facsimile-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #475569;
        padding-bottom: 0.6rem;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    mark.hl-entailment {
        background: rgba(16,185,129,0.35);
        color: #fff;
        font-weight: 600;
        padding: 2px 5px;
        border-radius: 3px;
        border-bottom: 2px solid #10b981;
    }
    mark.hl-contradiction {
        background: rgba(239,68,68,0.35);
        color: #fff;
        font-weight: 600;
        padding: 2px 5px;
        border-radius: 3px;
        border-bottom: 2px solid #ef4444;
    }
    mark.hl-neutral {
        background: rgba(245,158,11,0.35);
        color: #fff;
        font-weight: 600;
        padding: 2px 5px;
        border-radius: 3px;
        border-bottom: 2px solid #f59e0b;
    }

    /* ── Confidence Meter ──────────────────────────────────────── */
    .conf-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.45rem;
    }
    .conf-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        width: 100px;
        text-align: right;
        flex-shrink: 0;
    }
    .conf-bar-track {
        flex: 1;
        height: 6px;
        background: rgba(255,255,255,0.04);
        border-radius: 3px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s ease;
    }
    .conf-value {
        font-size: 0.75rem;
        font-weight: 700;
        color: #cbd5e1;
        width: 50px;
        flex-shrink: 0;
    }

    /* ── KPI Grid (Benchmark) ─────────────────────────────────── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .kpi-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    /* ── Fact Card SVG Container ───────────────────────────────── */
    .fact-card-wrap {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }

    /* ── Section Divider ───────────────────────────────────────── */
    .section-divider {
        height: 1px;
        background: rgba(255,255,255,0.05);
        margin: 1.6rem 0;
    }

    /* ── Animation ────────────────────────────────────────────── */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeUp 0.4s ease both;
    }

</style>
""", unsafe_allow_html=True)


# =============================================================================
# Cache Resources
# =============================================================================
@st.cache_resource
def get_engine():
    return ClaimVerificationEngine()

@st.cache_resource
def get_parser():
    return PDFParser()

engine = get_engine()
parser = get_parser()


# =============================================================================
# HELPER: VISUAL PDF FACSIMILE HIGHLIGHTING
# =============================================================================
def highlight_evidence_in_page(page_text: str, quote: str, label: int) -> str:
    """Highlights the exact evidence quote within the raw page text, handling arbitrary PDF linebreaks."""
    if not quote or not page_text:
        return html.escape(page_text)

    # Strip tags and quotation marks
    clean_quote = re.sub(r"^\[(?:page|seite)\s+\d+\]\s*\d*\s*", "", quote, flags=re.IGNORECASE).strip(' "«»')
    cls_name = "hl-entailment" if label == 0 else ("hl-contradiction" if label == 2 else "hl-neutral")

    # Normalize words
    words = [re.escape(w) for w in clean_quote.split() if len(w) > 0]
    if not words:
        return html.escape(page_text)

    # Search with flexible whitespace \s+ (matches spaces, tabs, newlines across tokens)
    for span_len in (len(words), min(len(words), 14), min(len(words), 8), min(len(words), 5), min(len(words), 3)):
        if span_len < 2:
            break
        pattern_str = r"\s+".join(words[:span_len])
        try:
            match = re.search(pattern_str, page_text, re.IGNORECASE)
            if match:
                start, end = match.span()
                before = html.escape(page_text[:start])
                matched = html.escape(page_text[start:end])
                after = html.escape(page_text[end:])
                return f"{before}<mark class=\"{cls_name}\">{matched}</mark>{after}"
        except re.error:
            pass

    return html.escape(page_text)


# =============================================================================
# HELPER: SHAREABLE SOCIAL MEDIA FACT-CARD (SVG)
# =============================================================================
def generate_fact_card_svg(
    claim: str,
    label: int,
    quote: str,
    page_number: Optional[int],
    proposal_id: Optional[int],
    date_str: str,
) -> str:
    """Generates a high-resolution, vector SVG Fact-Card for social media & sharing."""
    if label == 0:
        verdict_color = "#10b981"
        verdict_text = "GESTÜTZT · ENTAILMENT"
        verdict_icon = "✓"
        bg_glow = "rgba(16, 185, 129, 0.15)"
    elif label == 1:
        verdict_color = "#f59e0b"
        verdict_text = "NICHT GEKLÄRT · NEUTRAL"
        verdict_icon = "?"
        bg_glow = "rgba(245, 158, 11, 0.15)"
    else:
        verdict_color = "#ef4444"
        verdict_text = "WIDERSPRUCH · CONTRADICTION"
        verdict_icon = "✗"
        bg_glow = "rgba(239, 68, 68, 0.15)"

    clean_claim = (claim[:120] + "...") if len(claim) > 120 else claim
    clean_quote = (quote[:140] + "...") if len(quote) > 140 else quote
    page_str = f"Seite {page_number}" if page_number else "Amtliche Quelle"
    prop_str = f" · Vorlage {proposal_id}" if proposal_id else ""

    e_claim = html.escape(clean_claim)
    e_quote = html.escape(clean_quote)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="420" viewBox="0 0 760 420" style="max-width: 100%; height: auto;">
  <defs>
    <linearGradient id="cardGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#090d16"/>
    </linearGradient>
  </defs>

  <!-- Outer Card Frame -->
  <rect width="760" height="420" rx="20" fill="url(#cardGrad)" stroke="#334155" stroke-width="2"/>

  <!-- Top Swiss Banner -->
  <path d="M 0 20 Q 0 0 20 0 L 740 0 Q 760 0 760 20 L 760 52 L 0 52 Z" fill="#d52b1e"/>

  <!-- Swiss Cross Icon -->
  <rect x="24" y="15" width="22" height="22" rx="4" fill="white"/>
  <rect x="27" y="23" width="16" height="6" fill="#d52b1e"/>
  <rect x="32" y="18" width="6" height="16" fill="#d52b1e"/>

  <!-- Banner Text -->
  <text x="56" y="31" fill="#ffffff" font-family="'Inter', sans-serif" font-size="13" font-weight="700" letter-spacing="1">FAKTENCHECK GEGEN DAS ABSTIMMUNGSBÜCHLEIN · FACT ATTACK</text>
  <text x="736" y="31" fill="#ffffff" text-anchor="end" font-family="'Inter', sans-serif" font-size="11" opacity="0.9">KEIN AMTLICHES DOKUMENT</text>

  <!-- Verdict Stamp -->
  <rect x="36" y="74" width="340" height="48" rx="10" fill="{bg_glow}" stroke="{verdict_color}" stroke-width="2"/>
  <text x="52" y="104" fill="{verdict_color}" font-family="'Inter', sans-serif" font-size="19" font-weight="800" letter-spacing="1">{verdict_icon} {verdict_text}</text>

  <!-- Reference Meta -->
  <rect x="396" y="74" width="328" height="48" rx="10" fill="#1e293b" stroke="#475569" stroke-width="1"/>
  <text x="412" y="93" fill="#94a3b8" font-family="'Inter', sans-serif" font-size="10" font-weight="600" text-transform="uppercase">REFERENZ-DOKUMENT</text>
  <text x="412" y="110" fill="#f8fafc" font-family="'Inter', sans-serif" font-size="12" font-weight="600">{page_str}{prop_str} · {date_str}</text>

  <!-- Claim Section -->
  <text x="36" y="152" fill="#94a3b8" font-family="'Inter', sans-serif" font-size="11" font-weight="700" text-transform="uppercase" letter-spacing="0.5">GEPRÜFTE BEHAUPTUNG:</text>
  <rect x="36" y="164" width="688" height="64" rx="10" fill="#141c2e" stroke="#253248" stroke-width="1"/>
  <foreignObject x="48" y="172" width="664" height="48">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: 'Inter', sans-serif; font-size: 13px; color: #f1f5f9; line-height: 1.4; font-weight: 500;">
      "{e_claim}"
    </div>
  </foreignObject>

  <!-- Evidence Section -->
  <text x="36" y="254" fill="#94a3b8" font-family="'Inter', sans-serif" font-size="11" font-weight="700" text-transform="uppercase" letter-spacing="0.5">BELEG AUS DEM ABSTIMMUNGSBÜCHLEIN ({page_str}):</text>
  <rect x="36" y="266" width="688" height="82" rx="10" fill="#181e2b" stroke="{verdict_color}" stroke-opacity="0.4" stroke-width="1.5"/>
  <rect x="36" y="266" width="5" height="82" rx="2" fill="{verdict_color}"/>
  <foreignObject x="52" y="274" width="658" height="66">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: 'Inter', sans-serif; font-size: 12.5px; color: #e2e8f0; font-style: italic; line-height: 1.45;">
      "{e_quote}"
    </div>
  </foreignObject>

  <!-- Footer -->
  <line x1="36" y1="372" x2="724" y2="372" stroke="#334155" stroke-width="1"/>
  <text x="36" y="395" fill="#64748b" font-family="'Inter', sans-serif" font-size="11">Geprüft mit <b>Apertus v1.5</b> (CSCS) · Kein politischer Rat</text>
  <text x="724" y="395" fill="#64748b" text-anchor="end" font-family="'Inter', sans-serif" font-size="11">Quelle: Abstimmungsbüchlein, bk.admin.ch</text>
</svg>"""
    return svg


# =============================================================================
# DEMO DATA & HELPERS FOR THE FACT-CHECK TAB
# =============================================================================
BOOKLET_CATALOG = {
    "Juni 2026 — Nachhaltigkeitsinitiative & Zivildienst": "2026-06-14",
    "November 2024 — Nationalstrassen, Mietrecht, EFAS": "2024-11-24",
    "September 2024 — Biodiversität & BVG-Reform": "2024-09-22",
}
VOTE_AUTO = "Automatisch erkennen"

# Wording describes the relationship to the booklet, never a verdict on truth (OST ethics guideline)
VERDICTS = {
    0: ("entailment", "✓", "Vom Abstimmungsbüchlein gestützt",
        "Das Büchlein enthält Aussagen, die diese Behauptung stützen."),
    1: ("neutral", "?", "Im Abstimmungsbüchlein nicht geklärt",
        "Das Büchlein enthält keine Aussage, die diese Behauptung bestätigt oder widerlegt."),
    2: ("contradiction", "✗", "Widerspricht dem Abstimmungsbüchlein",
        "Das Büchlein enthält Aussagen, die dieser Behauptung widersprechen."),
}

SECTION_DE = {
    "Arguments of the initiative/referendum committee": "Argumente des Komitees",
    "Arguments of the Federal Council and Parliament": "Argumente Bundesrat & Parlament",
    "Voting text (legal text)": "Abstimmungstext",
    "Official explanation in detail": "Erläuterungen im Detail",
    "Overview / summary": "Übersicht",
}


@st.cache_data
def load_demo_data():
    cases = json.loads((config.DATA_DIR / "demo_cases.json").read_text(encoding="utf-8"))["cases"]
    votes = json.loads((config.DATA_DIR / "demo_votes.json").read_text(encoding="utf-8"))
    return cases, votes


DEMO_CASES, DEMO_VOTES = load_demo_data()


@st.cache_resource
def warm_up_demo_booklets():
    """Parse the demo booklets once at startup so the first live check is not slowed by PDF parsing."""
    for case in DEMO_CASES:
        pdf = config.BOOKLETS_DIR / f"{case['booklet_date']}_{case['booklet_language']}.pdf"
        if pdf.exists():
            engine._get_booklet_data(pdf)
    return True


warm_up_demo_booklets()


def _apply_case(case: dict, set_active: bool = True) -> None:
    """Button callback: load a demo case into the widgets (runs before the widgets are rebuilt)."""
    ballot = next(name for name, date in BOOKLET_CATALOG.items() if date == case["booklet_date"])
    st.session_state["ballot"] = ballot
    st.session_state["booklet_lang"] = case["booklet_language"].upper()
    st.session_state["vote"] = case["vote"]
    st.session_state["claim_input"] = case["claim"]
    st.session_state["claim_lang"] = case["claim_language"].upper()
    st.session_state["active_case"] = case["id"] if set_active else None
    st.session_state["auto_verify"] = set_active  # one click on an example runs the check


def find_cached_case(claim: str, booklet_file: str, vote: Optional[str]) -> Optional[dict]:
    for case in DEMO_CASES:
        if (case["claim"].strip() == claim.strip()
                and f"{case['booklet_date']}_{case['booklet_language']}.pdf" == booklet_file
                and case["vote"] == vote):
            return case
    return None


_LANG_HINTS = {
    "de": {"der", "die", "das", "und", "nicht", "ist", "wird", "dass", "mit", "für", "den", "eine", "laut", "bundesrat"},
    "fr": {"le", "la", "les", "des", "et", "est", "que", "une", "pour", "dans", "du", "selon", "conseil", "fédéral"},
    "it": {"il", "lo", "gli", "della", "che", "è", "per", "una", "del", "non", "secondo", "consiglio", "federale", "di"},
}


def guess_language(text: str) -> str:
    words = re.findall(r"[a-zàâçéèêëîïôûùüäöè]+", text.lower())
    scores = {lang: sum(w in hints for w in words) for lang, hints in _LANG_HINTS.items()}
    return max(scores, key=scores.get) if any(scores.values()) else "de"


# =============================================================================
# UI LAYOUT
# =============================================================================

# Ambient glow effect
st.markdown('<div class="ambient-glow"></div>', unsafe_allow_html=True)

# ── Top Bar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <div class="logo-mark">+</div>
        <span class="brand-text">Fact Attack</span>
    </div>
    <div class="topbar-meta">
        <span class="topbar-tag">Apertus v1.5 · CSCS Alps</span>
        <span class="topbar-tag">Track 2A (OST)</span>
        <div class="topbar-dot" title="Model Online"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <h1>Politische Behauptungen — am Abstimmungsbüchlein geprüft.</h1>
    <p>Gleicht Kampagnen-Claims mit den <strong>offiziellen Abstimmungsbüchlein</strong> des Bundesrates ab — seitengenau belegt, dreisprachig, nachvollziehbar.</p>
</div>
""", unsafe_allow_html=True)


# ── Navigation Tabs ──────────────────────────────────────────────────────────
tab_factcheck, tab_benchmark, tab_methodology = st.tabs([
    "Faktencheck",
    "Benchmark",
    "Architektur",
])


# =============================================================================
# TAB 1: INTERAKTIVER FAKTENCHECK
# =============================================================================
with tab_factcheck:
    if "ballot" not in st.session_state:
        _apply_case(DEMO_CASES[0], set_active=False)

    col_in, col_out = st.columns([1, 1.12], gap="large")

    # ── LEFT: input ──────────────────────────────────────────────────────────
    with col_in:
        c_ballot, c_lang = st.columns([1.5, 1])
        with c_ballot:
            selected_ballot_name = st.selectbox("Abstimmung", list(BOOKLET_CATALOG.keys()), key="ballot")
            selected_date = BOOKLET_CATALOG[selected_ballot_name]
        with c_lang:
            selected_lang = st.radio("Sprache des Büchleins", ["DE", "FR", "IT"], horizontal=True, key="booklet_lang")
            lang_code = selected_lang.lower()

        strat = "full" if st.session_state.get("strategy_option", "").startswith("Ganzes") else "hybrid"
        uploaded_file = st.session_state.get("uploaded_pdf")
        pdf_path = config.BOOKLETS_DIR / f"{selected_date}_{lang_code}.pdf"
        if uploaded_file is not None:
            pdf_path = config.BOOKLETS_DIR / f"custom_{uploaded_file.name}"
            pdf_path.write_bytes(uploaded_file.getbuffer())
            st.success(f"✓ {uploaded_file.name} geladen")

        vote_options = [VOTE_AUTO] + ([] if uploaded_file is not None else DEMO_VOTES.get(pdf_path.name, []))
        if st.session_state.get("vote") not in vote_options:
            st.session_state["vote"] = VOTE_AUTO
        selected_vote = st.selectbox("Vorlage", vote_options, key="vote")
        vote = None if selected_vote == VOTE_AUTO else selected_vote

        st.markdown('<div class="claim-stage-label">Beispiele aus dem OST-Datensatz · Klick prüft sofort</div>', unsafe_allow_html=True)
        primary_cases = [c for c in DEMO_CASES if c["primary"]]
        for row_start in range(0, len(primary_cases), 2):
            row_cols = st.columns(2)
            for col, case in zip(row_cols, primary_cases[row_start:row_start + 2]):
                with col:
                    st.button(case["title"], key=f"story_{case['id']}", width="stretch", on_click=_apply_case, args=(case,))
        with st.expander("Weitere Beispiele", expanded=False):
            for case in [c for c in DEMO_CASES if not c["primary"]]:
                st.button(case["title"], key=f"story_{case['id']}", on_click=_apply_case, args=(case,))

        active_case = next((c for c in DEMO_CASES if c["id"] == st.session_state.get("active_case")), None)
        if active_case:
            pair = f"{active_case['claim_language'].upper()} → {active_case['booklet_language'].upper()}"
            st.markdown(f'<div class="story-desc"><strong>{pair}</strong> · {html.escape(active_case["story"])}</div>', unsafe_allow_html=True)

        st.markdown('<div class="claim-stage-label">Behauptung</div>', unsafe_allow_html=True)
        claim = st.text_area(
            "Behauptung", key="claim_input", height=110,
            placeholder="Politische Behauptung hier eingeben (Deutsch, Französisch oder Italienisch)…",
            label_visibility="collapsed",
        )
        c_claim_lang, c_verify = st.columns([1, 2], vertical_alignment="bottom")
        with c_claim_lang:
            claim_lang_choice = st.selectbox("Sprache der Behauptung", ["Automatisch", "DE", "FR", "IT"], key="claim_lang")
        with c_verify:
            verify_clicked = st.button("Behauptung prüfen", type="primary", width="stretch")
        claim_lang = guess_language(claim) if claim_lang_choice == "Automatisch" else claim_lang_choice.lower()
        st.markdown(
            '<div class="disclaimer">Prüft nur, ob das offizielle Abstimmungsbüchlein die Behauptung stützt, offenlässt '
            'oder ihr widerspricht. Kein politischer Rat und keine Abstimmungsempfehlung.</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Erweitert: Kontext-Strategie & eigenes PDF", expanded=False):
            st.radio("Kontext für Apertus", ["Hybrid-Retrieval (10 Seiten)", "Ganzes Büchlein"], horizontal=True, key="strategy_option")
            st.file_uploader("Eigenes PDF laden", type=["pdf"], key="uploaded_pdf")

    # ── Verification ─────────────────────────────────────────────────────────
    signature = (claim.strip(), str(pdf_path), vote, strat, claim_lang)
    verify_clicked = verify_clicked or st.session_state.pop("auto_verify", False)
    if verify_clicked and claim.strip():
        if not pdf_path.exists():
            with col_out:
                st.error(f"PDF nicht gefunden: `{pdf_path.name}`. Bitte `python -m src download` ausführen.")
        else:
            result, notice = None, None
            cached_case = find_cached_case(claim, pdf_path.name, vote)
            try:
                if engine.client.mock:
                    raise RuntimeError("Kein LLM_API_KEY gesetzt")
                with col_out:
                    with st.spinner("Apertus prüft die Behauptung auf CSCS…"):
                        live = engine.verify_claim(
                            claim=claim, booklet_pdf=pdf_path, claim_language=claim_lang, strategy=strat, vote=vote,
                        )
                if live.error:
                    raise RuntimeError(live.error)
                result = live
            except Exception as exc:
                if cached_case and strat == "hybrid":
                    result = PredictionResult(**cached_case["cached_result"])
                    notice = ("Die Live-Verbindung zu Apertus ist gerade nicht verfügbar. Angezeigt wird das gespeicherte "
                              "Ergebnis dieses Beispiels (vorab live auf CSCS geprüft).")
                else:
                    with col_out:
                        st.error(f"Apertus ist gerade nicht erreichbar ({exc}). Bitte später erneut versuchen.")
            if result is not None:
                st.session_state["last_check"] = {"signature": signature, "result": result, "notice": notice}

    last_check = st.session_state.get("last_check")
    result = last_check["result"] if last_check and last_check["signature"] == signature else None

    # ── RIGHT: result ────────────────────────────────────────────────────────
    with col_out:
        if result is None:
            stale = bool(last_check and claim.strip())
            st.markdown(f"""
            <div class="result-placeholder">
                <div class="claim-stage-label">{'Eingabe geändert' if stale else 'Ergebnis'}</div>
                <p class="verdict-reasoning">{'«Behauptung prüfen» klicken, um das Ergebnis zu aktualisieren.' if stale else
                'Wähle links ein Beispiel oder gib eine Behauptung ein. Apertus vergleicht sie mit dem offiziellen Abstimmungsbüchlein:'}</p>
                <div class="outcome-row"><span class="outcome-dot dot-entailment"></span><strong>Gestützt</strong> — das Büchlein bestätigt die Aussage</div>
                <div class="outcome-row"><span class="outcome-dot dot-neutral"></span><strong>Nicht geklärt</strong> — das Büchlein sagt dazu nichts</div>
                <div class="outcome-row"><span class="outcome-dot dot-contradiction"></span><strong>Widerspruch</strong> — das Büchlein sagt etwas anderes</div>
                <div class="disclaimer">Jedes Ergebnis wird mit den Seiten des Büchleins belegt, auf die es sich stützt.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if last_check.get("notice"):
                st.markdown(f'<div class="notice-box">ℹ️ {last_check["notice"]}</div>', unsafe_allow_html=True)

            b_data = engine._get_booklet_data(pdf_path)
            total_pages = len(b_data["pages"])
            full_tokens = int(len(b_data["full_text"]) / 3.7)
            pages_read = len(result.context_pages) if result.context_pages else total_pages
            v_cls, v_icon, v_title, v_desc = VERDICTS[result.label]

            def fmt(n: int) -> str:
                return f"{n:,}".replace(",", "'")

            st.markdown(f"""
            <div class="verdict-container verdict-{v_cls} animate-in">
                <div class="verdict-label verdict-label-{v_cls}">
                    <span class="verdict-icon icon-{v_cls}">{v_icon}</span>
                    Label {result.label} · {result.label_name}
                </div>
                <h3 class="verdict-title">{v_title}</h3>
                <p class="verdict-reasoning">{v_desc}</p>
                <div class="verdict-stats">
                    <div><span class="vs-num">{claim_lang.upper()} → {lang_code.upper()}</span><span class="vs-lbl">Behauptung → Büchlein</span></div>
                    <div><span class="vs-num">{pages_read} / {total_pages}</span><span class="vs-lbl">Seiten gelesen</span></div>
                    <div><span class="vs-num">{fmt(result.tokens_prompt)}</span><span class="vs-lbl">Tokens statt ≈{fmt(full_tokens)}</span></div>
                    <div><span class="vs-num">{result.latency_ms / 1000:.1f} s</span><span class="vs-lbl">Antwortzeit</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if getattr(result, "numerical_conflict", None):
                st.markdown(f"""
                <div class="guardrail-banner guardrail-vac animate-in">
                    <span class="guardrail-icon">🔢</span>
                    <div class="guardrail-text"><strong>Hinweis Zahlenabgleich:</strong> {html.escape(result.numerical_conflict)}</div>
                </div>
                """, unsafe_allow_html=True)

            pages_by_no = {p["page_number"]: p for p in b_data["pages"]}
            section_by_page = {p["page_number"]: p.get("section") for p in b_data["paragraphs"]}
            primary_quote, primary_page = "", None

            def evidence_html(src, snippet: str) -> str:
                tags = f'<span class="evidence-tag tag-page">Seite {src.page_number}</span>' if src.page_number else ""
                section = SECTION_DE.get(section_by_page.get(src.page_number))
                if section:
                    tags += f'<span class="evidence-tag tag-section">{section}</span>'
                return (f'<div class="evidence-block animate-in"><div class="evidence-header">{tags}</div>'
                        f'<p class="evidence-quote">«{html.escape(snippet)}»</p></div>')

            if result.label != 1 and result.evidence_sources:
                st.markdown('<div class="claim-stage-label">Belegstellen im Abstimmungsbüchlein</div>', unsafe_allow_html=True)
                snippets = []
                for src in result.evidence_sources:
                    page_obj = pages_by_no.get(src.page_number)
                    snippets.append(best_snippet(page_obj["text"] if page_obj else src.quote, claim))
                first = result.evidence_sources[0]
                primary_quote, primary_page = snippets[0], first.page_number
                st.markdown(evidence_html(first, snippets[0]), unsafe_allow_html=True)
                if first.page_number in pages_by_no:
                    with st.expander(f"Ganze Seite {first.page_number} ansehen", expanded=False):
                        highlighted_html = highlight_evidence_in_page(pages_by_no[first.page_number]["text"], snippets[0], result.label)
                        st.markdown(f"""
                        <div class="doc-facsimile">
                            <div class="doc-facsimile-bar">
                                <span>Abstimmungsbüchlein {selected_date} ({lang_code.upper()})</span>
                                <span>Seite {first.page_number}</span>
                            </div>
                            <div style="white-space: pre-wrap;">{highlighted_html}</div>
                        </div>
                        """, unsafe_allow_html=True)
                if len(result.evidence_sources) > 1:
                    with st.expander(f"Weitere Belegstellen ({len(result.evidence_sources) - 1})", expanded=False):
                        for src, snippet in zip(result.evidence_sources[1:], snippets[1:]):
                            st.markdown(evidence_html(src, snippet), unsafe_allow_html=True)
            elif result.label == 1:
                chips = "".join(
                    f'<span class="page-chip">S. {pg}{" · " + SECTION_DE[section_by_page[pg]] if SECTION_DE.get(section_by_page.get(pg)) else ""}</span>'
                    for pg in sorted(result.context_pages)
                )
                st.markdown(f"""
                <div class="evidence-block animate-in">
                    <p class="verdict-reasoning">Keine Stelle im Büchlein bestätigt oder widerlegt diese Behauptung.
                    Apertus hat diese Seiten geprüft:</p>
                    <div class="page-chips">{chips}</div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("Konfidenz & Messwerte", expanded=False):
                st.markdown(f"""
                <div style="margin-bottom: 1.2rem;">
                    <div class="conf-row">
                        <span class="conf-label">Entailment</span>
                        <div class="conf-bar-track"><div class="conf-bar-fill" style="width: {result.p_entail*100:.1f}%; background: #10b981;"></div></div>
                        <span class="conf-value">{result.p_entail*100:.0f}%</span>
                    </div>
                    <div class="conf-row">
                        <span class="conf-label">Neutral</span>
                        <div class="conf-bar-track"><div class="conf-bar-fill" style="width: {result.p_neutral*100:.1f}%; background: #f59e0b;"></div></div>
                        <span class="conf-value">{result.p_neutral*100:.0f}%</span>
                    </div>
                    <div class="conf-row">
                        <span class="conf-label">Contradiction</span>
                        <div class="conf-bar-track"><div class="conf-bar-fill" style="width: {result.p_contra*100:.1f}%; background: #ef4444;"></div></div>
                        <span class="conf-value">{result.p_contra*100:.0f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Konfidenzen schätzt Apertus selbst; sie sind nicht kalibriert.")
                t1, t2, t3 = st.columns(3)
                t1.metric("Inferenzzeit", f"{result.latency_ms:.0f} ms")
                t2.metric("Input-Tokens", fmt(result.tokens_prompt))
                t3.metric("Output-Tokens", fmt(result.tokens_completion))
                st.markdown('<div class="claim-stage-label">Ausgabe im offiziellen OST-Format</div>', unsafe_allow_html=True)
                st.json(result.to_official_dict(), expanded=False)

            with st.expander("Faktencheck-Karte zum Teilen", expanded=False):
                card_quote = primary_quote or "Keine Stelle im Büchlein bestätigt oder widerlegt diese Behauptung."
                card_svg = generate_fact_card_svg(
                    claim=claim, label=result.label, quote=card_quote,
                    page_number=primary_page, proposal_id=None, date_str=selected_date,
                )
                st.markdown(f'<div class="fact-card-wrap">{card_svg}</div>', unsafe_allow_html=True)
                st.download_button(
                    label="SVG herunterladen", data=card_svg,
                    file_name=f"faktencheck_{result.label_name.lower()}.svg", mime="image/svg+xml", width="stretch",
                )


# =============================================================================
# TAB 2: BENCHMARK DASHBOARD
# =============================================================================
with tab_benchmark:
    summary = build_summary()
    head_adv = summary["headline"].get("advanced")
    head_beg = summary["headline"].get("beginner")

    # One row per configuration: keep the largest, then latest run
    latest: Dict[tuple, Dict[str, Any]] = {}
    for r in summary["runs"]:
        if r["split"] != "test":
            continue
        key = (r["task"], r["strategy"], r["top_k"], r.get("passage_chars"), r["prompt_mode"], r.get("ids_reason"))
        if key not in latest or (r["n"], r["timestamp"]) > (latest[key]["n"], latest[key]["timestamp"]):
            latest[key] = r
    test_runs = sorted(latest.values(), key=lambda r: (r["task"], -r["macro_f1"]))

    st.markdown("### Benchmark-Ergebnisse")
    st.markdown(
        "Offizieller OST-Datensatz (1'495 Paare, 60 Büchlein, ~66 % sprachübergreifend). Der **Test-Split** umfasst "
        "5 Abstimmungstermine, die nie zum Entwickeln verwendet wurden. Alle Zahlen stammen aus gespeicherten Läufen in `results/`."
    )

    def _kpi(value: str, label: str, color: str) -> str:
        return f'<div class="kpi-card"><div class="kpi-value" style="color: {color};">{value}</div><div class="kpi-label">{label}</div></div>'

    # Same 150-pair test sample for the context comparison
    def _pick(strategy: str, prompt: str, top_k: Optional[int] = None) -> Optional[Dict[str, Any]]:
        cands = [r for r in summary["runs"] if r["split"] == "test" and r["task"] == "advanced" and r["n"] == 150
                 and r["strategy"] == strategy and r["prompt_mode"] == prompt and (top_k is None or r["top_k"] == top_k)]
        return max(cands, key=lambda r: r["timestamp"]) if cands else None

    comparison = [
        ("Ganzes Büchlein", _pick("full", "json"), False),
        ("Erste Pipeline", _pick("retrieval", "json", 5), False),
        ("Fact Attack (aktuell)", _pick("hybrid", "ids", 10), True),
    ]
    comparison = [(name, run, ours) for name, run, ours in comparison if run]

    kpis = []
    if head_adv:
        kpis.append(_kpi(f"{head_adv['macro_f1']:.3f}", f"Macro-F1 Advanced · n={head_adv['n']}", "#34d399"))
    if head_beg:
        kpis.append(_kpi(f"{head_beg['macro_f1']:.3f}", f"Macro-F1 Beginner · n={head_beg['n']}", "#60a5fa"))
    if head_adv:
        kpis.append(_kpi(f"{head_adv['latency_p95_ms'] / 1000:.1f} s", "p95 Antwortzeit (Advanced)", "#fbbf24"))
    full_run = next((run for name, run, ours in comparison if name == "Ganzes Büchlein"), None)
    ours_run = next((run for name, run, ours in comparison if ours), None)
    if full_run and ours_run:
        kpis.append(_kpi(f"−{(1 - ours_run['avg_prompt_tokens'] / full_run['avg_prompt_tokens']) * 100:.0f}%",
                         "Input-Tokens vs. ganzes Büchlein", "#a78bfa"))
    if kpis:
        st.markdown(f'<div class="kpi-grid">{"".join(kpis)}</div>', unsafe_allow_html=True)
    else:
        st.info("Noch keine Test-Läufe mit den aktuellen Standardeinstellungen in results/.")

    # ── Core finding: selected context beats the full booklet ───────────────
    if len(comparison) >= 2:
        st.markdown("##### Ganzes Büchlein vs. ausgewählter Kontext (Advanced-Task, gleiche 150 Test-Paare)")

        def bar_chart(title: str, field: str, fmt, max_value: float, note: str) -> str:
            rows = []
            for name, run, ours in comparison:
                value = run[field]
                width = max(2.0, value / max_value * 100)
                tip = f"{name}: {fmt(value)} · Strategie {run['strategy']}, Prompt {run['prompt_mode']}, n={run['n']}"
                rows.append(
                    f'<div class="bar-row" title="{html.escape(tip)}">'
                    f'<span class="bar-name{" bar-ours" if ours else ""}">{name}</span>'
                    f'<div class="bar-track"><div class="bar-fill{" bar-fill-ours" if ours else ""}" style="width:{width:.1f}%"></div></div>'
                    f'<span class="bar-value">{fmt(value)}</span></div>'
                )
            return f'<div class="bar-chart"><div class="bar-title">{title}</div>{"".join(rows)}<div class="bar-note">{note}</div></div>'

        c_f1, c_tok = st.columns(2, gap="large")
        with c_f1:
            st.markdown(bar_chart("Macro-F1", "macro_f1", lambda v: f"{v:.3f}", 1.0, "höher ist besser"), unsafe_allow_html=True)
        with c_tok:
            max_tok = max(run["avg_prompt_tokens"] for _, run, _ in comparison)
            st.markdown(bar_chart("Ø Input-Tokens pro Behauptung", "avg_prompt_tokens",
                                  lambda v: f"{v:,.0f}".replace(",", "'"), max_tok, "tiefer ist besser"), unsafe_allow_html=True)
        st.caption("Mit dem ganzen Büchlein (bis ~70'000 Tokens) geht die relevante Stelle verloren: Ausgewählter Kontext ist genauer und rund 10× günstiger.")

    # ── Language-pair matrix ────────────────────────────────────────────────
    if head_adv:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"##### Advanced-Task nach Sprachpaar · Macro-F1 (n={head_adv['n']})")
        langs = ["de", "fr", "it"]
        names = {"de": "DE", "fr": "FR", "it": "IT"}
        cells = ['<div class="mx-corner">Behauptung ↓ · Büchlein →</div>'] + [f'<div class="mx-head">{names[b]}</div>' for b in langs]
        for c in langs:
            cells.append(f'<div class="mx-head">{names[c]}</div>')
            for b in langs:
                v = head_adv["by_language_pair"].get(f"{c}->{b}")
                if not v:
                    cells.append('<div class="mx-cell">–</div>')
                    continue
                alpha = min(0.85, max(0.12, (v["macro_f1"] - 0.7) / 0.3 * 0.85))
                tip = f"{names[c]} → {names[b]}: Macro-F1 {v['macro_f1']:.3f}, Accuracy {v['accuracy']:.3f}, n={v['count']}"
                cells.append(
                    f'<div class="mx-cell" style="background: rgba(57,135,229,{alpha:.2f})" title="{tip}">'
                    f'<span class="mx-val">{v["macro_f1"]:.2f}</span><span class="mx-n">n={v["count"]}</span></div>'
                )
        st.markdown(f'<div class="mx-grid">{"".join(cells)}</div>', unsafe_allow_html=True)
        st.caption("Diagonale = gleiche Sprache. Schwächstes Paar: italienische Behauptung gegen deutsches Büchlein.")

    # ── Table view of all configurations ────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### Alle Konfigurationen (Test-Split)")
    if test_runs:
        st.dataframe(
            [
                {
                    "Task": r["task"],
                    "Kontext": r["strategy"] + (f" k={r['top_k']}" if r["strategy"] in ("hybrid", "retrieval") else ""),
                    "Prompt": r["prompt_mode"],
                    "n": r["n"],
                    "Macro-F1": round(r["macro_f1"], 3),
                    "Cross-lingual F1": round(r["macro_f1_crosslingual"], 3) if r["macro_f1_crosslingual"] is not None else None,
                    "Ø Input-Tokens": int(r["avg_prompt_tokens"]),
                    "Ø Output-Tokens": int(r["avg_completion_tokens"]),
                    "p95 Latenz (s)": round(r["latency_p95_ms"] / 1000, 1),
                }
                for r in test_runs
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption("Latenz clientseitig gemessen; frühe Läufe teilten sich den Endpunkt mit anderen Läufen und sind dadurch langsamer.")

    # Live Evaluation
    with st.expander("Live-Benchmark ausführen", expanded=False):
        datasets = {"OST Test-Split": config.DATA_DIR / "hf" / "test.jsonl", "OST Dev-Split": config.DATA_DIR / "hf" / "dev.jsonl", "Demo (Juni 2026)": config.BENCHMARK_PATH}
        datasets = {k: v for k, v in datasets.items() if v.exists()}
        c_ds, c_task, c_strat, c_lim = st.columns([2, 1, 1, 1])
        with c_ds:
            eval_dataset = st.selectbox("Datensatz", list(datasets))
        with c_task:
            eval_task = st.selectbox("Task", ["advanced", "beginner"])
        with c_strat:
            eval_strat = st.selectbox("Strategie", ["hybrid", "retrieval", "full"])
        with c_lim:
            eval_lim = st.selectbox("Limit", [6, 15, 30, "Alle"], index=0)

        if st.button("Evaluierung starten", key="btn_run_live_eval"):
            lim_val = None if eval_lim == "Alle" else int(eval_lim)
            with st.spinner(f"Evaluiere {eval_lim} Samples…"):
                evaluator = BenchmarkEvaluator()
                live_report = evaluator.evaluate(
                    dataset_path=datasets[eval_dataset], strategy=eval_strat, task=eval_task, limit=lim_val, save=False,
                )

            st.success(f"Abgeschlossen — {live_report['sample_count']} Samples")
            lr1, lr2, lr3, lr4 = st.columns(4)
            lr1.metric("Macro-F1", f"{live_report['macro_f1']:.4f}")
            grounded = live_report["evidence"]["evidence_grounded_rate"]
            lr2.metric("Belege im Gold-Abschnitt", f"{grounded * 100:.0f}%" if grounded is not None else "–")
            lr3.metric("Ø Input-Tokens", f"{live_report['efficiency']['avg_prompt_tokens']:.0f}")
            lr4.metric("p95 Latenz", f"{live_report['efficiency']['latency_p95_ms']:.0f} ms")

            st.dataframe(
                [
                    {
                        "Behauptung": r["claim"][:80] + "...",
                        "Sprachpaar": r["pair"].upper(),
                        "True": r["true_label"],
                        "Pred": r["pred_label"],
                        "✓": "✅" if r["correct"] else "❌",
                        "Tokens": r["tokens_prompt"] + r["tokens_completion"],
                        "Latenz": f"{r['latency_ms']:.0f} ms",
                    }
                    for r in live_report["results"]
                ],
                width="stretch",
                hide_index=True,
            )


# =============================================================================
# TAB 3: ARCHITEKTUR
# =============================================================================
with tab_methodology:
    st.markdown("### So funktioniert Fact Attack")

    st.markdown("""
    <div class="flow">
        <div class="flow-step">
            <div class="flow-num">1</div>
            <div class="flow-title">Büchlein lesen</div>
            <div class="flow-text">PDF in Seiten zerlegen, jede Seite mit ihrem Abschnitt markieren: Komitee, Bundesrat, Abstimmungstext …</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">
            <div class="flow-num">2</div>
            <div class="flow-title">Relevante Seiten finden</div>
            <div class="flow-text">BM25 über alle Seiten: Behauptung + 2 × Titel der Vorlage. Die 10 besten Seiten gehen weiter.</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-step flow-step-apertus">
            <div class="flow-num">3</div>
            <div class="flow-title">Apertus entscheidet</div>
            <div class="flow-text">Seiten nummeriert [P1]…[P10]. Apertus nennt Label und die Nummern der Belegseiten (~55 Tokens).</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">
            <div class="flow-num">4</div>
            <div class="flow-title">Beleg anzeigen</div>
            <div class="flow-text">Label 0 / 1 / 2, wörtliche Belegstellen mit Seitenzahl, Tokens und Antwortzeit.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    col_a1, col_a2 = st.columns(2, gap="large")
    with col_a1:
        st.markdown("##### Warum der Titel der Vorlage?")
        st.markdown(
            "Der Vorlagentitel steht in der Sprache des Büchleins. Er findet die richtige Stelle auch dann, wenn die "
            "Behauptung auf Italienisch und das Büchlein auf Deutsch ist. Ein harter Filter auf eine erkannte Vorlage "
            "verlor dagegen die Übersichtsseiten vorne im Büchlein. Trefferquote des Gold-Abschnitts: 0.74 → 0.93."
        )
        st.markdown("##### Warum Seitennummern statt Zitate?")
        st.markdown(
            "Apertus nennt nur die Nummern der Belegseiten. So sind Belege immer wörtlicher Text aus dem Büchlein mit "
            "korrekter Seitenzahl, und die Antwort ist rund dreimal kürzer als mit Begründung und Zitaten."
        )
    with col_a2:
        st.markdown("##### Warum Abschnitte?")
        st.markdown(
            "Ein Büchlein enthält bewusst beide Seiten. «Das Komitee argumentiert …» wird nur am Text des Komitees "
            "geprüft — Gegenargumente des Bundesrats sind kein Widerspruch."
        )
        st.markdown("##### Was wir verworfen haben")
        st.markdown(
            "- **Ganzes Büchlein senden:** F1 0.73 bei ~60'000 Tokens\n"
            "- **Harter Zahlen-Override:** 5 von 5 Eingriffen falsch\n"
            "- **Nur Label-Ziffer abfragen:** schwach bei «Neutral» (F1 0.72)\n"
            "- **Kleine Textstücke statt Seiten:** präzisere Belege, aber F1 0.83 statt 0.91\n"
            "- **Zweitprüfung bei «Widerspruch»:** 0 von 23 Fehlern behoben, 24 neue\n"
            "- **Apertus 8B statt 70B:** 3,5× schneller, aber F1 0.85 statt 0.91"
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### Reproduzierbarkeit")
    st.code("""# OST-Datensatz, alle Büchlein und Dev/Test-Split laden
python -m src.hf_dataset

# Advanced-Task (Büchlein + Behauptung + Vorlage) auf dem Test-Split
python -m src benchmark -d data/hf/test.jsonl -t advanced

# Beginner-Task (Behauptung + Referenztext)
python -m src benchmark -d data/hf/test.jsonl -t beginner""", language="bash")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Hack Apertus 2026 — Team Fact Attack (Track 2A: OST) · Josip Jukic & Felipe Wüthrich · [GitHub](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026)")
