"""Explore saved benchmark cases without issuing model requests."""
import json
from pathlib import Path
from src.inference import attributed_section

LABELS = {0: 'Gestützt', 1: 'Nicht geklärt', 2: 'Widerspruch'}
SPEAKERS = {
    'Arguments of the initiative/referendum committee': 'Komitee',
    'Arguments of the Federal Council and Parliament': 'Bundesrat / Parlament',
}


def speaker(row):
    return SPEAKERS.get(attributed_section(row.get('claim', '')), 'Keine eindeutige Zuordnung')


def filter_cases(rows, pair='Alle', expected='Alle', predicted='Alle', actor='Alle', status='Fehlentscheidungen', query=''):
    """Filter technical failures separately from valid classifications."""
    return [r for r in rows
            if (pair == 'Alle' or r.get('pair') == pair)
            and (expected == 'Alle' or r.get('true_label') == expected)
            and (predicted == 'Alle' or r.get('pred_label') == predicted)
            and (actor == 'Alle' or speaker(r) == actor)
            and (status == 'Alle' or
                 (status == 'Technische Fehler' and bool(r.get('error'))) or
                 (status == 'Fehlentscheidungen' and not r.get('error') and r.get('true_label') != r.get('pred_label')) or
                 (status == 'Korrekte Vorhersagen' and not r.get('error') and r.get('true_label') == r.get('pred_label')))
            and (not query or query.casefold() in (r.get('claim', '') + ' ' + r.get('vote', '')).casefold())]


def render_error_analysis(runs, results_dir: Path, preferred_file=None):
    import streamlit as st
    st.markdown('##### Fehleranalyse')
    st.caption('Gespeicherte Modellantworten untersuchen. Sprecher werden anhand der Behauptung automatisch zugeordnet; es erfolgen keine API-Aufrufe.')
    if not runs:
        st.info('Keine gespeicherten Benchmark-Läufe vorhanden.')
        return
    ordered = sorted(runs, key=lambda r: r['timestamp'], reverse=True)
    files = [r['file'] for r in ordered]
    by_file = {r['file']: r for r in ordered}
    filename = st.selectbox('Benchmark-Lauf', files,
        index=files.index(preferred_file) if preferred_file in files else 0,
        format_func=lambda f: f"{by_file[f]['task']} · {by_file[f]['split']} · n={by_file[f]['n']} · {f}", key='analysis_run')
    try:
        report = json.loads((results_dir / filename).read_text(encoding='utf-8'))
        rows = report['results']
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Benchmark konnte nicht geladen werden: {exc}')
        return
    cols = st.columns(3)
    failures = sum(bool(r.get('error')) for r in rows)
    mistakes = sum(not r.get('error') and r['true_label'] != r['pred_label'] for r in rows)
    cols[0].metric('Fälle im Lauf', len(rows))
    cols[1].metric('Fehlentscheidungen', mistakes)
    cols[2].metric('Technische Fehler', failures)
    c1, c2, c3 = st.columns(3)
    pair = c1.selectbox('Sprachpaar', ['Alle'] + sorted({r.get('pair', '') for r in rows}), key='analysis_pair')
    expected = c2.selectbox('Erwartete Klasse', ['Alle', 0, 1, 2], format_func=lambda x: LABELS.get(x, x), key='analysis_expected')
    predicted = c3.selectbox('Vorhergesagte Klasse', ['Alle', 0, 1, 2], format_func=lambda x: LABELS.get(x, x), key='analysis_predicted')
    c1, c2 = st.columns(2)
    actor = c1.selectbox('Sprecher', ['Alle'] + sorted(set(SPEAKERS.values())) + ['Keine eindeutige Zuordnung'], key='analysis_actor')
    status = c2.selectbox('Falltyp', ['Fehlentscheidungen', 'Technische Fehler', 'Korrekte Vorhersagen', 'Alle'], key='analysis_status')
    query = st.text_input('Behauptung oder Vorlage durchsuchen', key='analysis_query')
    filtered = filter_cases(rows, pair, expected, predicted, actor, status, query)
    st.caption(f'{len(filtered)} von {len(rows)} Fällen entsprechen den Filtern.')
    if not filtered:
        st.info('Keine Fälle für diese Filter. Filter erweitern, um andere Fälle zu sehen.')
        return
    st.dataframe([{'ID': r.get('id'), 'Behauptung': r.get('claim'), 'Sprachpaar': r.get('pair'),
                   'Erwartet': LABELS.get(r.get('true_label')), 'Vorhersage': LABELS.get(r.get('pred_label')),
                   'Sprecher': speaker(r), 'Belegseiten': ', '.join(str(p) for p in r.get('evidence_pages', []) if p is not None)}
                  for r in filtered], hide_index=True, width='stretch')
    st.download_button('Gefilterte Fälle als JSON herunterladen', json.dumps(filtered, ensure_ascii=False, indent=2),
                       file_name=f'fehleranalyse_{Path(filename).stem}.json', mime='application/json', key='analysis_export')
    index = st.selectbox('Fall im Detail', list(range(len(filtered))),
                         format_func=lambda i: f"{filtered[i].get('id', i)} · {filtered[i].get('claim', '')[:90]}", key='analysis_case')
    row = filtered[index]
    st.markdown('**Behauptung**')
    st.write(row.get('claim', ''))
    st.write('Vorlage:', row.get('vote', ''))
    left, right = st.columns(2)
    left.metric('Erwartetes Ergebnis', LABELS.get(row.get('true_label'), 'Unbekannt'))
    right.metric('Modellvorhersage', LABELS.get(row.get('pred_label'), 'Unbekannt'))
    if row.get('error'):
        st.error(row['error'])
    for warning in row.get('stage_warnings', []):
        st.warning(warning)
    st.markdown('**Gespeicherte Modellantwort**')
    st.write(row.get('reasoning') or 'Keine Begründung gespeichert.')
    if row.get('decision_rule'):
        st.caption('Entscheidungsregel: ' + row['decision_rule'])
    if row.get('extracted_statements'):
        with st.expander('Stufe 1: Extrahierte Aussagen'):
            st.json(row['extracted_statements'])
    st.markdown('**Vom Modell angeführte Belege**')
    evidence = row.get('evidence', [])
    pages = row.get('evidence_pages', [])
    for i, quote in enumerate(evidence):
        page = pages[i] if i < len(pages) else None
        with st.expander(f'Beleg {i + 1} · Seite {page}' if page is not None else f'Beleg {i + 1} · ohne Seitenangabe'):
            st.write(quote)
    if not evidence:
        st.caption('Keine Belege gespeichert.')
    st.caption('Die historischen Berichte enthalten keine vollständige Gold-Referenz und keinen PDF-Pfad. Angezeigt werden die gespeicherten Zitate und Seitenangaben; sie werden hier nicht erneut verifiziert.')
