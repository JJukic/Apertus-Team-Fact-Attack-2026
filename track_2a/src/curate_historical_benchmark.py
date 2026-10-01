"""
Generates a gold-standard, human-curated style benchmark for the 2024-11-24 Swiss Federal Voting Booklet
across German, French, and Italian.
Follows the exact perturbation methodology of the OST challenge:
- Label 0 (Entailment): True factual assertions grounded in the booklet text
- Label 1 (Neutral): Plausible Swiss political statements on topics not in the booklet
- Label 2 (Contradiction): Direct numerical or polarity inversions of factual assertions
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "benchmark_2024-11-24.jsonl"

BENCHMARK_SAMPLES = [
    # =========================================================================
    # GERMAN SAMPLES (Booklet 2024-11-24, Vorlagen 1 - 4)
    # =========================================================================
    # Vorlage 1: Nationalstrassen
    {
        "claim": "Der Ausbauschritt 2023 für die Nationalstrassen umfasst Massnahmen auf sechs Autobahnabschnitten.",
        "claim_language": "de",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "de",
        "proposal_id": 1,
    },
    {
        "claim": "Bundesrat und Parlament wollen für den Ausbauschritt 2023 der Nationalstrassen rund 5,3 Milliarden Franken investieren.",
        "claim_language": "de",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "de",
        "proposal_id": 1,
    },
    {
        "claim": "Bundesrat und Parlament wollen für den Ausbauschritt 2023 der Nationalstrassen weniger als 1 Milliarde Franken investieren.",
        "claim_language": "de",
        "entailment_label": 2,
        "booklet_date": "2024-11-24",
        "booklet_language": "de",
        "proposal_id": 1,
    },
    {
        "claim": "Der Bundesrat argumentiert, dass die Einführung einer allgemeinen Flugticketabgabe die Klimaziele bis 2030 beschleunigen soll.",
        "claim_language": "de",
        "entailment_label": 1,
        "booklet_date": "2024-11-24",
        "booklet_language": "de",
        "proposal_id": 1,
    },

    # Vorlage 2: Mietrecht: Untermiete
    {
        "claim": "Künftig müssen Mieterinnen und Mieter die schriftliche Zustimmung der Vermieterin oder des Vermieters einholen, wenn sie Räume untervermieten wollen.",
        "claim_language": "de",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "de",
        "proposal_id": 2,
    },
    {
        "claim": "Nach der Vorlage zur Untermiete dürfen Mieterinnen und Mieter ihre Wohnung ohne jegliche Information oder Zustimmung der Eigentümerschaft dauerhaft untervermieten.",
        "claim_language": "de",
        "entailment_label": 2,
        "booklet_date": "2024-11-24",
        "booklet_language": "de",
        "proposal_id": 2,
    },
    {
        "claim": "Das Parlament empfiehlt Volk und Ständen, die Vorlage zur Untermiete anzunehmen.",
        "claim_language": "de",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "de",
        "proposal_id": 2,
    },

    # Vorlage 3: Mietrecht: Kündigung wegen Eigenbedarfs
    {
        "claim": "Bei der Vorlage zum Eigenbedarf soll eine Kündigung möglich sein, wenn die Eigentümerin einen bedeutenden und aktuellen Eigenbedarf geltend macht.",
        "claim_language": "de",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "de",
        "proposal_id": 3,
    },
    {
        "claim": "Bundesrat und Parlament empfehlen Volk und Ständen, die Vorlage zur Kündigung wegen Eigenbedarfs abzulehnen.",
        "claim_language": "de",
        "entailment_label": 2,
        "booklet_date": "2024-11-24",
        "booklet_language": "de",
        "proposal_id": 3,
    },
    {
        "claim": "Der Bundesrat schlägt vor, die Mehrwertsteuer auf Grundnahrungsmittel per Verordnung um 0,5 Prozentpunkte zu senken.",
        "claim_language": "de",
        "entailment_label": 1,
        "booklet_date": "2024-11-24",
        "booklet_language": "de",
        "proposal_id": 3,
    },

    # Vorlage 4: EFAS (Finanzierung Gesundheitsleistungen)
    {
        "claim": "Die Vorlage zur einheitlichen Finanzierung (EFAS) sieht vor, dass ambulante und stationäre Leistungen nach dem gleichen Schlüssel von Kantonen und Krankenkassen finanziert werden.",
        "claim_language": "de",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "de",
        "proposal_id": 4,
    },
    {
        "claim": "Gemäss der EFAS-Vorlage müssen die Kantone künftig sämtliche Kosten für stationäre Spitalbehandlungen zu 100 Prozent alleine tragen.",
        "claim_language": "de",
        "entailment_label": 2,
        "booklet_date": "2024-11-24",
        "booklet_language": "de",
        "proposal_id": 4,
    },

    # =========================================================================
    # FRENCH SAMPLES (Booklet 2024-11-24, Objets 1 - 4)
    # =========================================================================
    {
        "claim": "L'étape d'aménagement 2023 des routes nationales prévoit des projets sur six tronçons autoroutiers.",
        "claim_language": "fr",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "fr",
        "proposal_id": 1,
    },
    {
        "claim": "Le Conseil fédéral et le Parlement prévoient un crédit d'environ 5,3 milliards de francs pour l'aménagement des routes nationales.",
        "claim_language": "fr",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "fr",
        "proposal_id": 1,
    },
    {
        "claim": "Le Conseil fédéral et le Parlement prévoient moins de 500 millions de francs pour les projets de routes nationales.",
        "claim_language": "fr",
        "entailment_label": 2,
        "booklet_date": "2024-11-24",
        "booklet_language": "fr",
        "proposal_id": 1,
    },
    {
        "claim": "Le Conseil fédéral soutient la création d'un fonds spécial de solidarité énergétique financé par une taxe carbone.",
        "claim_language": "fr",
        "entailment_label": 1,
        "booklet_date": "2024-11-24",
        "booklet_language": "fr",
        "proposal_id": 1,
    },
    {
        "claim": "Désormais, le locataire devra obtenir le consentement écrit du bailleur pour toute sous-location.",
        "claim_language": "fr",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "fr",
        "proposal_id": 2,
    },
    {
        "claim": "Le projet sur le droit du bail autorise le locataire à sous-louer sans aucune autorisation préalable du propriétaire.",
        "claim_language": "fr",
        "entailment_label": 2,
        "booklet_date": "2024-11-24",
        "booklet_language": "fr",
        "proposal_id": 2,
    },
    {
        "claim": "Le Conseil fédéral recommande d'accepter la modification concernant la résiliation pour besoin propre.",
        "claim_language": "fr",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "fr",
        "proposal_id": 3,
    },
    {
        "claim": "Le Conseil fédéral recommande de rejeter le projet de financement uniforme des prestations de santé (EFAS).",
        "claim_language": "fr",
        "entailment_label": 2,
        "booklet_date": "2024-11-24",
        "booklet_language": "fr",
        "proposal_id": 4,
    },
    {
        "claim": "Le projet EFAS vise à financer les prestations ambulatoires et stationnaires selon la même clé de répartition.",
        "claim_language": "fr",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "fr",
        "proposal_id": 4,
    },
    {
        "claim": "Le Conseil fédéral préconise une réforme du système de bourses d'études universitaires d'ici 2028.",
        "claim_language": "fr",
        "entailment_label": 1,
        "booklet_date": "2024-11-24",
        "booklet_language": "fr",
        "proposal_id": 4,
    },

    # =========================================================================
    # ITALIAN SAMPLES (Booklet 2024-11-24, Oggetti 1 - 4)
    # =========================================================================
    {
        "claim": "La fase di potenziamento 2023 delle strade nazionali riguarda interventi su sei tratti autostradali.",
        "claim_language": "it",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "it",
        "proposal_id": 1,
    },
    {
        "claim": "Il Consiglio federale e il Parlamento prevedono circa 5,3 miliardi di franchi per il potenziamento delle strade nazionali.",
        "claim_language": "it",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "it",
        "proposal_id": 1,
    },
    {
        "claim": "Il Consiglio federale e il Parlamento prevedono di stanziare meno di 1 miliardo di franchi per le strade nazionali.",
        "claim_language": "it",
        "entailment_label": 2,
        "booklet_date": "2024-11-24",
        "booklet_language": "it",
        "proposal_id": 1,
    },
    {
        "claim": "Il Consiglio federale sostiene un programma straordinario di incentivi per l'acquisto di veicoli a idrogeno.",
        "claim_language": "it",
        "entailment_label": 1,
        "booklet_date": "2024-11-24",
        "booklet_language": "it",
        "proposal_id": 1,
    },
    {
        "claim": "In futuro l'inquilino dovrà ottenere il consenso scritto del locatore prima di procedere alla sublocazione.",
        "claim_language": "it",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "it",
        "proposal_id": 2,
    },
    {
        "claim": "La nuova legge sulla sublocazione stabilisce che non è richiesta alcuna autorizzazione del proprietario per subaffittare l'alloggio.",
        "claim_language": "it",
        "entailment_label": 2,
        "booklet_date": "2024-11-24",
        "booklet_language": "it",
        "proposal_id": 2,
    },
    {
        "claim": "Il Consiglio federale raccomanda di respingere con un NO la modifica relativa alla sublocazione.",
        "claim_language": "it",
        "entailment_label": 2,
        "booklet_date": "2024-11-24",
        "booklet_language": "it",
        "proposal_id": 2,
    },
    {
        "claim": "La modifica relativa al bisogno personale consente la disdetta qualora il proprietario faccia valere un bisogno importante e attuale.",
        "claim_language": "it",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "it",
        "proposal_id": 3,
    },
    {
        "claim": "Il progetto EFAS prevede che le prestazioni ambulatoriali e stazionarie siano finanziate secondo la stessa chiave di ripartizione.",
        "claim_language": "it",
        "entailment_label": 0,
        "booklet_date": "2024-11-24",
        "booklet_language": "it",
        "proposal_id": 4,
    },
    {
        "claim": "Il Consiglio federale propone di riformare il sistema di imposta sulle donazioni a partire dal 2028.",
        "claim_language": "it",
        "entailment_label": 1,
        "booklet_date": "2024-11-24",
        "booklet_language": "it",
        "proposal_id": 4,
    },
]


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in BENCHMARK_SAMPLES:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {len(BENCHMARK_SAMPLES)} gold-standard benchmark samples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
