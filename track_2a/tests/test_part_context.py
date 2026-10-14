"""Contents-guided part selection (src/part_context.py): claim openings, contents pages, vote matching."""

import unittest

from src import part_context as pc

TOC_DE = """Erste Vorlage
Volksinitiative «Für eine massvolle Zuwanderung
(Begrenzungsinitiative)»
In Kürze
4 – 5
Im Detail
14
Argumente
20
Abstimmungstext
24
Zweite Vorlage
Änderung des Jagdgesetzes
In Kürze
6 – 7
Im Detail
26
Argumente
34
Abstimmungstext
38
"""
# Navigation bar printed on a summary page: same parts, but no summary entry
NAV_BAR = "Erste Vorlage: Begrenzungsinitiative\nIm Detail\n14\nArgumente\n20\nAbstimmungstext\n24\n"


def booklet():
    pages = {n: "" for n in range(1, 41)}
    pages[2] = TOC_DE
    pages[4] = "Erste Vorlage\nVolksinitiative «Für eine massvolle Zuwanderung (Begrenzungsinitiative)»\nIn Kürze\n" + NAV_BAR
    pages[6] = "Zweite Vorlage\nÄnderung des Jagdgesetzes\nIn Kürze"
    for n, head in ((14, "Im Detail"), (26, "Im Detail"), (20, "Argumente Initiativkomitee"), (34, "Argumente"),
                    (24, "Abstimmungstext"), (38, "Abstimmungstext")):
        pages[n] = head + "\nText"
    return pages


class TestClaimPart(unittest.TestCase):
    def test_openings_in_three_languages(self):
        cases = {
            "Laut der Zusammenfassung sieht die Vorlage vor, …": pc.SUMMARY,
            "Le comité estime que la loi est inutile.": pc.COMMITTEE,
            "Il Consiglio federale raccomanda di respingere l'iniziativa.": pc.FEDERAL_COUNCIL,
            "Selon le texte soumis au vote, la Confédération …": pc.LEGAL,
            "Wird die Abstimmung angenommen, steigen die Abgaben.": pc.DETAIL,
            "Se il voto viene accettato, i Cantoni …": pc.DETAIL,
        }
        for claim, part in cases.items():
            self.assertEqual(pc.claim_part(claim), part, claim)

    def test_earliest_cue_wins_and_unknown_opening(self):
        self.assertEqual(pc.claim_part("Le Conseil fédéral recommande d'accepter la loi."), pc.FEDERAL_COUNCIL)
        self.assertEqual(pc.claim_part("Si le vote est accepté, le Conseil fédéral devra …"), pc.DETAIL)
        self.assertIsNone(pc.claim_part("Die Schweiz hat 26 Kantone."))

    def test_other_wordings_of_the_source(self):
        cases = {
            "Die Befürworter der Vorlage sind der Meinung, dass …": pc.COMMITTEE,
            "Le gouvernement est d'avis que la loi …": pc.FEDERAL_COUNCIL,
            "Stando alla sintesi, l'iniziativa …": pc.SUMMARY,
            "Selon le libellé du projet, la Confédération …": pc.LEGAL,
            "Bei einem Ja müssen die Kantone …": pc.DETAIL,
        }
        for claim, part in cases.items():
            self.assertEqual(pc.claim_part(claim), part, claim)

    def test_actor_in_the_content_is_not_a_source(self):
        # the source is named at the very start; a later mention leaves the claim to the regular retrieval
        self.assertIsNone(pc.claim_part("L'iniziativa sulla neutralità è stata respinta dal Consiglio federale."))
        self.assertIsNone(pc.claim_part("Die Schweiz übernimmt heute die Sanktionen, so der Bundesrat."))


class TestContents(unittest.TestCase):
    def test_parts_and_page_ranges(self):
        parts = pc.booklet_parts(booklet(), {20: pc.COMMITTEE, 21: pc.COMMITTEE, 22: pc.FEDERAL_COUNCIL})
        self.assertEqual(len(parts), 2)  # the navigation bar on page 4 is not a third vote
        first = parts[0]["parts"]
        self.assertEqual(first[pc.SUMMARY], [4, 5])
        self.assertEqual(first[pc.DETAIL], list(range(14, 20)))
        self.assertEqual(first[pc.LEGAL], [24, 25])
        self.assertEqual(first[pc.COMMITTEE], [20, 21])
        self.assertEqual(first[pc.FEDERAL_COUNCIL], [22])

    def test_vote_matching_and_part_pages(self):
        parts = pc.booklet_parts(booklet(), {})
        self.assertEqual(pc.part_pages("Laut der Zusammenfassung …", "Änderung des Jagdgesetzes", parts),
                         (pc.SUMMARY, [6, 7]))
        self.assertIsNone(pc.part_pages("Laut der Zusammenfassung …", "Covid-19-Gesetz", parts))
        self.assertIsNone(pc.part_pages("Die Schweiz hat 26 Kantone.", "Änderung des Jagdgesetzes", parts))

    def test_no_contents_page(self):
        self.assertIsNone(pc.booklet_parts({1: "Titel", 2: "Text ohne Inhaltsverzeichnis"}, {}))


if __name__ == "__main__":
    unittest.main()
