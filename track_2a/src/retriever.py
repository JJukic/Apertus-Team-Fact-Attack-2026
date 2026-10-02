"""
Passage Retriever for Document-Grounded NLI.
Implements BM25 and token-overlap retrieval over booklet paragraphs to select
the most relevant passages for a given claim.
"""

import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi


class PassageRetriever:
    def __init__(self, paragraphs: List[Dict[str, Any]]):
        self.paragraphs = paragraphs
        self.tokenized_corpus = [self._tokenize(p["text"]) for p in paragraphs]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

        # Dynamically build keyword signatures for each proposal found in this booklet
        self.proposal_keywords: Dict[int, set] = self._extract_proposal_keywords(paragraphs)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Lowercase and split on words across languages
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    @classmethod
    def _extract_proposal_keywords(cls, paragraphs: List[Dict[str, Any]]) -> Dict[int, set]:
        """
        Dynamically extract distinctive signature keywords from the title/intro
        of each proposal in the booklet.
        """
        keywords_by_prop: Dict[int, set] = {}
        title_re = re.compile(
            r"(?:erste|zweite|dritte|vierte|fünfte|sechste|"
            r"premier|deuxi[eè]me|troisi[eè]me|quatri[eè]me|cinqui[eè]me|sixi[eè]me|"
            r"primo|secondo|terzo|quarto|quinto|sesto)\s+"
            r"(?:vorlage|objet|oggetto)\s*:\s*([^\n\r]+)",
            re.IGNORECASE,
        )
        stopwords = {
            "vorlage", "erste", "zweite", "dritte", "vierte", "fünfte", "sechste",
            "premier", "deuxième", "troisième", "quatrième", "cinquième", "sixième",
            "primo", "secondo", "terzo", "quarto", "quinto", "sesto",
            "objet", "oggetto", "detail", "bundesbeschluss", "über", "einen", "blick",
            "bundesrat", "parlament", "empfiehlt", "schweiz", "volk", "stände",
            "pour", "dans", "avec", "delle", "della", "dello", "dalla", "dalle",
            "dass", "wird", "werden", "noch", "damit", "viele", "mehr", "oder", "auch", "aber",
            "nach", "eine", "einer", "eines", "einem", "einen", "nicht", "kann", "können", "soll",
            "sollen", "muss", "müssen", "haben", "hatte", "sein", "waren", "wurde", "wurden",
            "durch", "unter", "zwischen", "sont", "être", "avoir", "plus", "tout", "tous",
            "sono", "essere", "avere", "hanno", "questo", "questa", "anche", "più",
        }

        for p in paragraphs:
            pid = p.get("proposal_id", 0)
            if pid > 0 and pid not in keywords_by_prop:
                txt = p.get("text", "")[:400]
                m = title_re.search(txt)
                title_text = m.group(1) if m else txt[:150]
                quotes = re.findall(r'[«"“]([^»"”]+)[»"”]', txt)
                combined = title_text + " " + " ".join(quotes)
                tokens = set(re.findall(r"[a-zäöüéèà]{4,}", combined.lower()))
                keywords_by_prop[pid] = tokens - stopwords

        return keywords_by_prop

    def _detect_proposal(self, claim: str) -> int:
        """
        Detect which proposal a claim targets using dynamic keyword and subword overlap
        with proposal titles, falling back to heuristic keywords.
        """
        stopwords = {
            "dass", "wird", "werden", "noch", "damit", "viele", "mehr", "oder", "auch", "aber",
            "nach", "eine", "einer", "eines", "einem", "einen", "nicht", "kann", "können", "soll",
            "sollen", "muss", "müssen", "haben", "hatte", "sein", "waren", "wurde", "wurden",
            "bundesrat", "parlament", "empfiehlt", "schweiz", "volk", "stände",
            "conseil", "fédéral", "fédérale", "selon", "texte", "vote", "projet", "mesures", "mesure",
            "consiglio", "federale", "secondo", "testo", "voto", "progetto", "misure", "misura",
        }
        claim_lower = claim.lower()
        claim_tokens = set(re.findall(r"[a-zäöüéèà]{4,}", claim_lower)) - stopwords

        # 1. Dynamic overlap with proposal title keywords (exact or subword for compounds)
        if self.proposal_keywords:
            scores = {}
            for pid, kws in self.proposal_keywords.items():
                matches = sum(1 for ct in claim_tokens if any(ct == kw or (len(ct) >= 5 and ct in kw) for kw in kws))
                scores[pid] = matches

            best_score = max(scores.values()) if scores else 0
            if best_score > 0:
                return max(scores, key=scores.get)

        # 2. Generic BM25 proposal affinity:
        # If title keywords do not trigger a match, calculate max BM25 score per proposal
        if self.bm25:
            tokens = self._tokenize(claim)
            if tokens:
                all_scores = self.bm25.get_scores(tokens)
                prop_max_scores: Dict[int, float] = {}
                for idx, p in enumerate(self.paragraphs):
                    pid = p.get("proposal_id", 0)
                    if pid > 0:
                        s = float(all_scores[idx])
                        if s > prop_max_scores.get(pid, 0.0):
                            prop_max_scores[pid] = s

                if prop_max_scores:
                    best_pid = max(prop_max_scores, key=prop_max_scores.get)
                    if prop_max_scores[best_pid] > 0.0:
                        return best_pid

        # If no proposal clearly dominates, return 0 to search all candidate paragraphs
        return 0

    def retrieve(self, claim: str, top_k: int = 5, target_vote: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most relevant paragraphs for a claim using proposal-aware filtering.
        Optionally accepts target_vote (proposal title) from the official benchmark schema.
        """
        if not self.paragraphs:
            return []

        prop_id = 0
        if target_vote:
            prop_id = self._detect_proposal(target_vote)
        if prop_id == 0:
            prop_id = self._detect_proposal(claim)

        if prop_id > 0 and any(p.get('proposal_id', 0) == prop_id for p in self.paragraphs):
            candidate_paras = [p for p in self.paragraphs if p.get('proposal_id', 0) == prop_id]
        else:
            candidate_paras = self.paragraphs

        if not candidate_paras:
            candidate_paras = self.paragraphs
            
        tokenized_corpus = [self._tokenize(p["text"]) for p in candidate_paras]
        if not tokenized_corpus:
            return []
            
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_claim = self._tokenize(claim)
        
        if not tokenized_claim:
            return candidate_paras[:top_k]

        scores = bm25.get_scores(tokenized_claim)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices[:top_k]:
            para = candidate_paras[idx].copy()
            para["retrieval_score"] = float(scores[idx])
            results.append(para)

        # Check if recommendation anchor page should be prioritized
        claim_lower = claim.lower()
        rec_words = ["empfiehl", "empfehlung", "ablehn", "annahme", "recommande", "recommandation", "rejeter", "accepter", "raccomanda", "raccomandazione", "respingere", "approvare", "bundesrat", "conseil fédéral", "consiglio federale"]
        if any(w in claim_lower for w in rec_words):
            rec_pattern = re.compile(r"(?:empfehl|recommand|raccomand).*(?:bundesrat|conseil f[eé]d[eé]ral|consiglio federale|parlament)", re.IGNORECASE)
            rec_paras = [p for p in candidate_paras if rec_pattern.search(p.get("text", ""))]
            if rec_paras and not any(p.get("page_number") == rec_paras[0].get("page_number") for p in results):
                para = rec_paras[0].copy()
                para["retrieval_score"] = 999.0
                results = [para] + results[:-1]

        return results
