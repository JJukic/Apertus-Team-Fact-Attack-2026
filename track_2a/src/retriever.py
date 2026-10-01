"""
Passage Retriever for Document-Grounded NLI.
Implements BM25 and token-overlap retrieval over booklet paragraphs to select
the most relevant passages for a given claim.
"""

import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class PassageRetriever:
    def __init__(self, paragraphs: List[Dict[str, Any]]):
        self.paragraphs = paragraphs
        self.tokenized_corpus = [self._tokenize(p["text"]) for p in paragraphs]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Lowercase and split on words across languages
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def _detect_proposal(claim: str) -> int:
        claim_lower = claim.lower()
        v2_words = ["zivildienst", "militär", "ersatzdienst", "zdg", "armee", "service civil", "armée", "servizio civile", "militare", "esercito"]
        v1_words = ["initiative", "nachhaltig", "10-millionen", "10 million", "zuwanderung", "wohnbevölkerung", "personenfreizügigkeit", "schengen", "dublin", "durabilité", "immigration", "sostenibilità", "popolazione", "libera circolazione"]
        
        has_v2 = any(w in claim_lower for w in v2_words)
        has_v1 = any(w in claim_lower for w in v1_words)
        
        if has_v2 and not has_v1:
            return 2
        if has_v1 and not has_v2:
            return 1
        # Default to Vorlage 1 if general initiative
        return 1 if ("initiative" in claim_lower or "iniziativa" in claim_lower) else 0

    def retrieve(self, claim: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most relevant paragraphs for a claim using proposal-aware filtering.
        """
        if not self.paragraphs:
            return []

        prop_id = self._detect_proposal(claim)
        if prop_id in (1, 2):
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
            anchor_page = 5 if prop_id == 1 else (7 if prop_id == 2 else 5)
            # Check if anchor page is already present
            if not any(p['page_number'] == anchor_page for p in results):
                anchor_paras = [p for p in candidate_paras if p['page_number'] == anchor_page]
                if anchor_paras:
                    # Insert at the top of results
                    para = anchor_paras[0].copy()
                    para["retrieval_score"] = 999.0
                    results = [para] + results[:-1]

        return results
