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

    def retrieve(self, claim: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most relevant paragraphs for a claim.
        """
        if not self.bm25 or not self.paragraphs:
            return []

        tokenized_claim = self._tokenize(claim)
        if not tokenized_claim:
            return self.paragraphs[:top_k]

        scores = self.bm25.get_scores(tokenized_claim)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices[:top_k]:
            para = self.paragraphs[idx].copy()
            para["retrieval_score"] = float(scores[idx])
            results.append(para)

        return results
