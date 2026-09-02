"""
A3 - BM25 sparse/lexical index.

Dense embeddings alone will often miss exact-identifier queries ("section 318", "BNSS 103")
because a section number isn't semantically distinctive to an embedding model the way it is
to a keyword matcher. BM25 catches these by exact token overlap. Combined with dense search
via RRF, this is the "hybrid retrieval" requirement.
"""
import re
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        # Index over section_number + text, so a bare "103" or "s.103" token hits directly.
        corpus = [
            _tokenize(f"{c.get('section_number', '')} {c.get('section_title', '') or ''} {c['text']}")
            for c in chunks
        ]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"id": i, "score": float(scores[i]), "payload": self.chunks[i]} for i in ranked if scores[i] > 0]
