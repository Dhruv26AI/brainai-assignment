"""
A3 - Hybrid retrieval orchestrator.

Query flow:
  1. Direct-lookup check: does the query name an explicit section number? If so, bypass
     everything else and return that section deterministically.
  2. Otherwise: run dense search (Qdrant) and BM25 search (lexical) in parallel, fuse
     their ranked lists with Reciprocal Rank Fusion, then optionally rerank the fused
     top-k with a cross-encoder for final precision.
"""
from .section_intent import detect_section_intent
from .qdrant_store import VectorStore
from .bm25_index import BM25Index

RRF_K = 60  # standard RRF constant
FILTERABLE_FIELDS = ("chapter", "act_short", "section_number")


def _matches_filter(payload: dict, filters: dict) -> bool:
    return all(payload.get(k) == v for k, v in filters.items() if k in FILTERABLE_FIELDS and v)


def reciprocal_rank_fusion(*ranked_lists: list[dict], k: int = RRF_K) -> list[dict]:
    """
    Each ranked_list is a list of {"id", "score", "payload"} already sorted best-first.
    Fuses them by rank position (not raw score, which isn't comparable across dense
    cosine similarity and BM25's unbounded scores).
    """
    fused_scores: dict[int, float] = {}
    payload_by_id: dict[int, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            fused_scores[item["id"]] = fused_scores.get(item["id"], 0.0) + 1.0 / (k + rank + 1)
            payload_by_id[item["id"]] = item["payload"]

    ranked_ids = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)
    return [{"id": i, "score": fused_scores[i], "payload": payload_by_id[i]} for i in ranked_ids]


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index, embed_fn, reranker=None):
        """
        embed_fn: callable(str) -> list[float], must apply the "query: " prefix internally
                  (see A2's query/passage prefix requirement).
        reranker: optional callable(query, list[chunk_dict]) -> reordered list[chunk_dict]
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embed_fn = embed_fn
        self.reranker = reranker

    def search(self, query: str, top_k: int = 5, filters: dict | None = None) -> dict:
        intent = detect_section_intent(query)

        if intent:
            hits = self.vector_store.exact_section_lookup(
                intent["section_number"], intent["subsection"]
            )
            return {
                "results": hits[:top_k],
                "retrieval_path": "direct_lookup",
                "detected_section": intent["section_number"],
            }

        query_vector = self.embed_fn(query)
        dense_hits = self.vector_store.dense_search(query_vector, top_k=20, filters=filters)
        sparse_hits = self.bm25_index.search(query, top_k=20)
        if filters:
            # Qdrant applies the filter server-side for dense_hits already; rank_bm25 has
            # no native filtering, so apply the same metadata filter manually here --
            # otherwise unfiltered BM25 hits would leak back in through RRF fusion.
            sparse_hits = [h for h in sparse_hits if _matches_filter(h["payload"], filters)]

        fused = reciprocal_rank_fusion(dense_hits, sparse_hits)[:20]

        if self.reranker:
            fused = self.reranker(query, fused)

        return {
            "results": fused[:top_k],
            "retrieval_path": "hybrid_rrf" + ("+rerank" if self.reranker else ""),
            "detected_section": None,
        }
