"""
Interactive query script -- wires together the existing A3 modules
(qdrant_store.py, bm25_index.py, hybrid_search.py, section_intent.py) against your REAL,
already-populated Qdrant collection (673 points, loaded by index_to_qdrant.py) and your
REAL BGE embeddings (from A2).

Run from the project root, with Qdrant already running (docker run -p 6333:6333 qdrant/qdrant):

    python backend/app/retrieval/query_bnss.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from sentence_transformers import SentenceTransformer, CrossEncoder
from backend.app.retrieval.qdrant_store import VectorStore
from backend.app.retrieval.bm25_index import BM25Index
from backend.app.retrieval.hybrid_search import HybridRetriever

CHUNKS_FILE = "data/processed/bnss_chunks.jsonl"
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768

print("Loading BM25 index (needs the chunk texts -- built fresh, fast, no network)...")
chunks = [json.loads(l) for l in open(CHUNKS_FILE, encoding="utf-8")]
bm25 = BM25Index(chunks)

print(f"Connecting to live Qdrant collection at {QDRANT_URL} ...")
store = VectorStore(url=QDRANT_URL, vector_size=VECTOR_SIZE)
# NOTE: does NOT call store.ensure_collection() here -- the collection was already created
# and populated by index_to_qdrant.py. Calling ensure_collection() again is harmless (it
# checks existence first) but we don't want to re-upsert here.

print(f"Loading embedding model ({EMBED_MODEL}) -- this may take a minute the first time...")
model = SentenceTransformer(EMBED_MODEL)


def embed_query(q: str) -> list[float]:
    # BGE requires the "query: " prefix for search queries (different from the "passage: "
    # prefix used when embedding the corpus in A2) -- getting this wrong silently halves recall.
    return model.encode("query: " + q, normalize_embeddings=True).tolist()


print("Loading cross-encoder reranker...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    pairs = [(query, c["payload"]["text"]) for c in candidates]
    scores = cross_encoder.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


retriever = HybridRetriever(store, bm25, embed_query, reranker=rerank)

print("\nReady. Type a question about BNSS (or 'quit' to exit).\n")

while True:
    query = input("Query> ").strip()
    if not query or query.lower() in ("quit", "exit"):
        break

    result = retriever.search(query, top_k=5)
    print(f"\n[retrieval_path: {result['retrieval_path']}]")
    for r in result["results"]:
        p = r["payload"]
        print(
            f"  Section {p['section_number']}"
            + (f"{p['subsection']}" if p.get("subsection") else "")
            + f" - {p.get('section_title')}"
            f"  (score={r['score']:.4f}"
            + (f", rerank={r.get('rerank_score'):.4f}" if "rerank_score" in r else "")
            + f", page {p['page_start']}-{p['page_end']})"
        )
        print(f"    {p['text'][:180].strip()}...")
    print()
