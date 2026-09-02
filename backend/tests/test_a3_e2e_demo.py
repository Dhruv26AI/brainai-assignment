"""
A3 end-to-end smoke test against real ingested chunks.

NOTE: this sandbox has no network access to huggingface.co, so this test uses TF-IDF
vectors (scikit-learn, no network needed) as a stand-in "dense" embedding purely to prove
the retrieval WIRING is correct: Qdrant upsert/search, BM25 fusion via RRF, direct-lookup,
metadata filtering, and the reranker interface. It does NOT prove embedding quality --
that was already validated in your real A2 run (BGE-base-en-v1.5, 768-dim, on your machine,
which does have internet access). Swap embed_fn back to the real SentenceTransformer call
(as shown commented out below) when running this for real.
"""
import json
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, "/home/claude/nyaya")

from backend.app.retrieval.qdrant_store import VectorStore
from backend.app.retrieval.bm25_index import BM25Index
from backend.app.retrieval.hybrid_search import HybridRetriever

print("Loading chunks...")
chunks = [json.loads(l) for l in open("data/processed/bnss_chunks.jsonl")]
print(f"{len(chunks)} chunks loaded")

print("Building TF-IDF + SVD stand-in dense vectors (network-free substitute for BGE)...")
texts = [c["text"] for c in chunks]
tfidf = TfidfVectorizer(max_features=20000)
tfidf_matrix = tfidf.fit_transform(texts)
svd = TruncatedSVD(n_components=128, random_state=42)
dense_matrix = svd.fit_transform(tfidf_matrix)
norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
norms[norms == 0] = 1
dense_matrix = dense_matrix / norms
embeddings = dense_matrix.tolist()
VECTOR_SIZE = 128

print("Building Qdrant in-memory store + BM25 index...")
store = VectorStore(url=":memory:", vector_size=VECTOR_SIZE)
store.ensure_collection()
store.upsert_chunks(chunks, embeddings)
bm25 = BM25Index(chunks)


def embed_query(q: str) -> list[float]:
    # Stand-in: project the query through the same fitted TF-IDF+SVD space.
    # Real production version (used with your actual A2 output):
    #   model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    #   return model.encode("query: " + q, normalize_embeddings=True).tolist()
    vec = svd.transform(tfidf.transform([q]))[0]
    n = np.linalg.norm(vec)
    return (vec / n if n else vec).tolist()


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    # Stand-in cross-encoder: lexical overlap scoring (network-free).
    # Real production version:
    #   cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    #   scores = cross_encoder.predict([(query, c["payload"]["text"]) for c in candidates])
    q_tokens = set(query.lower().split())
    for c in candidates:
        t_tokens = set(c["payload"]["text"].lower().split())
        c["rerank_score"] = len(q_tokens & t_tokens) / (len(q_tokens) + 1)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


retriever = HybridRetriever(store, bm25, embed_query, reranker=rerank)

print("\n" + "=" * 70)
print("TEST 1: direct-lookup path (exact section number query)")
print("=" * 70)
res = retriever.search("what is section 41 BNSS", top_k=3)
print("retrieval_path:", res["retrieval_path"])
for r in res["results"]:
    p = r["payload"]
    print(f"  -> Section {p['section_number']} ({p.get('section_title')}) subsection={p.get('subsection')}")

print("\n" + "=" * 70)
print("TEST 2: hybrid semantic query (no explicit section number)")
print("=" * 70)
res = retriever.search("can police arrest someone without a warrant", top_k=5)
print("retrieval_path:", res["retrieval_path"])
for r in res["results"]:
    p = r["payload"]
    print(f"  -> Section {p['section_number']} ({p.get('section_title')})  rrf_score={r['score']:.4f}  rerank_score={r.get('rerank_score', 'n/a')}")

print("\n" + "=" * 70)
print("TEST 3: metadata-filtered search (restrict to a chapter)")
print("=" * 70)
sample_chapter = chunks[100]["chapter"]
print(f"filtering to chapter={sample_chapter}")
res = retriever.search("bail conditions", top_k=3, filters={"chapter": sample_chapter})
print("retrieval_path:", res["retrieval_path"])
for r in res["results"]:
    p = r["payload"]
    print(f"  -> chapter={p['chapter']} Section {p['section_number']} ({p.get('section_title')})")

print("\n" + "=" * 70)
print("TEST 4: out-of-scope query -- should still return something via BM25/dense,")
print("        but with low relevance (citation-validation guard in A4 will refuse this)")
print("=" * 70)
res = retriever.search("what is the punishment for jaywalking in Ohio", top_k=3)
print("retrieval_path:", res["retrieval_path"])
for r in res["results"]:
    p = r["payload"]
    print(f"  -> Section {p['section_number']} score={r['score']:.4f}")
