
"""
Part F - Test 1
Out-of-scope refusal through the actual answer pipeline.

Tests the 5 must_refuse questions from golden_set.jsonl by running:

golden set -> HybridRetriever -> answer_question()

This is NOT a retrieval-only test.
"""

import json
import os
import sys
import time

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from sentence_transformers import SentenceTransformer, CrossEncoder

from backend.app.retrieval.qdrant_store import VectorStore
from backend.app.retrieval.bm25_index import BM25Index
from backend.app.retrieval.hybrid_search import HybridRetriever

from backend.app.llm.providers import get_llm_provider
from backend.app.llm.answer import answer_question


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

GOLDEN_FILE = "eval/golden_set.jsonl"

CHUNKS_FILE = "data/processed/bnss_chunks.jsonl"

QDRANT_URL = "http://localhost:6333"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

VECTOR_SIZE = 768


# ---------------------------------------------------------
# Load golden set
# ---------------------------------------------------------

def load_golden_set(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


# ---------------------------------------------------------
# Load retrieval system
# ---------------------------------------------------------

print("=" * 70)
print("PART F - TEST 1: ANSWER-PIPELINE REFUSAL")
print("=" * 70)

print("\nLoading BNSS chunks...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

print(f"Loaded {len(chunks)} BNSS chunks")


print(f"\nLoading embedding model: {EMBED_MODEL}")

model = SentenceTransformer(EMBED_MODEL)

print("Embedding model loaded.")


def embed_query(query: str) -> list[float]:
    return model.encode(
        "query: " + query,
        normalize_embeddings=True,
    ).tolist()


print("\nLoading BM25 index...")

bm25 = BM25Index(chunks)

print("BM25 index loaded.")


print(f"\nConnecting to Qdrant at {QDRANT_URL}...")

store = VectorStore(
    url=QDRANT_URL,
    vector_size=VECTOR_SIZE,
)

print("Qdrant connection ready.")


print("\nLoading cross-encoder reranker...")

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Cross-encoder loaded.")


def rerank(query: str, candidates: list[dict]) -> list[dict]:

    if not candidates:
        return []

    pairs = [
        (
            query,
            candidate["payload"].get("text", "")
        )
        for candidate in candidates
    ]

    scores = cross_encoder.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    return sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )


retriever = HybridRetriever(
    store,
    bm25,
    embed_query,
    reranker=rerank,
)


# ---------------------------------------------------------
# Load LLM
# ---------------------------------------------------------

print("\nLoading LLM provider...")

llm = get_llm_provider()

print("LLM provider ready.")


# ---------------------------------------------------------
# Load must-refuse questions
# ---------------------------------------------------------

rows = load_golden_set(GOLDEN_FILE)

refusal_questions = [
    row
    for row in rows
    if row.get("type") == "must_refuse"
]

print(
    f"\nLoaded {len(rows)} golden-set questions."
)

print(
    f"Found {len(refusal_questions)} must-refuse questions."
)


# ---------------------------------------------------------
# Run actual answer pipeline
# ---------------------------------------------------------

correct = 0
latencies = []

print("\n" + "=" * 70)
print("RUNNING ANSWER-PIPELINE REFUSAL TEST")
print("=" * 70)


for index, row in enumerate(
    refusal_questions,
    start=1,
):

    question = row["q"]

    print("\n" + "-" * 70)
    print(
        f"Question {index}/{len(refusal_questions)}"
    )
    print(f"Q: {question}")

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    start = time.perf_counter()

    retrieval_result = retriever.search(
        question,
        top_k=10,
    )

    retrieval_time = time.perf_counter() - start

    results = retrieval_result.get(
        "results",
        [],
    )

    top_score = None

    if results:
        top_score = results[0].get(
            "rerank_score",
            results[0].get("score"),
        )

    print(
        f"Retrieved results: {len(results)}"
    )

    print(
        f"Top score: {top_score}"
    )

    print(
        f"Retrieval latency: "
        f"{retrieval_time * 1000:.2f} ms"
    )

    # -----------------------------------------------------
    # Answer pipeline
    # -----------------------------------------------------

    start = time.perf_counter()

    answer_result = answer_question(
        question,
        retrieval_result,
        llm,
    )

    answer_time = time.perf_counter() - start

    total_time = retrieval_time + answer_time

    latencies.append(total_time)

    refused = answer_result.get(
        "refused",
        False,
    )

    refusal_reason = answer_result.get(
        "refusal_reason"
    )

    print(
        f"Refused: {refused}"
    )

    print(
        f"Refusal reason: "
        f"{refusal_reason}"
    )

    print(
        f"Answer-pipeline latency: "
        f"{answer_time * 1000:.2f} ms"
    )

    print(
        f"Total latency: "
        f"{total_time * 1000:.2f} ms"
    )

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    if refused:
        correct += 1
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        print(
            "Answer:",
            answer_result.get("answer", ""),
        )


# ---------------------------------------------------------
# Final metrics
# ---------------------------------------------------------

total = len(refusal_questions)

refusal_accuracy = (
    correct / total
    if total
    else 0.0
)


print("\n" + "=" * 70)
print("TEST 1 RESULTS")
print("=" * 70)

print(
    f"Must-refuse questions: {total}"
)

print(
    f"Correctly refused:     {correct}"
)

print(
    f"Refusal accuracy:      {refusal_accuracy:.4f}"
)

print(
    f"Refusal accuracy:      "
    f"{refusal_accuracy * 100:.2f}%"
)

if refusal_accuracy == 1.0:
    print("\nSTATUS: PASS")
else:
    print("\nSTATUS: FAIL")

print("=" * 70)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

output = {
    "test": "answer_pipeline_refusal",
    "total_must_refuse": total,
    "correctly_refused": correct,
    "refusal_accuracy": refusal_accuracy,
    "status": "PASS" if refusal_accuracy == 1.0 else "FAIL",
}

output_file = "eval/refusal_results.json"

with open(
    output_file,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        output,
        f,
        indent=2,
    )

print(
    f"\nResults written to: "
    f"{output_file}"
)

