"""
Part F - Test 4
End-to-end latency evaluation.

Measures:
- Retrieval latency
- LLM generation latency
- End-to-end latency
- p50 / p95 for each

Uses the existing BNSS retrieval + LLM pipeline.
"""

import json
import os
import sys
import time
import statistics

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

from sentence_transformers import SentenceTransformer, CrossEncoder

from backend.app.retrieval.qdrant_store import VectorStore
from backend.app.retrieval.bm25_index import BM25Index
from backend.app.retrieval.hybrid_search import HybridRetriever

from backend.app.llm.providers import get_llm_provider
from backend.app.llm.answer import answer_question


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHUNKS_FILE = "data/processed/bnss_chunks.jsonl"

QDRANT_URL = "http://localhost:6333"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

VECTOR_SIZE = 768

TOP_K = 10


# ---------------------------------------------------------
# Test questions
# ---------------------------------------------------------

QUESTIONS = [
    "What is the short title of the Bharatiya Nagarik Suraksha Sanhita, 2023?",
    "What is the punishment for culpable homicide not amounting to murder?",
    "What does Section 35 say about arrest without warrant?",
    "Does Section 35 require reasons to be recorded when an arrest is not made?",
    "What does BNSS Section 78 say about bringing an arrested person before a Court?",
    "What does BNSS Section 482 provide for a person apprehending arrest?",
    "What is the procedure for arrest when a police officer receives a requisition from another police officer?",
    "Can a police officer arrest a person for preventing the commission of a cognizable offence?",
    "What are the provisions regarding detention during investigation under Section 187?",
    "What does Section 105 provide regarding search and seizure?"
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def percentile_95(values):
    if not values:
        return 0.0

    values = sorted(values)

    index = min(
        len(values) - 1,
        int(len(values) * 0.95)
    )

    return values[index]


def stats(values):
    if not values:
        return {
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "average_ms": 0.0,
        }

    return {
        "p50_ms": statistics.median(values) * 1000,
        "p95_ms": percentile_95(values) * 1000,
        "average_ms": statistics.mean(values) * 1000,
    }


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

print("=" * 70)
print("PART F - TEST 4: END-TO-END LATENCY")
print("=" * 70)

print("\nLoading BM25 index...")

with open(
    CHUNKS_FILE,
    encoding="utf-8"
) as f:
    chunks = [
        json.loads(line)
        for line in f
    ]

bm25 = BM25Index(chunks)

print("BM25 index loaded.")


print(
    f"\nConnecting to Qdrant at {QDRANT_URL}..."
)

store = VectorStore(
    url=QDRANT_URL,
    vector_size=VECTOR_SIZE
)

print("Qdrant connection ready.")


print(
    f"\nLoading embedding model: {EMBED_MODEL}"
)

model = SentenceTransformer(
    EMBED_MODEL
)


def embed_query(q):
    return model.encode(
        "query: " + q,
        normalize_embeddings=True
    ).tolist()


print("\nLoading cross-encoder reranker...")

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Cross-encoder loaded.")


def rerank(query, candidates):

    if not candidates:
        return []

    pairs = [
        (
            query,
            c["payload"]["text"]
        )
        for c in candidates
    ]

    scores = cross_encoder.predict(pairs)

    for candidate, score in zip(
        candidates,
        scores
    ):
        candidate["rerank_score"] = float(score)

    return sorted(
        candidates,
        key=lambda c: c["rerank_score"],
        reverse=True
    )


retriever = HybridRetriever(
    store,
    bm25,
    embed_query,
    reranker=rerank
)


print("\nLoading LLM provider...")

llm = get_llm_provider()

print("LLM provider ready.")

print(
    f"\nLatency questions: {len(QUESTIONS)}"
)

print("=" * 70)
print("RUNNING LATENCY TEST")
print("=" * 70)


# ---------------------------------------------------------
# Run test
# ---------------------------------------------------------

retrieval_latencies = []
generation_latencies = []
end_to_end_latencies = []

question_results = []


for i, question in enumerate(
    QUESTIONS,
    1
):

    print("\n" + "-" * 70)

    print(
        f"Question {i}/{len(QUESTIONS)}"
    )

    print(
        f"Q: {question}"
    )


    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    retrieval_start = time.perf_counter()

    retrieval_result = retriever.search(
        question,
        top_k=TOP_K
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )


    # -----------------------------------------------------
    # Generation + answer pipeline
    # -----------------------------------------------------

    generation_start = time.perf_counter()

    result = answer_question(
        question,
        retrieval_result,
        llm
    )

    generation_time = (
        time.perf_counter()
        - generation_start
    )


    # -----------------------------------------------------
    # End-to-end
    # -----------------------------------------------------

    end_to_end_time = (
        retrieval_time
        + generation_time
    )


    retrieval_latencies.append(
        retrieval_time
    )

    generation_latencies.append(
        generation_time
    )

    end_to_end_latencies.append(
        end_to_end_time
    )


    question_results.append(
        {
            "question": question,
            "retrieval_ms":
                retrieval_time * 1000,
            "generation_ms":
                generation_time * 1000,
            "end_to_end_ms":
                end_to_end_time * 1000,
            "refused":
                result.get(
                    "refused",
                    False
                ),
        }
    )


    print(
        f"Retrieval latency: "
        f"{retrieval_time * 1000:.2f} ms"
    )

    print(
        f"Generation latency: "
        f"{generation_time * 1000:.2f} ms"
    )

    print(
        f"End-to-end latency: "
        f"{end_to_end_time * 1000:.2f} ms"
    )


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

retrieval_stats = stats(
    retrieval_latencies
)

generation_stats = stats(
    generation_latencies
)

end_to_end_stats = stats(
    end_to_end_latencies
)


print("\n")
print("=" * 70)
print("TEST 4 RESULTS")
print("=" * 70)

print(
    f"Evaluation questions: "
    f"{len(QUESTIONS)}"
)

print("\nRetrieval:")
print(
    f"  p50:     "
    f"{retrieval_stats['p50_ms']:.2f} ms"
)

print(
    f"  p95:     "
    f"{retrieval_stats['p95_ms']:.2f} ms"
)

print(
    f"  average: "
    f"{retrieval_stats['average_ms']:.2f} ms"
)


print("\nGeneration:")
print(
    f"  p50:     "
    f"{generation_stats['p50_ms']:.2f} ms"
)

print(
    f"  p95:     "
    f"{generation_stats['p95_ms']:.2f} ms"
)

print(
    f"  average: "
    f"{generation_stats['average_ms']:.2f} ms"
)


print("\nEnd-to-end:")
print(
    f"  p50:     "
    f"{end_to_end_stats['p50_ms']:.2f} ms"
)

print(
    f"  p95:     "
    f"{end_to_end_stats['p95_ms']:.2f} ms"
)

print(
    f"  average: "
    f"{end_to_end_stats['average_ms']:.2f} ms"
)

print("=" * 70)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

output = {
    "test": "end_to_end_latency",
    "questions": len(QUESTIONS),

    "retrieval": retrieval_stats,

    "generation": generation_stats,

    "end_to_end": end_to_end_stats,

    "question_results": question_results,
}


output_file = (
    "eval/latency_results.json"
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )


print(
    f"\nResults written to: "
    f"{output_file}"
)

print(
    "=" * 70
)