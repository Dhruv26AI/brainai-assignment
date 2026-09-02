"""
Part F - Retrieval Evaluation

Evaluates the BNSS retrieval system using two configurations:

Configuration 1:
    Dense retrieval + Cross-Encoder reranking
    This represents the current baseline.

Configuration 2:
    Dense retrieval only
    Same embedding model and Qdrant collection, but without reranking.

Metrics:
    - Recall@5
    - Recall@10
    - MRR
    - Retrieval p50
    - Retrieval p95
    - Must-refuse set is reported separately

Important:
    This file evaluates retrieval only.
    It does not generate LLM answers yet.
"""

import json
import os
import sys
import time
import statistics


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

from backend.app.retrieval.routed_retrieval import (
    embed_bnss_query,
    retrieve_statute,
    qdrant,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

GOLDEN_FILE = "eval/golden_set.jsonl"

TOP_K = 10

BNSS_COLLECTION = "bnss_sections"


# ---------------------------------------------------------
# Golden set
# ---------------------------------------------------------

def load_golden_set(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


# ---------------------------------------------------------
# Section normalization
# ---------------------------------------------------------

def normalize_section(section):
    """
    Normalize section identifiers.

    Examples:

        BNSS s.35 -> 35
        BNSS s.105 -> 105
        Section 35 -> 35
        35 -> 35
    """

    if section is None:
        return None

    text = str(section).strip().lower()

    text = text.replace("bnss", "")
    text = text.replace("bns", "")
    text = text.replace("section", "")
    text = text.replace("sec.", "")
    text = text.replace("sec", "")
    text = text.replace("s.", "")

    return text.strip()


def expected_section_numbers(row):
    expected = set()

    for section in row.get("expected_sections", []):
        normalized = normalize_section(section)

        if normalized:
            expected.add(normalized)

    return expected


# ---------------------------------------------------------
# Extract sections
# ---------------------------------------------------------

def result_section_numbers(results):
    """
    Extract section numbers from either:

        {"payload": {...}}

    or directly from payload dictionaries.
    """

    found = []

    for result in results:

        payload = result.get(
            "payload",
            result,
        )

        section = payload.get(
            "section_number"
        )

        if section is not None:
            found.append(
                normalize_section(section)
            )

    return found


# ---------------------------------------------------------
# Configuration 1
# ---------------------------------------------------------

def retrieve_baseline(question):
    """
    Configuration 1:

        BGE embedding
            ↓
        Qdrant dense retrieval
            ↓
        Cross-encoder reranking

    This uses the existing retrieve_statute()
    implementation.
    """

    start = time.perf_counter()

    results = retrieve_statute(
        query=question,
        top_k=TOP_K,
    )

    elapsed = time.perf_counter() - start

    return results, elapsed


# ---------------------------------------------------------
# Configuration 2
# ---------------------------------------------------------

def retrieve_dense_only(question):
    """
    Configuration 2:

        BGE embedding
            ↓
        Qdrant dense retrieval
            ↓
        Top-k

    No cross-encoder reranking.

    Explicit section lookup is intentionally NOT used
    here because this configuration is intended to measure
    pure dense retrieval.
    """

    start = time.perf_counter()

    query_vector = embed_bnss_query(question)

    results = qdrant.dense_search(
        query_vector=query_vector,
        top_k=TOP_K,
    )

    elapsed = time.perf_counter() - start

    return results, elapsed


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(
    rows,
    retrieval_function,
    configuration_name,
):
    recall5_hits = 0
    recall10_hits = 0

    reciprocal_ranks = []

    latencies = []

    evaluated = 0
    refusal_total = 0

    print()
    print("=" * 70)
    print(configuration_name)
    print("=" * 70)

    for index, row in enumerate(rows, start=1):

        question = row["q"]
        question_type = row.get("type")

        expected = expected_section_numbers(row)

        print()
        print("-" * 70)
        print(
            f"Question {index}/{len(rows)}"
        )
        print(f"Q: {question}")
        print(f"Type: {question_type}")
        print(f"Expected: {sorted(expected)}")

        results, elapsed = retrieval_function(
            question
        )

        latencies.append(elapsed)

        retrieved = result_section_numbers(
            results
        )

        print(
            f"Retrieved top-10: {retrieved}"
        )

        print(
            f"Latency: "
            f"{elapsed * 1000:.2f} ms"
        )

        # -------------------------------------------------
        # Must-refuse questions
        # -------------------------------------------------

        if question_type == "must_refuse":

            refusal_total += 1

            continue

        # -------------------------------------------------
        # Retrieval questions
        # -------------------------------------------------

        if not expected:
            continue

        evaluated += 1

        # -------------------------------------------------
        # Recall@5
        # -------------------------------------------------

        top5 = retrieved[:5]

        hit5 = any(
            section in top5
            for section in expected
        )

        if hit5:
            recall5_hits += 1

        # -------------------------------------------------
        # Recall@10
        # -------------------------------------------------

        top10 = retrieved[:10]

        hit10 = any(
            section in top10
            for section in expected
        )

        if hit10:
            recall10_hits += 1

        # -------------------------------------------------
        # MRR
        # -------------------------------------------------

        rank = None

        for position, section in enumerate(
            top10,
            start=1,
        ):

            if section in expected:
                rank = position
                break

        if rank is not None:
            reciprocal_ranks.append(
                1.0 / rank
            )
        else:
            reciprocal_ranks.append(
                0.0
            )

    # -----------------------------------------------------
    # Aggregate
    # -----------------------------------------------------

    recall5 = (
        recall5_hits / evaluated
        if evaluated
        else 0.0
    )

    recall10 = (
        recall10_hits / evaluated
        if evaluated
        else 0.0
    )

    mrr = (
        statistics.mean(
            reciprocal_ranks
        )
        if reciprocal_ranks
        else 0.0
    )

    sorted_latencies = sorted(
        latencies
    )

    if sorted_latencies:

        p50 = statistics.median(
            sorted_latencies
        )

        p95_index = min(
            int(
                len(sorted_latencies)
                * 0.95
            ),
            len(sorted_latencies) - 1,
        )

        p95 = sorted_latencies[
            p95_index
        ]

    else:

        p50 = 0.0
        p95 = 0.0

    # -----------------------------------------------------
    # Print metrics
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"{configuration_name} RESULTS"
    )
    print("=" * 70)

    print(
        f"Total questions:        {len(rows)}"
    )

    print(
        f"Retrieval questions:    {evaluated}"
    )

    print(
        f"Must-refuse questions:  {refusal_total}"
    )

    print()

    print(
        f"Recall@5:               "
        f"{recall5:.4f}"
    )

    print(
        f"Recall@10:              "
        f"{recall10:.4f}"
    )

    print(
        f"MRR:                    "
        f"{mrr:.4f}"
    )

    print()

    print(
        f"Retrieval p50:          "
        f"{p50 * 1000:.2f} ms"
    )

    print(
        f"Retrieval p95:          "
        f"{p95 * 1000:.2f} ms"
    )

    print("=" * 70)

    return {
        "configuration": configuration_name,
        "questions": len(rows),
        "retrieval_questions": evaluated,
        "must_refuse": refusal_total,
        "recall@5": recall5,
        "recall@10": recall10,
        "mrr": mrr,
        "retrieval_p50_ms": p50 * 1000,
        "retrieval_p95_ms": p95 * 1000,
    }


# ---------------------------------------------------------
# Comparison
# ---------------------------------------------------------

def print_comparison(
    baseline,
    dense_only,
):
    print()
    print()
    print("=" * 70)
    print("PART F - RETRIEVAL CONFIGURATION COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Metric':<25}"
        f"{'Dense + Reranker':>20}"
        f"{'Dense-only':>20}"
    )

    print("-" * 70)

    print(
        f"{'Recall@5':<25}"
        f"{baseline['recall@5']:>20.4f}"
        f"{dense_only['recall@5']:>20.4f}"
    )

    print(
        f"{'Recall@10':<25}"
        f"{baseline['recall@10']:>20.4f}"
        f"{dense_only['recall@10']:>20.4f}"
    )

    print(
        f"{'MRR':<25}"
        f"{baseline['mrr']:>20.4f}"
        f"{dense_only['mrr']:>20.4f}"
    )

    print(
        f"{'Retrieval p50 (ms)':<25}"
        f"{baseline['retrieval_p50_ms']:>20.2f}"
        f"{dense_only['retrieval_p50_ms']:>20.2f}"
    )

    print(
        f"{'Retrieval p95 (ms)':<25}"
        f"{baseline['retrieval_p95_ms']:>20.2f}"
        f"{dense_only['retrieval_p95_ms']:>20.2f}"
    )

    print("=" * 70)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    if not os.path.exists(
        GOLDEN_FILE
    ):
        raise FileNotFoundError(
            f"Golden set not found: "
            f"{GOLDEN_FILE}"
        )

    rows = load_golden_set(
        GOLDEN_FILE
    )

    print(
        f"Loaded golden set: "
        f"{len(rows)} questions"
    )

    # -----------------------------------------------------
    # Configuration 1
    # -----------------------------------------------------

    baseline = calculate_metrics(
        rows=rows,
        retrieval_function=retrieve_baseline,
        configuration_name=(
            "CONFIGURATION 1 - "
            "DENSE + CROSS-ENCODER"
        ),
    )

    # -----------------------------------------------------
    # Configuration 2
    # -----------------------------------------------------

    dense_only = calculate_metrics(
        rows=rows,
        retrieval_function=retrieve_dense_only,
        configuration_name=(
            "CONFIGURATION 2 - "
            "DENSE-ONLY"
        ),
    )

    # -----------------------------------------------------
    # Comparison
    # -----------------------------------------------------

    print_comparison(
        baseline,
        dense_only,
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    output = {
        "golden_set": GOLDEN_FILE,
        "configurations": {
            "dense_plus_reranker": baseline,
            "dense_only": dense_only,
        },
    }

    output_file = (
        "eval/results.json"
    )

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

    print()
    print(
        f"Results written to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()