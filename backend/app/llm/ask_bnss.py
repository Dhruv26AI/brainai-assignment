"""
Full end-to-end test: real hybrid retrieval (A3) -> real LLM generation via Ollama (A4) ->
real citation validation. This is the first script that proves the WHOLE citation contract
works with an actual model, not a mock.

Run from project root, with Qdrant running AND Ollama running (ollama pull llama3.1 first):

    python backend/app/llm/ask_bnss.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from sentence_transformers import SentenceTransformer, CrossEncoder
from backend.app.retrieval.qdrant_store import VectorStore
from backend.app.retrieval.bm25_index import BM25Index
from backend.app.retrieval.hybrid_search import HybridRetriever
from backend.app.llm.providers import get_llm_provider
from backend.app.llm.answer import answer_question


CHUNKS_FILE = "data/processed/bnss_chunks.jsonl"
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768


print("Loading BM25 index...")
chunks = [json.loads(l) for l in open(CHUNKS_FILE, encoding="utf-8")]
bm25 = BM25Index(chunks)


print(f"Connecting to Qdrant at {QDRANT_URL} ...")
store = VectorStore(
    url=QDRANT_URL,
    vector_size=VECTOR_SIZE
)


print(f"Loading embedding model ({EMBED_MODEL})...")
model = SentenceTransformer(EMBED_MODEL)


def embed_query(q: str) -> list[float]:
    return model.encode(
        "query: " + q,
        normalize_embeddings=True
    ).tolist()


print("Loading cross-encoder reranker...")
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    pairs = [
        (query, c["payload"]["text"])
        for c in candidates
    ]

    scores = cross_encoder.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

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


print(
    "Connecting to LLM provider "
    "(set LLM_PROVIDER=ollama or groq; default ollama)..."
)

llm = get_llm_provider()


print("\nReady. Ask a real question about BNSS (or 'quit' to exit).\n")


while True:
    question = input("Ask> ").strip()

    if not question or question.lower() in ("quit", "exit"):
        break

    # ============================================================
    # 1. RETRIEVE
    # ============================================================

    # Increased from top_k=5 to top_k=10
    # so that related provisions such as BNSS Section 35
    # have a better chance of being included in the context.
    retrieval_result = retriever.search(
        question,
        top_k=10
    )

    # ============================================================
    # 2. DEBUG: PRINT RETRIEVED CHUNKS
    # ============================================================

    print("\n========== RETRIEVED CHUNKS ==========")

    for i, r in enumerate(
        retrieval_result["results"],
        1
    ):
        payload = r["payload"]

        print(f"\n--- Chunk {i} ---")

        print(
            f"Section: "
            f"BNSS s.{payload['section_number']}"
        )

        print(
            f"Title: "
            f"{payload.get('section_title')}"
        )

        print(
            f"Rerank score: "
            f"{r.get('rerank_score')}"
        )

        print(
            f"Pages: "
            f"{payload.get('page_start')}-"
            f"{payload.get('page_end')}"
        )

        print(
            f"Text:\n"
            f"{payload['text']}"
        )

    print(
        "\n========== END RETRIEVED CHUNKS ==========\n"
    )

    # ============================================================
    # 3. GENERATE ANSWER + CITATION VALIDATION
    # ============================================================

    result = answer_question(
        question,
        retrieval_result,
        llm
    )

    print(
        f"\n[retrieval_path: "
        f"{result.get('retrieval_path')}]"
    )

    # ============================================================
    # 4. DEBUG: RAW LLM OUTPUT
    # ============================================================

    if os.environ.get("DEBUG_RAW") == "1":

        if "_debug_raw_answer" in result:
            print(
                "[DEBUG raw LLM output]:\n"
                f"{result['_debug_raw_answer']}\n"
            )
        else:
            print(
                "[DEBUG raw LLM output]: "
                "<not available>\n"
            )

    # ============================================================
    # 5. FINAL RESULT
    # ============================================================

    if result.get("refused", False):

        refusal_reason = result.get(
            "refusal_reason",
            "refusal_reason was not provided by answer.py"
        )

        print(
            f"REFUSED "
            f"(reason: {refusal_reason}, "
            f"top_score: {result.get('top_score')})"
        )

        print(
            f"Answer: "
            f"{result.get('answer', '')}"
        )

    else:

        print(
            f"Answer:\n"
            f"{result.get('answer', '')}\n"
        )

        if result.get(
            "had_hallucination",
            False
        ):
            print(
                "!! Citation guard stripped "
                f"{len(result.get('invalid_citations_stripped', []))} "
                "hallucinated citation(s): "
                f"{result.get('invalid_citations_stripped', [])}"
            )

        print(
            "\nSources "
            "(for the expandable source panel):"
        )

        for s in result.get("sources", []):

            print(
                f"  - BNSS s."
                f"{s['section_number']}"
                f"{s['subsection'] or ''} "
                f"({s['section_title']}), "
                f"page "
                f"{s['page_start']}-"
                f"{s['page_end']}"
            )

    print()


