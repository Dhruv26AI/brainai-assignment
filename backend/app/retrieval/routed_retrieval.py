"""
A5 - Routed Retrieval

Routes a question to:
    - user document index
    - BNSS statute index
    - both indexes

User-document evidence and statutory evidence are kept separate
in the returned result.

Uploaded documents are untrusted data. This module only retrieves
their content; it does not treat document text as instructions.
"""

import sys
import os
import json

# ---------------------------------------------------------
# Make project root importable when running this file directly
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from sentence_transformers import SentenceTransformer, CrossEncoder

from backend.app.retrieval.section_intent import detect_section_intent
from backend.app.retrieval.router import route_query
from backend.app.retrieval.qdrant_store import VectorStore
from backend.app.retrieval.bm25_index import BM25Index


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

QDRANT_URL = "http://localhost:6333"

BNSS_CHUNKS_FILE = "data/processed/bnss_chunks.jsonl"

BNSS_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DOCUMENT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"

VECTOR_SIZE = 768

DOCUMENT_COLLECTION = "user_documents"
BNSS_COLLECTION = "bnss_sections"


# ---------------------------------------------------------
# Load BNSS components
# ---------------------------------------------------------

print("Loading BNSS chunks...")

with open(BNSS_CHUNKS_FILE, "r", encoding="utf-8") as f:
    bnss_chunks = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

print(f"Loaded {len(bnss_chunks)} BNSS chunks")


print(f"Loading BNSS embedding model: {BNSS_EMBED_MODEL}")

bnss_model = SentenceTransformer(BNSS_EMBED_MODEL)

print("BNSS embedding model loaded.")


def embed_bnss_query(query: str) -> list[float]:
    """
    Create a BGE query embedding for the BNSS corpus.
    """

    return bnss_model.encode(
        "query: " + query,
        normalize_embeddings=True,
    ).tolist()


# ---------------------------------------------------------
# Load document embedding model
# ---------------------------------------------------------

print(
    f"Loading document embedding model: "
    f"{DOCUMENT_EMBED_MODEL}"
)

document_model = SentenceTransformer(DOCUMENT_EMBED_MODEL)

print("Document embedding model loaded.")


def embed_document_query(query: str) -> list[float]:
    """
    Create a BGE query embedding for user documents.
    """

    return document_model.encode(
        "query: " + query,
        normalize_embeddings=True,
    ).tolist()


# ---------------------------------------------------------
# Qdrant
# ---------------------------------------------------------

print(f"Connecting to Qdrant at {QDRANT_URL}...")

qdrant = VectorStore(
    url=QDRANT_URL,
    vector_size=VECTOR_SIZE,
)

print("Qdrant connection ready.")


# ---------------------------------------------------------
# Document reranker
# ---------------------------------------------------------

print("Loading cross-encoder reranker...")

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Cross-encoder loaded.")


def rerank(
    query: str,
    results: list[dict],
) -> list[dict]:
    """
    Rerank retrieved results using the cross encoder.
    """

    if not results:
        return []

    pairs = []

    for result in results:
        text = result.get("payload", {}).get("text", "")
        pairs.append((query, text))

    scores = cross_encoder.predict(pairs)

    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    return sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )


# ---------------------------------------------------------
# Document retrieval
# ---------------------------------------------------------

def retrieve_document(
    query: str,
    session_id: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Search ONLY the user_documents collection.

    The session_id is mandatory and is applied as a Qdrant
    payload filter. This prevents one user's document chunks
    from being returned for another user's session.
    """

    if not session_id:
        raise ValueError(
            "session_id is required for document retrieval."
        )

    query_vector = embed_document_query(query)

    client = qdrant.client

    # -----------------------------------------------------
    # IMPORTANT SECURITY BOUNDARY
    # -----------------------------------------------------
    #
    # Only retrieve:
    #     source_type = user_document
    #
    # AND:
    #     session_id = current session
    #
    # This prevents cross-session document leakage.
    # -----------------------------------------------------

    from qdrant_client.models import (
        Filter,
        FieldCondition,
        MatchValue,
    )

    document_filter = Filter(
        must=[
            FieldCondition(
                key="session_id",
                match=MatchValue(value=session_id),
            ),
            FieldCondition(
                key="source_type",
                match=MatchValue(
                    value="user_document"
                ),
            ),
        ]
    )

    hits = client.query_points(
        collection_name=DOCUMENT_COLLECTION,
        query=query_vector,
        query_filter=document_filter,
        limit=top_k,
    ).points

    results = [
        {
            "id": hit.id,
            "dense_score": float(hit.score),
            "payload": hit.payload,
            "source": "user_document",
        }
        for hit in hits
    ]

    results = rerank(query, results)

    return results[:top_k]


# ---------------------------------------------------------
# BNSS retrieval
# ---------------------------------------------------------

def retrieve_statute(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Search ONLY the BNSS statute collection.

    Explicit section numbers use deterministic exact lookup.
    General questions use semantic retrieval.
    """

    # -----------------------------------------------------
    # EXACT SECTION LOOKUP
    # -----------------------------------------------------

    intent = detect_section_intent(query)

    if intent:
        hits = qdrant.exact_section_lookup(
            intent["section_number"],
            intent.get("subsection"),
        )

        results = []

        for hit in hits:
            results.append(
                {
                    "id": hit["id"],
                    "dense_score": 1.0,
                    "rerank_score": 1.0,
                    "payload": hit["payload"],
                    "source": "statute",
                    "retrieval_path": "direct_lookup",
                }
            )

        return results[:top_k]

    # -----------------------------------------------------
    # NORMAL SEMANTIC SEARCH
    # -----------------------------------------------------

    query_vector = embed_bnss_query(query)

    client = qdrant.client

    hits = client.query_points(
        collection_name=BNSS_COLLECTION,
        query=query_vector,
        limit=top_k,
    ).points

    results = [
        {
            "id": hit.id,
            "dense_score": float(hit.score),
            "payload": hit.payload,
            "source": "statute",
        }
        for hit in hits
    ]

    results = rerank(query, results)

    return results[:top_k]


# ---------------------------------------------------------
# Main routed retrieval
# ---------------------------------------------------------

def routed_search(
    query: str,
    session_id: str | None = None,
    top_k: int = 5,
) -> dict:
    """
    Route the question and retrieve from the correct corpus.

    Possible routes:

        document
        statute
        both
    """

    route = route_query(query)

    print()
    print("=" * 60)
    print("A5 ROUTED RETRIEVAL")
    print("=" * 60)
    print(f"Question: {query}")
    print(f"Route: {route}")

    document_results = []
    statute_results = []

    # -----------------------------------------------------
    # DOCUMENT
    # -----------------------------------------------------

    if route in ("document", "both"):

        if not session_id:
            raise ValueError(
                "A session_id is required when "
                "document retrieval is needed."
            )

        document_results = retrieve_document(
            query=query,
            session_id=session_id,
            top_k=top_k,
        )

    # -----------------------------------------------------
    # STATUTE
    # -----------------------------------------------------

    if route in ("statute", "both"):

        statute_results = retrieve_statute(
            query=query,
            top_k=top_k,
        )

    # -----------------------------------------------------
    # Return clearly separated evidence
    # -----------------------------------------------------

    return {
        "query": query,
        "route": route,
        "session_id": session_id,
        "document_results": document_results,
        "statute_results": statute_results,
    }


# ---------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------

def print_results(result: dict):

    print()
    print("=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    print(f"Route: {result['route']}")

    # -----------------------------------------------------
    # USER DOCUMENT RESULTS
    # -----------------------------------------------------

    if result["document_results"]:

        print()
        print("-" * 60)
        print("USER DOCUMENT EVIDENCE")
        print("-" * 60)

        for i, item in enumerate(
            result["document_results"],
            start=1,
        ):

            payload = item["payload"]

            print()
            print(f"--- Document Result {i} ---")

            print(
                f"Document: "
                f"{payload.get('document_name')}"
            )

            print(
                f"Page: "
                f"{payload.get('page_number')}"
            )

            print(
                f"Session: "
                f"{payload.get('session_id')}"
            )

            print(
                f"Dense score: "
                f"{item.get('dense_score', 0):.4f}"
            )

            print(
                f"Rerank score: "
                f"{item.get('rerank_score', 0):.4f}"
            )

            print(
                f"Text: "
                f"{payload.get('text', '')[:500]}"
            )

    # -----------------------------------------------------
    # STATUTE RESULTS
    # -----------------------------------------------------

    if result["statute_results"]:

        print()
        print("-" * 60)
        print("STATUTORY AUTHORITY")
        print("-" * 60)

        for i, item in enumerate(
            result["statute_results"],
            start=1,
        ):

            payload = item["payload"]

            print()
            print(f"--- Statute Result {i} ---")

            print(
                f"Section: "
                f"{payload.get('section_number')}"
            )

            print(
                f"Title: "
                f"{payload.get('section_title')}"
            )

            print(
                f"Dense score: "
                f"{item.get('dense_score', 0):.4f}"
            )

            print(
                f"Rerank score: "
                f"{item.get('rerank_score', 0):.4f}"
            )

            print(
                f"Text: "
                f"{payload.get('text', '')[:500]}"
            )

    print()
    print("=" * 60)
    print("END RETRIEVAL")
    print("=" * 60)


# ---------------------------------------------------------
# Command-line test
# ---------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 3:

        print(
            "Usage:"
        )

        print(
            'python backend/app/retrieval/'
            'routed_retrieval.py '
            '"question" "session_id"'
        )

        print()
        print("Examples:")

        print(
            'python backend/app/retrieval/'
            'routed_retrieval.py '
            '"What does my notice say about payment?" '
            '"session_001"'
        )

        print(
            'python backend/app/retrieval/'
            'routed_retrieval.py '
            '"What is BNSS Section 35?" '
            '"session_001"'
        )

        print(
            'python backend/app/retrieval/'
            'routed_retrieval.py '
            '"Does this notice comply with Section 35?" '
            '"session_001"'
        )

        sys.exit(1)

    question = sys.argv[1]
    session_id = sys.argv[2]

    result = routed_search(
        query=question,
        session_id=session_id,
        top_k=5,
    )

    print_results(result)