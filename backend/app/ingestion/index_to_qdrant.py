"""
Loads the real embeddings produced by A2 (bns_embeddings.jsonl) into a running Qdrant
instance. Run this AFTER Qdrant is up (docker run -p 6333:6333 qdrant/qdrant) and AFTER
A2's embed script has produced data/embeddings/bns_embeddings.jsonl.

This is a one-time (or re-runnable) cold-start ingestion job -- not something that should
run on every container boot. Re-running it simply re-upserts the same points (same IDs),
so it's safe to run again if you regenerate embeddings.

Usage:
    python backend/app/ingestion/index_to_qdrant.py
"""
import json
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

INPUT_FILE = "data/embeddings/bns_embeddings.jsonl"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bnss_sections"


def load_embedded_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def main():
    print("=" * 60)
    print("Indexing embeddings into Qdrant")
    print("=" * 60)

    print(f"\nReading embedded chunks from: {INPUT_FILE}")
    records = load_embedded_chunks(INPUT_FILE)
    print(f"Loaded {len(records)} embedded chunks")

    vector_size = records[0]["embedding_dimension"]
    print(f"Vector size: {vector_size}")

    print(f"\nConnecting to Qdrant at {QDRANT_URL} ...")
    client = QdrantClient(url=QDRANT_URL)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists -- recreating it fresh.")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION_NAME}'")

    for field_name in ("chapter", "act_short", "section_number"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema="keyword",
        )
    print("Created payload indexes for: chapter, act_short, section_number")

    print("\nUpserting points...")
    start = time.perf_counter()

    points = []
    for idx, record in enumerate(records):
        embedding = record.pop("embedding")  # pull vector out, rest becomes payload
        points.append(PointStruct(id=idx, vector=embedding, payload=record))

    batch_size = 256
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"  upserted {min(i + batch_size, len(points))}/{len(points)}")

    elapsed = time.perf_counter() - start

    count = client.count(collection_name=COLLECTION_NAME).count
    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"Points in collection '{COLLECTION_NAME}': {count}")
    print(f"Time taken: {elapsed:.2f}s")
    print(f"Verify at: {QDRANT_URL}/dashboard")


if __name__ == "__main__":
    main()
