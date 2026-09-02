"""
A3 - Qdrant vector store wrapper.

Handles collection creation, upserting chunks (with dense vectors + full metadata as
payload), metadata-filtered dense search, and the deterministic direct-lookup path.

Connection: pass url="http://qdrant:6333" (the docker-compose service name) in production.
For local dev/testing without a running Qdrant server, pass url=":memory:" -- same API,
in-process only, not persisted.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

COLLECTION_NAME = "bnss_sections"


class VectorStore:
    def __init__(self, url: str = ":memory:", vector_size: int = 384):
        self.client = QdrantClient(url) if url != ":memory:" else QdrantClient(location=":memory:")
        self.vector_size = vector_size

    def ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            # Payload indexes so metadata filtering (chapter / act / section_number) is fast,
            # not a full collection scan.
            for field_name in ("chapter", "act_short", "section_number"):
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field_name,
                    field_schema="keyword",
                )

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload=chunk,  # full chunk JSON (act, chapter, section_number, text, ...)
                )
            )
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def dense_search(self, query_vector: list[float], top_k: int = 10, filters: dict | None = None):
        qfilter = self._build_filter(filters) if filters else None
        hits = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=qfilter,
        ).points
        return [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits]

    def exact_section_lookup(self, section_number: str, subsection: str | None = None):
        """Deterministic direct-lookup path -- bypasses vector similarity entirely."""
        must = [FieldCondition(key="section_number", match=MatchValue(value=section_number))]
        if subsection:
            must.append(FieldCondition(key="subsection", match=MatchValue(value=f"({subsection})")))
        results, _ = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=must),
            limit=20,
        )
        return [{"id": r.id, "score": 1.0, "payload": r.payload} for r in results]

    @staticmethod
    def _build_filter(filters: dict) -> Filter:
        must = []
        for key in ("chapter", "act_short", "section_number"):
            if filters.get(key):
                must.append(FieldCondition(key=key, match=MatchValue(value=filters[key])))
        return Filter(must=must)
