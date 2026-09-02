"""
A5 - User document ingestion.

Handles uploaded user documents separately from the BNSS statute corpus.

Flow:
    PDF
      ↓
    Extract text
      ↓
    Chunk text
      ↓
    Add session/document metadata
      ↓
    Generate BGE embeddings
      ↓
    Store in a separate Qdrant collection

Important:
    Uploaded documents are UNTRUSTED DATA.
    Their contents must never be treated as system instructions.
"""

import os
import re
import uuid
import datetime
from typing import Optional

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

QDRANT_URL = "http://localhost:6333"

# IMPORTANT:
# Keep user documents separate from the BNSS statute collection.
DOCUMENT_COLLECTION = "user_documents"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

BATCH_SIZE = 32
MAX_CHARS = 1600


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

print(f"Loading embedding model: {EMBED_MODEL}")
model = SentenceTransformer(EMBED_MODEL)

VECTOR_SIZE = model.get_embedding_dimension()

print(f"Embedding dimension: {VECTOR_SIZE}")


# ---------------------------------------------------------
# Qdrant
# ---------------------------------------------------------

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_document_collection(client: QdrantClient):
    """
    Create the user-document collection if it does not exist.
    """

    existing = [
        c.name
        for c in client.get_collections().collections
    ]

    if DOCUMENT_COLLECTION not in existing:
        client.create_collection(
            collection_name=DOCUMENT_COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        # These indexes allow Qdrant to efficiently restrict
        # searches to one user's session/document.
        for field_name in (
            "session_id",
            "document_id",
            "source_type",
        ):
            client.create_payload_index(
                collection_name=DOCUMENT_COLLECTION,
                field_name=field_name,
                field_schema="keyword",
            )

        print(
            f"Created Qdrant collection: "
            f"{DOCUMENT_COLLECTION}"
        )


# ---------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------

def extract_pdf_pages(pdf_path: str) -> list[dict]:
    """
    Extract text page-by-page.

    Returns:
        [
            {
                "page_number": 1,
                "text": "..."
            },
            ...
        ]
    """

    doc = fitz.open(pdf_path)

    pages = []

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]

            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "page_number": page_index + 1,
                        "text": text,
                    }
                )
    finally:
        doc.close()

    return pages


# ---------------------------------------------------------
# Text chunking
# ---------------------------------------------------------

def split_text(
    text: str,
    max_chars: int = MAX_CHARS,
) -> list[str]:
    """
    Simple paragraph/sentence-aware chunking.

    This is intentionally separate from the BNSS parser because
    uploaded documents can be FIRs, notices, agreements,
    judgments, etc.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    chunks = []
    current = ""

    for sentence in sentences:

        if not sentence.strip():
            continue

        candidate = (
            f"{current} {sentence}".strip()
            if current
            else sentence
        )

        if len(candidate) <= max_chars:
            current = candidate
        else:

            if current:
                chunks.append(current)

            # Handle a single very large sentence.
            if len(sentence) > max_chars:
                for i in range(
                    0,
                    len(sentence),
                    max_chars,
                ):
                    chunks.append(
                        sentence[i:i + max_chars].strip()
                    )
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------
# Build document chunks
# ---------------------------------------------------------

def build_document_chunks(
    pdf_path: str,
    session_id: str,
    document_id: Optional[str] = None,
) -> list[dict]:
    """
    Convert a PDF into chunks with session isolation metadata.
    """

    if document_id is None:
        document_id = str(uuid.uuid4())

    pages = extract_pdf_pages(pdf_path)

    chunks = []

    for page in pages:

        page_chunks = split_text(
            page["text"]
        )

        for chunk_index, text in enumerate(
            page_chunks,
            start=1,
        ):

            chunks.append(
                {
                    "source_type": "user_document",

                    # SECURITY BOUNDARY
                    "session_id": session_id,

                    "document_id": document_id,

                    "document_name": os.path.basename(
                        pdf_path
                    ),

                    "page_number": page[
                        "page_number"
                    ],

                    "chunk_index": chunk_index,

                    "text": text,

                    "ingested_at": (
                        datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                    ),
                }
            )

    return chunks


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------

def embed_chunks(
    chunks: list[dict],
) -> list[list[float]]:

    texts = [
        "passage: " + chunk["text"]
        for chunk in chunks
    ]

    embeddings = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):

        batch = texts[
            start:start + BATCH_SIZE
        ]

        vectors = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings.extend(
            vectors.tolist()
        )

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match "
            "chunk count."
        )

    return embeddings


# ---------------------------------------------------------
# Store in Qdrant
# ---------------------------------------------------------

def store_document(
    chunks: list[dict],
    embeddings: list[list[float]],
):
    """
    Store document chunks in the USER_DOCUMENTS collection.
    """

    client = get_qdrant_client()

    ensure_document_collection(client)

    points = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):

        # Random UUID avoids collisions between documents
        # and sessions.
        point_id = str(uuid.uuid4())

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=chunk,
            )
        )

    batch_size = 256

    for start in range(
        0,
        len(points),
        batch_size,
    ):

        batch = points[
            start:start + batch_size
        ]

        client.upsert(
            collection_name=DOCUMENT_COLLECTION,
            points=batch,
        )

    print(
        f"Stored {len(points)} chunks "
        f"in {DOCUMENT_COLLECTION}"
    )


# ---------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------

def ingest_document(
    pdf_path: str,
    session_id: str,
    document_id: Optional[str] = None,
) -> dict:
    """
    Complete A5 document ingestion pipeline.

    Returns metadata about the ingested document.
    """

    if not session_id:
        raise ValueError(
            "session_id is required."
        )

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"Document not found: {pdf_path}"
        )

    if document_id is None:
        document_id = str(uuid.uuid4())

    print("=" * 60)
    print("A5 USER DOCUMENT INGESTION")
    print("=" * 60)

    print(f"Document: {pdf_path}")
    print(f"Session ID: {session_id}")
    print(f"Document ID: {document_id}")

    # 1. Extract + chunk
    chunks = build_document_chunks(
        pdf_path=pdf_path,
        session_id=session_id,
        document_id=document_id,
    )

    print(
        f"Created {len(chunks)} document chunks"
    )

    if not chunks:
        raise ValueError(
            "No text could be extracted from "
            "the uploaded document."
        )

    # 2. Embed
    print("Generating embeddings...")

    embeddings = embed_chunks(chunks)

    print(
        f"Generated {len(embeddings)} embeddings"
    )

    # 3. Store
    print("Storing document in Qdrant...")

    store_document(
        chunks,
        embeddings,
    )

    print("=" * 60)
    print("DOCUMENT INGESTION COMPLETE")
    print("=" * 60)

    return {
        "document_id": document_id,
        "session_id": session_id,
        "document_name": os.path.basename(
            pdf_path
        ),
        "chunks": len(chunks),
        "source_type": "user_document",
    }


# ---------------------------------------------------------
# Command-line test
# ---------------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 3:
        print(
            "Usage:"
        )
        print(
            "python "
            "backend/app/ingestion/document_ingest.py "
            "<pdf_path> <session_id>"
        )
        sys.exit(1)

    pdf_path = sys.argv[1]
    session_id = sys.argv[2]

    result = ingest_document(
        pdf_path=pdf_path,
        session_id=session_id,
    )

    print("\nResult:")
    print(result)