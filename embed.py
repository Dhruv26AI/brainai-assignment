import json
import os
import time

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = "data/processed/bnss_chunks.jsonl"
OUTPUT_FILE = "data/embeddings/bns_embeddings.jsonl"

MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Start with 32.
# If your computer runs out of memory, change this to 16 or 8.
BATCH_SIZE = 32


# ---------------------------------------------------------
# Load BNS chunks
# ---------------------------------------------------------

print("=" * 60)
print("Nyaya A2 - BNS Embedding Pipeline")
print("=" * 60)

print(f"\nReading chunks from: {INPUT_FILE}")

chunks = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

print(f"Loaded {len(chunks)} chunks")


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

print(f"\nLoading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

embedding_dimension = model.get_embedding_dimension()

print("Model loaded successfully")
print(f"Embedding dimension: {embedding_dimension}")


# ---------------------------------------------------------
# Prepare text for embedding
# ---------------------------------------------------------

# BGE works with "passage:" for documents.
#
# We keep the original legal text untouched.
# The prefix is only added to the text sent to the model.

texts = []

for chunk in chunks:
    text = chunk.get("text", "").strip()

    if not text:
        text = chunk.get("section_title", "").strip()

    texts.append("passage: " + text)


# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# ---------------------------------------------------------
# Generate embeddings in batches
# ---------------------------------------------------------

print("\nGenerating embeddings...")
print(f"Batch size: {BATCH_SIZE}")
print(f"Total chunks: {len(chunks)}")

start_time = time.perf_counter()

all_embeddings = []

total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_number, start in enumerate(
    range(0, len(texts), BATCH_SIZE),
    start=1
):
    end = min(start + BATCH_SIZE, len(texts))

    batch_texts = texts[start:end]

    embeddings = model.encode(
        batch_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    all_embeddings.extend(embeddings.tolist())

    print(
        f"Batch {batch_number}/{total_batches} "
        f"completed ({end}/{len(texts)} chunks)"
    )


elapsed = time.perf_counter() - start_time

throughput = len(chunks) / elapsed if elapsed > 0 else 0


# ---------------------------------------------------------
# Validate embeddings
# ---------------------------------------------------------

print("\nValidating embeddings...")

if len(all_embeddings) != len(chunks):
    raise RuntimeError(
        f"Embedding count mismatch: "
        f"{len(all_embeddings)} embeddings for "
        f"{len(chunks)} chunks"
    )

for i, embedding in enumerate(all_embeddings):
    if len(embedding) != embedding_dimension:
        raise RuntimeError(
            f"Wrong embedding dimension at index {i}: "
            f"expected {embedding_dimension}, "
            f"got {len(embedding)}"
        )

print("Embedding validation passed")


# ---------------------------------------------------------
# Save embeddings
# ---------------------------------------------------------

print(f"\nSaving embeddings to: {OUTPUT_FILE}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for chunk, embedding in zip(chunks, all_embeddings):

        # Keep the original A1 metadata.
        record = dict(chunk)

        # Add embedding information.
        record["embedding_model"] = MODEL_NAME
        record["embedding_dimension"] = embedding_dimension
        record["embedding"] = embedding

        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


# ---------------------------------------------------------
# Final statistics
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("A2 EMBEDDING COMPLETE")
print("=" * 60)

print(f"Chunks processed:     {len(chunks)}")
print(f"Embeddings generated: {len(all_embeddings)}")
print(f"Embedding dimension:  {embedding_dimension}")
print(f"Batch size:           {BATCH_SIZE}")
print(f"Time taken:           {elapsed:.2f} seconds")
print(f"Throughput:           {throughput:.2f} chunks/second")
print(f"Output file:          {OUTPUT_FILE}")

print("=" * 60)