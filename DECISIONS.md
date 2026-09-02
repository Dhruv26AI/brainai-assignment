# DECISIONS.md

## Corpus: BNS vs BNSS

The linked PDF is entirely the **Bharatiya Nagarik Suraksha Sanhita (BNSS)** — procedural law — not the Bharatiya Nyaya Sanhita (BNS) referenced in the brief's example schema.

This was verified using the title on page 1 and the content of Section 63. The forms range (pages 190–249) also matches the expected BNSS structure.

All chunks use `act_short: "BNSS"`.

The system can answer procedural questions such as arrest, bail, and warrants. It cannot reliably answer substantive offence questions because BNS text is not included in the corpus.

---

## A1: Structure-Aware Ingestion

Used **PyMuPDF block coordinates** instead of plain text extraction because BNSS section headnotes/titles appear in a side margin that alternates between the left and right sides of odd/even pages.

Plain text extraction initially caused these margin notes to be merged incorrectly with the main section text.

The parser therefore treats the **section as the atomic unit**. Long sections are split at subsection and clause boundaries rather than arbitrary character positions.

Design rules:

* Never split in the middle of a sentence.
* Never detach a proviso from its parent section.
* Keep exceptions and illustrations attached to their parent section.
* Preserve chapter → section → subsection → clause structure.
* Stop parsing at the Schedule boundary so the forms do not bleed into Section 531.

Result:

* **530/530 sections parsed**
* **673 chunks generated**
* Cross-references detected.

Known limitation: **2/530 sections are missing extracted titles** because PyMuPDF occasionally merges two adjacent headnote blocks.

---

## A2: Embeddings

Used:

`BAAI/bge-base-en-v1.5`

Configuration:

* 768-dimensional embeddings
* Local execution through `sentence-transformers`
* `passage:` prefix for document chunks
* `query:` prefix for search queries
* One-time batch embedding job

The embedding job is not executed during container startup.

---

## A3: Retrieval

Implemented a hybrid retrieval system using:

* **Qdrant** for dense vector retrieval
* **BM25** for sparse/lexical retrieval
* **Reciprocal Rank Fusion (RRF)** for combining results
* **Cross-encoder reranking** using `ms-marco-MiniLM-L-6-v2`

RRF uses `k=60`.

RRF was chosen instead of directly combining dense and BM25 scores because cosine similarity and BM25 scores are on different, non-comparable scales.

### Direct Section Lookup

Queries containing an explicit section number use a deterministic direct-lookup path.

For example:

`"What does Section 41 say?"`

This bypasses semantic similarity search and directly retrieves the requested section from Qdrant.

### Metadata Filtering

Metadata filters are supported for:

* chapter
* act
* section number

A bug was found during development where filters were applied to Qdrant but not BM25. This allowed unfiltered BM25 results to re-enter through RRF fusion.

The issue was fixed by applying the same metadata filtering to BM25 results before fusion.

---

## A4: Answer Generation & Citation Contract

The confidence threshold uses the cross-encoder rerank score, not the RRF score. RRF is rank-based and is not a reliable absolute relevance measure.

After generation, a citation guard validates every cited section and subsection against the retrieved context. This prevents citations to real sections with fabricated subsections (for example, s.41(99)). If a substantive answer contains no valid citations, the system refuses to return it, protecting the assignment's citation requirement.

Known limitation: differently formatted citations such as [Section 41] instead of [BNSS s.41] may not be recognized by the citation guard and therefore rely on prompt compliance.

LLM choice

Generation is performed locally using Ollama with llama3.2:3b (2.0 GB).

We tested different Ollama model sizes:

A larger ~4.9 GB model was too large to reliably load alongside the embedding model and cross-encoder reranker on the available 8 GB RAM.
A smaller ~1.5 GB model was too weak for reliable answer generation and citation-following.
llama3.2:3b (~2.0 GB) was selected as the practical balance between memory usage and answer quality.

The final evaluation configuration uses:

LLM_PROVIDER=ollama

OLLAMA_MODEL=llama3.2:3b

Ollama runs locally, so no external LLM API key is required for answer generation.

## A5: User Document Ingestion

User-uploaded PDFs are processed separately from the BNSS statute corpus.

The ingestion pipeline is:

`PDF → text extraction → chunking → metadata → embeddings → Qdrant`

User documents are stored in a separate Qdrant collection:

`user_documents`

Each chunk contains session and document metadata, including:

* `session_id`
* `document_id`
* `document_name`
* `page_number`
* `chunk_index`
* `source_type`

This provides session/document isolation between uploaded documents and the BNSS corpus.

Uploaded documents are treated as **untrusted data** and are not allowed to act as system instructions.

**User documents are stored separately from the statutory BNSS collection to prevent uploaded content from being mixed with authoritative statute chunks. Each chunk carries session and document identifiers so retrieval can be restricted to the relevant uploaded document.**

## Known Limitations

The current implementation has several known limitations:

1. Two of the 530 BNSS sections do not have correctly extracted titles because of a PyMuPDF margin-note block merging edge case.
2. Citation validation depends on the expected citation format. Some differently formatted citations may not be recognized.
3. Citation evaluation currently shows 61.90% citation accuracy, so citation validation is not perfect.
4. The local 3B model has higher generation latency than a larger hosted model.
5. Prometheus/Grafana observability and token-based cost tracking are not implemented.
6. The complete React frontend required for Part C was not implemented.
7. The current API implementation covers the Forms API; the complete application API described in the assignment is not implemented.

---

## What I Would Do Differently With Two More Weeks

With additional development time, the main priorities would be:

1. Improve citation validation and citation normalization to increase citation accuracy.
2. Add Prometheus metrics and a Grafana dashboard for retrieval, generation, refusal, and system health.
3. Add token accounting and cost-per-query estimation.
4. Implement the complete FastAPI application API.
5. Build the React frontend with streaming responses, conversation history, uploads, citations, and the forms panel.
6. Improve latency through caching, optimized reranking, and asynchronous processing.
7. Add more evaluation questions and adversarial tests for citation and retrieval failures.

8. ### CI / Self-Hosted Runner

Status: Partial

A GitHub Actions self-hosted runner was successfully registered and connected to the repository. The runner reached the `Listening for Jobs` state.

However, the CI workflow was not fully integrated and validated because the overall project was not completed. Therefore, this part is marked as partial rather than fully implemented.
