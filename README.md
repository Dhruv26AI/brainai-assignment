
# DhronAI — Technical Assignment

## 1. What Has Been Implemented

| Part | Area | Status |
|---|---|---|
| **Part A** | Retrieval & Indexing | **Done** |
| **Part B** | Forms Extraction Pipeline | **Done** |
| **Part C** | Frontend & UX | **Not Attempted** |
| **Part D** | Backend/API | **Partial** |
| **Part E** | Security / Reliability | **Partial** |
| **Part F** | Evaluation & Observability | **Partial — ~90% of implemented evaluation requirements** |

### Part A — Retrieval & Indexing
**Status: Done**

- A1 — Structure-aware ingestion: **Done**
- A2 — Embeddings: **Done**
- A3 — Vector store & hybrid retrieval: **Done**
- A4 — Query understanding, answer generation & citation contract: **Done**
- A5 — User document ingestion: **Done**

### Part B — Forms Extraction Pipeline
**Status: Done**

- B1 — Page-perfect PDF extraction: **Done**
- B2 — Scraped title-based filenames: **Done**
- B3 — Multi-page form detection: **Done**
- B4 — `forms_manifest.json`: **Done**
- B5 — OCR fallback: **Done**
- B6 — Idempotent extraction: **Done**
- B7 — Forms API endpoints: **Done**

Validation:

- 58 forms extracted
- FORM 33 correctly extracted as one 3-page PDF
- 0 forms flagged for review
- 5/5 forms tests passed

### Part C — Frontend & UX

**Status: Partial / Started**

A basic React + Tailwind frontend has been started with the initial DhronAI chat layout.

Implemented:

- Basic React application structure.
- Initial two-panel layout with a sidebar and chat panel.
- Basic chat interface with message input and Send button.
- Empty state with four example questions.
- Initial Tailwind styling and basic focus states.

The following Part C requirements are not yet implemented:

- Token streaming using SSE/WebSocket.
- Multi-turn conversation history.
- Conversation list with rename and delete functionality.
- Citation chips and source drawer showing statutory text and page.
- Drag-and-drop/click document upload with parse → chunk → embed → ready progress.
- Markdown rendering, code/quote blocks, copy button, stop-generation, and regenerate.
- Useful error states for file size, unsupported files, model timeout, and empty retrieval.
- Forms panel with searchable/filterable forms, preview, single download, and bulk ZIP download.
- Full responsive/mobile optimization.
- Complete keyboard accessibility, ARIA support, and WCAG AA review.
- Dark/light mode.
- Protection against layout shift while long answers stream.

### Part D — Backend / API
**Status: Partial**

| Requirement | Status |
|---|---|
| Forms API | **Done** |
| `GET /api/v1/forms` | **Done** |
| `GET /api/v1/forms/{id}/download` | **Done** |
| `GET /api/v1/forms/download-all` | **Done** |
| `GET /api/v1/forms/search?q=` | **Done** |
| BNSS question/retrieval pipeline | **Done** |
| Local Ollama LLM integration | **Done** |
| Citation validation / refusal handling | **Done** |
| User-document ingestion | **Done** |
| Complete HTTP API for chat/query/upload | **Not Attempted / Partial** |
| Authentication | **Not Attempted** |
| Full API-level upload → ready → query flow | **Not Attempted** |

### Part E — Security / Reliability
**Status: Partial**

Document/session isolation and citation validation are implemented.

User-uploaded documents are kept separate from the BNSS statutory corpus, and uploaded content is treated as untrusted data.

Full authentication, authorization, rate limiting, and production API security were not implemented.

### Part F — Evaluation & Observability
**Status: Partial — ~90% of implemented evaluation requirements**

Implemented:

- Golden-set retrieval evaluation
- Recall@5 / Recall@10
- MRR
- Dense vs Dense + Cross-Encoder comparison
- Refusal evaluation
- Citation accuracy evaluation
- Citation coverage evaluation
- Hallucination measurement
- Retrieval latency
- Generation latency
- End-to-end latency
- End-to-end query pipeline test
- Forms extraction tests

Not implemented:

- Prometheus metrics
- Grafana dashboard
- Token usage/cost tracking
- API authentication/validation tests
- Full upload → ready → query HTTP E2E test



1. Part A — Retrieval & Indexing (AI Engineer / Backend) — 30%

A1. Structure-aware ingestion — Done
Custom PyMuPDF-based parser (backend/app/ingestion/parser.py) extracts BNSS sections by structure (chapter → section → subsection → clause), not fixed character count. 530/530 sections parsed, 673 chunks, with provisos, exceptions, and illustrations kept attached to parent sections. Cross-references are detected. Known gap: 2/530 sections are missing extracted titles due to a margin-note block merging edge case.

A2. Embeddings — Done
BAAI/bge-base-en-v1.5, 768-dimensional, open-source, run locally via sentence-transformers. passage:/query: prefixes are applied correctly. Embeddings are generated through a one-time batch job rather than during container boot.

A3. Vector store & retrieval — Done
Qdrant + BM25 hybrid retrieval, fused using Reciprocal Rank Fusion (RRF). Metadata filtering by chapter/act/section was verified. Cross-encoder reranking using ms-marco-MiniLM-L-6-v2 is implemented. A deterministic direct-lookup path handles explicit section queries (e.g. "section 41 BNSS") and bypasses similarity search entirely.

A4. Query understanding & answer generation — Done
Queries are checked for explicit section intent first. Explicit section queries use deterministic Qdrant section lookup; other queries use dense + BM25 hybrid retrieval followed by Cross-Encoder reranking. Answer generation runs locally through Ollama using llama3.2:3b, with a provider abstraction that also supports Groq. The answer pipeline applies a confidence threshold, refusal path, and citation validation to prevent unsupported or uncited legal answers.

A5. User document ingestion — Done
User PDFs are processed separately from the BNSS statute corpus using PyMuPDF. Documents are extracted page-by-page, split into sentence-aware chunks, embedded using BAAI/bge-base-en-v1.5, and stored in a separate Qdrant user_documents collection. Each chunk includes session_id, document_id, document name, page number, and ingestion timestamp, providing session/document isolation and keeping uploaded documents separate from the statutory corpus.

### Cross-Encoder Reranking

We use `cross-encoder/ms-marco-MiniLM-L-6-v2` after hybrid retrieval.

The reason is that dense retrieval and BM25 are optimized for different signals. RRF combines their rankings effectively, but the resulting top candidates can still contain less relevant results. A Cross-Encoder evaluates the **query and candidate text together**, providing a stronger final relevance signal.

We selected `ms-marco-MiniLM-L-6-v2` because it is a relatively lightweight model that can run locally and provides a practical balance between reranking quality and latency.

Evaluation confirmed the benefit of reranking:

* Dense + Cross-Encoder Recall@5: **0.92**
* Dense-only Recall@5: **0.68**
* Dense + Cross-Encoder MRR: **0.857**
* Dense-only MRR: **0.509**

The trade-off is increased retrieval latency, but the improvement in retrieval quality was significant enough to select the reranked configuration.

## BM25 + Dense Retrieval

Dense retrieval alone can miss exact legal terminology, section references, and keyword-specific matches. BM25 provides a complementary lexical retrieval signal.

We therefore combine:

Dense Qdrant search + BM25 → RRF → Cross-Encoder

This is particularly useful for legal text where exact terms such as section numbers, legal phrases, and specific procedural terminology are important

## Qdrant

Qdrant was selected as the vector database because it provides local vector storage, cosine-similarity search, metadata filtering, and payload storage in the same system.

The full chunk metadata is stored with each vector, allowing retrieved results to retain section, chapter, page, and other legal-document information required for citation generation and filtering.

## 2. — Forms Extraction Pipeline

## Part B — Forms Extraction Pipeline (20%)

**Status: Done**

Pages 190–249 of the BNSS PDF contain the statutory forms. The project extracts these forms into a downloadable PDF library.

### B1. Page-perfect PDF extraction — Done

The forms are extracted directly from the source PDF using `pypdf`, preserving the original PDF pages rather than re-rendering them as screenshots.

**Result:** 58 forms extracted successfully.

### B2. Title-based filename generation — Done

Form titles are scraped from the source PDF rather than hardcoded. Filenames are generated using the required deterministic format:

`FORM-<number>_<slugified-title>.pdf`

The filenames are filesystem-safe, contain no spaces, and are checked for collisions.

### B3. Multi-page form detection — Done

Multi-page forms are detected and kept as a single PDF.

**Verified example:**

- FORM 33 spans pages 222–224.
- It is extracted as one 3-page PDF.

This avoids the one-page-one-file error described in the assignment.

### B4. Forms manifest — Done

The pipeline generates:

`data/forms/forms_manifest.json`

The manifest records the form number, scraped title, source page range, output filename, byte size, SHA-256, extraction confidence, and `needs_review` status.

58 forms were detected and **0 forms were flagged for review**.

### B5. OCR fallback — Done

An OCR fallback using Tesseract is implemented for pages where the PDF text layer is missing or unusable.

For the supplied BNSS PDF, the normal text layer was sufficient, so no pages required OCR.

### B6. Idempotency — Done

The extraction pipeline is deterministic. Running it multiple times on the same source PDF produces byte-identical PDF outputs and does not duplicate manifest entries.

### B7. Forms API — Done

The forms are exposed through the following endpoints:

- `GET /api/v1/forms` — list all extracted forms
- `GET /api/v1/forms/{id}/download` — download a single form
- `GET /api/v1/forms/download-all` — download all forms as a ZIP
- `GET /api/v1/forms/search?q=` — search forms by title

### Part B Validation

**5/5 tests passed:**

1. Title extraction test
2. Multi-page form detection test
3. Manifest validation test
4. Filename uniqueness/collision test
5. Idempotency test

Test result:

```text
5 passed in 14.79s

## 7. Part F — Evaluation & Observability

Retrieval Evaluation

A golden evaluation set is maintained at:

eval/golden_set.jsonl

It contains 30 questions:

25 in-scope retrieval questions
5 out-of-scope questions that must be refused

Two retrieval configurations were evaluated.

Metric	Dense + Cross-Encoder	Dense-Only
Questions	30	30
Retrieval questions	25	25
Recall@5	0.92	0.68
Recall@10	0.96	0.92
MRR	0.857	0.509
Retrieval p50	585.66 ms	69.34 ms
Retrieval p95	2519.78 ms	86.27 ms

The Dense + Cross-Encoder configuration was selected because it achieved substantially better retrieval quality: Recall@5 increased from 0.68 to 0.92 and MRR from 0.509 to 0.857. The trade-off is higher retrieval latency.

Refusal Evaluation

Five out-of-scope questions were tested.

Must-refuse questions: 5
Correctly refused: 5
Refusal accuracy: 100%
Status: PASS

This ensures the system does not confidently answer questions outside the BNSS corpus.

Citation Accuracy

The answer pipeline was evaluated on 25 questions.

Metric	Result
Evaluation questions	25
Total citations generated	21
Valid citations	13
Invalid citations	8
Citation accuracy	61.90%
Citation coverage	84.00%
Answers with citations	21
Answers without citations	4
Answers with hallucinated citations	8
Hallucination rate	38.10%

The citation guard validates generated section citations against the retrieved context and removes invalid citations.

End-to-End Latency

A 10-question latency test measured retrieval, generation, and total request time.

Stage	     p50	     p95	    Average
Retrieval	139.17 ms	6966.10 ms	1712.83 ms
Generation	5437.13 ms	17778.45 ms	5928.71 ms
End-to-end	6051.99 ms	23600.38 ms	7641.54 ms

The measurements show that LLM generation is the main contributor to end-to-end latency, while retrieval is generally much faster.

Test Coverage

The evaluation suite includes:

Golden-set retrieval evaluation
Recall@5 and Recall@10
Mean Reciprocal Rank (MRR)
Out-of-scope refusal testing
Citation accuracy validation
Retrieval latency measurement
Generation latency measurement
End-to-end latency measurement
End-to-end upload/query pipeline test

The end-to-end test successfully completed with:

1 passed in 103.11s

Remaining Part F Items

Prometheus/Grafana observability, token/cost tracking, and API-based tests were not implemented. The current project does not have an HTTP API layer, so API happy-path, authentication, validation, and full upload → ready → query E2E tests were not performed. The forms parser test was also not implemented because no forms parser module exists in the project.

------------------------------------------------------------------------------------------------
## How to Run the Project

## Prerequisites

- Python 3.10+
- Docker
- Ollama with llama3.2:3b

## Install Dependencies

```bash
pip install pymupdf sentence-transformers qdrant-client rank_bm25 scikit-learn requests fastapi uvicorn python-multipart pypdf pytesseract pdf2image pytest

## How to Run each part 2) A1 A2 A3 A4 A5 F

## A1 — Ingestion

Installs: pymupdf

python backend/app/ingestion/parser.py data/raw/bnss_bare_act_2023.pdf

→ writes data/processed/bnss_chunks.jsonl

## A2 — Embeddings

Installs: sentence-transformers

python backend/app/ingestion/embed_chunks.py

writes data/embeddings/bns_embeddings.jsonl

Downloads BAAI/bge-base-en-v1.5 (~440 MB) on first run.

## A3 — Vector store & retrieval

Installs: qdrant-client, rank_bm25

Start Qdrant:
docker run -p 6333:6333 qdrant/qdrant
Load embeddings:
python backend/app/ingestion/index_to_qdrant.py
Query:
python backend/app/retrieval/query_bnss.py

A4 — Citation contract and generation

Requires: Ollama with llama3.2:3b

ollama pull llama3.2:3b

Set the provider:

set LLM_PROVIDER=ollama
set OLLAMA_MODEL=llama3.2:3b

Run:

python backend/app/llm/ask_bnss.py

Requires Qdrant running from A3.

## Run A4 tests:

python -m pytest -q 
backend/tests/test_a4_citation_guard.py 
backend/tests/test_a4_edge_cases.py 
backend/tests/test_a4_no_citation_guard.py

## A5 — User document ingestion

Upload/ingest a PDF directly through the ingestion function:

python backend/app/ingestion/document_ingest.py path\to\document.pdf session_1

The pipeline extracts the PDF, creates chunks, generates BGE embeddings, and stores the document in the separate user_documents Qdrant collection.

-----------------------------------------------------------------------------------------------

## Part B — Forms Extraction

Run:

python backend/app/forms/extract_forms.py

Expected output:

Detecting forms on pages 190-249...
Detected 58 forms (1 multi-page)
Extracting page-perfect PDFs to data/forms/ ...
0/58 forms flagged needs_review
Wrote manifest: data/forms/forms_manifest.json

Generated forms:

data/forms/

Manifest:

data/forms/forms_manifest.json

The pipeline automatically detects form titles, creates one PDF per form, handles multi-page forms, generates deterministic filenames, creates the manifest, and provides OCR fallback support.

FORM 33 is correctly extracted as a single 3-page PDF.

Run Forms Tests
python -m pytest -q eval/test_forms.py

Expected:

5 passed

# Forms API

Start the API:

uvicorn backend.app.main:app --reload

Base URL:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

Available endpoints:

GET /api/v1/forms
GET /api/v1/forms/search?q=
GET /api/v1/forms/{id}/download
GET /api/v1/forms/download-all

List forms:

curl http://127.0.0.1:8000/api/v1/forms

Search forms:

curl "http://127.0.0.1:8000/api/v1/forms/search?q=warrant"

Download a form:

curl -o FORM-33.pdf http://127.0.0.1:8000/api/v1/forms/33/download

Download all forms:

curl -o bnss_forms.zip http://127.0.0.1:8000/api/v1/forms/download-all

-----------------------------------------------------------------------------------------------

## Part F — Evaluation

Golden-set retrieval evaluation:

python eval/run_evaluation.py

This compares:

Dense + Cross-Encoder
Dense-only

and reports Recall@5, Recall@10, MRR, and retrieval latency.

Citation accuracy:

python eval/test_citation_accuracy.py

Refusal evaluation:

python eval/test_refusal.py

End-to-end latency:

python eval/test_latency.py

Reports retrieval, generation, and end-to-end p50/p95 latency.

End-to-end functional test:

python -m pytest -q eval/test_e2e.py

The current E2E test verifies:

query → retrieval → answer generation → citation validation

Expected result:

1 passed
Part F — Tests not implemented

The current project does not expose an HTTP API, so API endpoint, authentication, and validation tests are not included.

Prometheus/Grafana observability and token-based cost tracking are also not implemented in the current version.

-------------------------------------------------------------------------------------------

## API Usage

The project uses the following APIs/frameworks:

- **FastAPI** — used to expose the backend HTTP API and Forms endpoints.
- **Qdrant API** — used for vector storage and similarity search.
- **Ollama API** — used for local LLM inference with `llama3.2:3b`.
- **Tavily Search API** — used for web search where required by the research/search pipeline.

### Forms API Endpoints

- `GET /api/v1/forms` — list all extracted forms.
- `GET /api/v1/forms/search?q=<query>` — search forms.
- `GET /api/v1/forms/{id}/download` — download a specific form.
- `GET /api/v1/forms/download-all` — download all forms as a ZIP.

The FastAPI application can be started with:

```bash
uvicorn backend.app.main:app --reload

-------------------------------------------------------------------------------------

## 8. AI Usage Disclosure

AI was used extensively during this project, especially because I am relatively new to building an advanced RAG system. I used AI as a learning and development assistant to understand the architecture, implement unfamiliar components, debug errors, and improve the system through testing.

I did not treat AI-generated code as automatically correct. My approach was to first look at the error myself and try to understand where the issue was coming from. I would debug and try to solve the problem on my own first; if I could not resolve it, I would then use AI to understand the issue and find possible solutions. I primarily used ChatGPT and Claude throughout the project for learning, debugging, and implementation assistance.

### Where AI Was Used

Most of the advanced RAG-related parts were AI-assisted, including:

- PDF parsing and structure-aware chunking.
- Embedding generation and Sentence Transformers integration.
- Qdrant vector indexing.
- BM25 retrieval.
- Hybrid retrieval and Reciprocal Rank Fusion (RRF).
- Cross-Encoder reranking.
- Ollama/LLM integration.
- Citation validation.
- User-document ingestion.
- Forms extraction and multi-page form handling.
- Evaluation and testing scripts.
- Debugging and troubleshooting.

AI assistance was especially useful for understanding how the different RAG components fit together and how to debug issues between the individual stages.

### Code and Parts Written by Me

Although AI was used for a large portion of the implementation, I personally wrote and/or configured important parts of the project and FastAPI endpoint structure..

For example, I wrote the main pipeline code and configuration sections myself and understood how the individual components connect.

Manually created the `__init__.py` files in the different packages so that the project sections/modules were properly structured and importable.

One example of code I wrote/configured myself is the embedding configuration:

```python
# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = "data/processed/bnss_chunks.jsonl"
OUTPUT_FILE = "data/embeddings/bns_embeddings.jsonl"

MODEL_NAME = "BAAI/bge-base-en-v1.5"

BATCH_SIZE = 32

## Some representative prompts I used during development were:

"How can I parse the BNSS PDF while preserving its chapter, section, subsection, and clause structure?"
"How can I generate embeddings for my chunks using BAAI/bge-base-en-v1.5?"
"How should I combine Qdrant and BM25 results using Reciprocal Rank Fusion?"
"How can I implement citation validation so the generated answer only cites retrieved BNSS sections?"
"How can I extract the forms from the BNSS PDF while keeping multi-page forms together and generating deterministic filenames?"

### How I Refined Prompts

I usually tried to understand and debug an issue myself first. If I could not solve it, I used ChatGPT or Claude and provided the actual error, output, or code so the response could be based on the real problem rather than a generic solution.

For example, while working on the retrieval pipeline, I noticed that some results returned by BM25 were not respecting the same metadata filtering as the Qdrant results. I shared the retrieval output and the relevant filtering code with AI, which helped identify that filtering was being applied to Qdrant but not consistently to BM25. I then modified the implementation and tested the results again.

Another example was selecting the local Ollama model. This was mainly done through my own testing rather than AI-generated code. I tested different model sizes and found that a model around 4.9 GB was too large to run reliably together with the embedding model and Cross-Encoder on my system, while a model around 1.5 GB was not giving sufficiently good results. I then tested `llama3.2:3b`, which is around 2 GB, and found it to be a good balance between memory usage and answer quality. I selected it based on the actual runtime behavior and testing.

---------------------------------------------------------------------------------------------------

## Conclusion

This project helped me understand RAG much better than I could have learned from tutorials alone. Working with a real legal document, building the retrieval pipeline, debugging issues, evaluating the results, and connecting the different components gave me a much clearer understanding of how RAG systems actually work.

Going forward, I want to build more RAG projects myself, with a stronger understanding of the code and architecture behind each component rather than relying only on ready-made implementations.