# DhronAI — Architecture

## 1. System Overview

```mermaid
flowchart TD
    U[User] --> API[FastAPI]
    API --> R[Query Router]
    R --> Q[Qdrant]
    R --> B[BM25]
    Q --> F[RRF]
    B --> F
    F --> CE[Cross-Encoder]
    CE --> LLM[Ollama]
    LLM --> C[Citation Guard]
    C --> A[Final Answer]

    PDF[PDF Upload] --> P[PyMuPDF]
    P --> CH[Chunking]
    CH --> E[BGE Embeddings]
    E --> Q

2. Request Lifecycle
Upload
PDF → PyMuPDF → Sentence-aware chunks → BGE embeddings
→ User Documents Qdrant collection → Ready
Statute Question
Question → Query Router → Dense + BM25 → RRF
→ Cross-Encoder → Ollama → Citation Guard → Answer

Explicit section queries can use direct BNSS section lookup.

Document Question
Question → Document Scope → User-document Qdrant
→ Cross-Encoder → Ollama → Citation Validation → Answer

User documents are kept separate from the BNSS statute corpus.

3. Chunking Schema

BNSS chunks preserve:

Chapter
 └── Section
      └── Subsection
           └── Clause
                └── Proviso / Exception / Illustration

Typical metadata:

{
  "act_short": "BNSS",
  "section": "35",
  "section_title": "...",
  "subsection": "...",
  "clause": "...",
  "text": "...",
  "cross_references": []
}

The corpus contains 530 sections and 673 chunks.

User PDFs use sentence-aware chunks with document and page metadata.

4. Retrieval Flow
Query
  ↓
Dense Retrieval ──┐
                  ├── RRF → Cross-Encoder → LLM
BM25 Retrieval ───┘
Qdrant: semantic/vector retrieval.
BM25: lexical retrieval.
RRF: combines rankings (k=60).
Cross-Encoder: reranks candidates.
Citation Guard: validates generated citations.
5. Main Components
Component	Technology
API	FastAPI
PDF parsing	PyMuPDF
Embeddings	BAAI/bge-base-en-v1.5
Vector store	Qdrant
Lexical search	BM25
Reranker	ms-marco-MiniLM-L-6-v2
LLM	Ollama llama3.2:3b
Frontend	React
