import json
from sentence_transformers import SentenceTransformer
from backend.app.retrieval.qdrant_store import VectorStore
from backend.app.retrieval.bm25_index import BM25Index
from backend.app.retrieval.hybrid_search import HybridRetriever
from backend.app.llm.answer import answer_question
from backend.app.llm.providers import get_llm_provider

with open("data/processed/bnss_chunks.jsonl", encoding="utf-8") as f:
    chunks = [json.loads(x) for x in f]

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embed_fn = lambda q: model.encode("query: " + q, normalize_embeddings=True).tolist()

bm25 = BM25Index(chunks)
vector_store = VectorStore("http://localhost:6333", 384)
retriever = HybridRetriever(vector_store, bm25, embed_fn)
llm = get_llm_provider()

def test_end_to_end():
    question = "What does Section 35 say about arrest without warrant?"
    retrieval = retriever.search(question, top_k=5)
    result = answer_question(question, retrieval, llm)

    assert retrieval["results"]
    assert result["answer"]
    assert result["refused"] is False
    assert result["sources"]
    assert result["had_hallucination"] is False