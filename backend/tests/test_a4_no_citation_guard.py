import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.app.llm.providers import MockProvider
from backend.app.llm.answer import answer_question

fake_chunks = [
    {"section_number": "41", "section_title": "Arrest by Magistrate.", "subsection": None,
     "text": "When any offence is committed in the presence of a Magistrate...",
     "page_start": 15, "page_end": 15},
]
high_confidence_retrieval = {
    "results": [{"id": 0, "score": 0.03, "rerank_score": 3.2, "payload": fake_chunks[0]}],
    "retrieval_path": "hybrid_rrf+rerank",
    "detected_section": None,
}

print("=" * 70)
print("TEST 9: an answer with ZERO citations is refused, never shown as a normal answer")
print("(this is the direct guard against the assignment's automatic-rejection criterion:")
print(" 'A chatbot that answers legal questions without citations')")
print("=" * 70)
uncited_llm = MockProvider(canned_response="Yes, a magistrate can arrest someone directly in this situation.")
res = answer_question("can a magistrate arrest someone", high_confidence_retrieval, uncited_llm)
print("refused:", res["refused"], "| reason:", res.get("refusal_reason"))
print("answer shown to user:", res["answer"])
assert res["refused"] is True
assert res["refusal_reason"] == "no_valid_citations_generated"
print("PASSED -- uncited answer correctly blocked, not shown\n")

print("=" * 70)
print("TEST 10: a properly-cited answer still passes through normally (no false positive)")
print("=" * 70)
cited_llm = MockProvider(canned_response="Yes, per [BNSS s.41] a magistrate may arrest directly.")
res2 = answer_question("can a magistrate arrest someone", high_confidence_retrieval, cited_llm)
print("refused:", res2["refused"])
assert res2["refused"] is False
print("PASSED\n")

print("=" * 70)
print("TEST 11: a legitimate 'I don't have enough information' refusal is NOT double-blocked")
print("=" * 70)
refusal_llm = MockProvider(canned_response="I don't have enough information in the statute to answer that.")
res3 = answer_question("can a magistrate arrest someone", high_confidence_retrieval, refusal_llm)
print("refused:", res3["refused"], "| reason:", res3.get("refusal_reason"))
assert res3["refused"] is True
assert res3.get("refusal_reason") != "no_valid_citations_generated"  # should pass through as-is, not relabeled
print("PASSED\n")

print("ALL TESTS PASSED -- system can never present an uncited legal answer as valid")
