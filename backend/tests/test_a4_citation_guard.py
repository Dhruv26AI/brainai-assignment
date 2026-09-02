import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.llm.citation_guard import validate_and_strip_citations, extract_cited_sections
from backend.app.llm.providers import MockProvider
from backend.app.llm.answer import answer_question

# ---- fake retrieved context: only sections 41 and 57 were actually retrieved ----
fake_chunks_payload = [
    {"section_number": "41", "section_title": "Arrest by Magistrate.", "subsection": None,
     "text": "When any offence is committed in the presence of a Magistrate...",
     "page_start": 15, "page_end": 15},
    {"section_number": "57", "section_title": "Person arrested to be taken before Magistrate.", "subsection": None,
     "text": "A police officer making an arrest without warrant shall...",
     "page_start": 19, "page_end": 19},
]

print("=" * 70)
print("TEST 1: citation guard strips a hallucinated section (s.999 doesn't exist in context)")
print("=" * 70)
fake_answer = (
    "A Magistrate may arrest someone directly [BNSS s.41]. "
    "The person must then be produced before a Magistrate [BNSS s.57]. "
    "Failure to do so voids the arrest entirely [BNSS s.999]."  # <-- hallucinated
)
result = validate_and_strip_citations(fake_answer, fake_chunks_payload)
print("Cited sections found:", extract_cited_sections(fake_answer))
print("Valid citations:     ", result["valid_citations"])
print("Invalid citations:   ", result["invalid_citations"])
print("had_hallucination:   ", result["had_hallucination"])
print("Cleaned answer:\n ", result["cleaned_answer"])
assert result["had_hallucination"] is True
assert ("999", None) in result["invalid_citations"]
assert "[BNSS s.999]" not in result["cleaned_answer"]
assert "[BNSS s.41]" in result["cleaned_answer"]  # valid citation untouched
print("PASSED\n")

print("=" * 70)
print("TEST 2: clean answer with only valid citations passes through untouched")
print("=" * 70)
clean_answer = "The Magistrate may act directly [BNSS s.41]. Otherwise [BNSS s.57] applies."
result2 = validate_and_strip_citations(clean_answer, fake_chunks_payload)
print("had_hallucination:", result2["had_hallucination"])
assert result2["had_hallucination"] is False
assert result2["cleaned_answer"] == clean_answer
print("PASSED\n")

print("=" * 70)
print("TEST 3: full pipeline - refusal path fires on low confidence")
print("=" * 70)
low_confidence_retrieval = {
    "results": [{"id": 0, "score": 0.01, "rerank_score": -5.2, "payload": fake_chunks_payload[0]}],
    "retrieval_path": "hybrid_rrf+rerank",
    "detected_section": None,
}
mock_llm = MockProvider(canned_response="this should never be called")
res = answer_question("what is the punishment for jaywalking in Ohio", low_confidence_retrieval, mock_llm)
print("refused:", res["refused"], "| reason:", res["refusal_reason"], "| top_score:", res["top_score"])
assert res["refused"] is True
assert res["answer"] != "this should never be called"
print("PASSED\n")

print("=" * 70)
print("TEST 4: full pipeline - high confidence generates and validates an answer")
print("=" * 70)
high_confidence_retrieval = {
    "results": [
        {"id": 0, "score": 0.03, "rerank_score": 3.2, "payload": fake_chunks_payload[0]},
        {"id": 1, "score": 0.02, "rerank_score": 2.1, "payload": fake_chunks_payload[1]},
    ],
    "retrieval_path": "hybrid_rrf+rerank",
    "detected_section": None,
}
mock_llm_hallucinating = MockProvider(
    canned_response="A Magistrate can arrest directly [BNSS s.41]. Appeals go under [BNSS s.999]."
)
res2 = answer_question("can a magistrate arrest someone", high_confidence_retrieval, mock_llm_hallucinating)
print("refused:", res2["refused"])
print("had_hallucination:", res2["had_hallucination"])
print("invalid_citations_stripped:", res2["invalid_citations_stripped"])
print("final answer:", res2["answer"])
print("num sources returned for source panel:", len(res2["sources"]))
assert res2["refused"] is False
assert res2["had_hallucination"] is True
assert "[BNSS s.999]" not in res2["answer"]
print("PASSED\n")

print("ALL A4 TESTS PASSED")
