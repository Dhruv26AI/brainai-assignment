import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.app.llm.citation_guard import validate_and_strip_citations
from backend.app.llm.providers import MockProvider
from backend.app.llm.answer import answer_question

fake_chunks = [
    {"section_number": "41", "section_title": "Arrest by Magistrate.", "subsection": "(1)",
     "text": "When any offence is committed in the presence of a Magistrate...",
     "page_start": 15, "page_end": 15},
    {"section_number": "57", "section_title": "Person arrested to be taken before Magistrate.", "subsection": None,
     "text": "A police officer making an arrest without warrant shall...",
     "page_start": 19, "page_end": 19},
]

print("=" * 70)
print("TEST 5: direct_lookup path bypasses confidence threshold entirely")
print("(even though dense score is very low, direct-lookup is an exact match, not a guess)")
print("=" * 70)
direct_lookup_result = {
    "results": [{"id": 0, "score": 1.0, "payload": fake_chunks[0]}],  # no rerank_score at all
    "retrieval_path": "direct_lookup",
    "detected_section": "41",
}
llm = MockProvider(canned_response="A Magistrate may arrest directly [BNSS s.41(1)].")
res = answer_question("what is section 41", direct_lookup_result, llm)
print("refused:", res["refused"], "| retrieval_path:", res["retrieval_path"])
assert res["refused"] is False, "direct_lookup should never be refused regardless of score"
print("PASSED\n")

print("=" * 70)
print("TEST 6: multiple hallucinated citations all get stripped, valid ones survive")
print("=" * 70)
messy_answer = (
    "Section 41 lets a magistrate act [BNSS s.41(1)]. "
    "But [BNSS s.777] overrides this in emergencies, and [BNSS s.888(3)] adds a penalty. "
    "Also see [BNSS s.57] for procedure."
)
result = validate_and_strip_citations(messy_answer, fake_chunks)
print("Invalid citations found:", result["invalid_citations"])
assert len(result["invalid_citations"]) == 2
assert ("777", None) in result["invalid_citations"]
assert ("888", "3") in result["invalid_citations"]
assert "[BNSS s.41(1)]" in result["cleaned_answer"]
assert "[BNSS s.57]" in result["cleaned_answer"]
assert "s.777" not in result["cleaned_answer"]
assert "s.888" not in result["cleaned_answer"]
print("PASSED\n")

print("=" * 70)
print("TEST 7: a citation with a WRONG subsection on a VALID section IS caught")
print("(fixed: guard now checks subsection match, not just section number)")
print("=" * 70)
# Section 41 is real, its actual subsection is (1) -- this cites a fabricated (99)
sneaky_answer = "This is covered under [BNSS s.41(99)]."
result7 = validate_and_strip_citations(sneaky_answer, fake_chunks)
print("had_hallucination:", result7["had_hallucination"])
print("invalid_citations:", result7["invalid_citations"])
assert result7["had_hallucination"] is True
assert ("41", "99") in result7["invalid_citations"]
# Sanity: citing section 41 WITHOUT a subsection, or with its real subsection, still passes
assert validate_and_strip_citations("[BNSS s.41]", fake_chunks)["had_hallucination"] is False
assert validate_and_strip_citations("[BNSS s.41(1)]", fake_chunks)["had_hallucination"] is False
print("PASSED\n")

print("=" * 70)
print("TEST 8 (KNOWN LIMITATION): a differently-formatted citation isn't recognized at all")
print("e.g. '[Section 41]' instead of '[BNSS s.41]' -- passes through unchecked")
print("=" * 70)
unformatted_answer = "This is covered under [Section 41] of the code."
result8 = validate_and_strip_citations(unformatted_answer, fake_chunks)
print("citations found:", result8["valid_citations"], result8["invalid_citations"])
assert result8["valid_citations"] == [] and result8["invalid_citations"] == []
print("CONFIRMED LIMITATION -- relies on the system prompt enforcing exact format (see DECISIONS.md)\n")

print("ALL A4 EDGE-CASE TESTS COMPLETE (2 known limitations documented, not silently hidden)")
