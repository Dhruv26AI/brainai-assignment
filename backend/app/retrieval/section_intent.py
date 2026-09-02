"""
A3 - Direct-lookup intent detection.

If a user asks "what is section 103 BNSS" / "s.103" / "section 103(1)", retrieval must
return section 103 *deterministically* -- not whatever the embedding's cosine similarity
happened to favour. This module detects that intent and extracts the section number (and
optional subsection) so the caller can bypass semantic search and go straight to a metadata
filter lookup.
"""
import re

# Matches: "section 103", "sec 103", "s.103", "s 103(1)", "BNSS 103", "BNS s.103"
SECTION_INTENT_RE = re.compile(
    r"""
    (?:section|sec\.?|s\.|BNSS?)     # a "section" keyword, or the act's short name
    \s*                              # optional space
    (\d{1,3}[A-Z]?)                  # the section number, e.g. 103 or 103A
    (?:\s*\((\d+)\))?                # optional subsection, e.g. (1)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def detect_section_intent(query: str) -> dict | None:
    """
    Returns {"section_number": "103", "subsection": "1"} if the query contains an
    explicit section reference, else None. Only fires on a genuine section-number
    pattern -- it will not fire on generic questions like "what is bail".
    """
    match = SECTION_INTENT_RE.search(query)
    if not match:
        return None
    return {
        "section_number": match.group(1),
        "subsection": match.group(2),  # may be None
    }


if __name__ == "__main__":
    tests = [
        "what is section 103 BNSS",
        "explain s.41(1)",
        "BNSS 480",
        "what happens if the police don't file a chargesheet on time",  # should NOT match
        "tell me about bail conditions",  # should NOT match
    ]
    for t in tests:
        print(f"{t!r:65} -> {detect_section_intent(t)}")
