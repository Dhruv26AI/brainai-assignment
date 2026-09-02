"""
A5 - Query Router

Routes a user question to:
    - document      -> user uploaded document only
    - statute       -> BNSS statute only
    - both          -> both document and statute

Important:
Uploaded documents are untrusted data.
This router does NOT execute instructions found inside documents.
It only classifies the user's question.
"""

import re


# ---------------------------------------------------------
# Route types
# ---------------------------------------------------------

DOCUMENT = "document"
STATUTE = "statute"
BOTH = "both"


# ---------------------------------------------------------
# Statute-related patterns
# ---------------------------------------------------------

STATUTE_PATTERNS = [
    r"\bbnss\b",
    r"\bbns\b",
    r"\bsection\s+\d+[a-z]?\b",
    r"\bsec\.?\s*\d+[a-z]?\b",
    r"\bsubsection\b",
    r"\bsub-section\b",
    r"\bclause\b",
    r"\bproviso\b",
    r"\bbare act\b",
    r"\bpolice\s+arrest\b",
    r"\barrest\s+without\s+(a\s+)?warrant\b",
    r"\bbail\b",
    r"\bcognizable\s+offence\b",
    r"\bmagistrate\b",
]


# ---------------------------------------------------------
# Document-related patterns
# ---------------------------------------------------------

DOCUMENT_PATTERNS = [
    r"\bmy\s+(document|notice|fir|agreement|judgment|judgement)\b",
    r"\bthis\s+(document|notice|fir|agreement|judgment|judgement)\b",
    r"\bthe\s+(document|notice|fir|agreement|judgment|judgement)\b",
    r"\buploaded\s+(document|notice|fir|agreement|judgment|judgement)\b",
    r"\bnotice\b",
    r"\bfir\b",
    r"\bagreement\b",
    r"\bjudgment\b",
    r"\bjudgement\b",
    r"\baccording\s+to\s+(the|my)\b",
    r"\bwhat\s+does\s+(it|this)\s+say\b",
    r"\bwho\s+(sent|issued|received)\b",
    r"\bmentioned\s+in\s+(the|my)\b",
    r"\bin\s+(the|my)\s+document\b",
]


# ---------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------

def _matches_patterns(
    query: str,
    patterns: list[str],
) -> bool:
    """
    Return True if any pattern matches the query.
    """

    return any(
        re.search(pattern, query, re.IGNORECASE)
        for pattern in patterns
    )


# ---------------------------------------------------------
# Main routing function
# ---------------------------------------------------------

def route_query(query: str) -> str:
    """
    Decide which corpus should be searched.

    Returns:
        "document"
        "statute"
        "both"
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    query = query.strip()

    statute_match = _matches_patterns(
        query,
        STATUTE_PATTERNS,
    )

    document_match = _matches_patterns(
        query,
        DOCUMENT_PATTERNS,
    )

    # -----------------------------------------------------
    # Both
    # -----------------------------------------------------
    #
    # Example:
    # "Does this notice comply with Section 35?"
    #
    # The question contains both document context
    # ("notice") and statutory context ("Section 35").
    # -----------------------------------------------------

    if statute_match and document_match:
        return BOTH

    # -----------------------------------------------------
    # Statute only
    # -----------------------------------------------------

    if statute_match:
        return STATUTE

    # -----------------------------------------------------
    # Document only
    # -----------------------------------------------------

    if document_match:
        return DOCUMENT

    # -----------------------------------------------------
    # Default
    # -----------------------------------------------------
    #
    # If the question does not clearly mention either corpus,
    # use the statute corpus because the main application is
    # a legal statute QA system.
    # -----------------------------------------------------

    return STATUTE


# ---------------------------------------------------------
# Detailed routing information
# ---------------------------------------------------------

def classify_query(query: str) -> dict:
    """
    Return routing decision plus useful debugging information.
    """

    route = route_query(query)

    return {
        "query": query,
        "route": route,
        "search_document": route in (DOCUMENT, BOTH),
        "search_statute": route in (STATUTE, BOTH),
    }


# ---------------------------------------------------------
# Command-line test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("A5 QUERY ROUTER TEST")
    print("=" * 60)

    test_queries = [
        "What does my notice say?",
        "Who sent this notice?",
        "What is BNSS Section 35?",
        "Can police arrest someone without a warrant?",
        "Does this notice comply with Section 35?",
        "What does the agreement say about payment?",
        "What does Section 57 say?",
    ]

    for query in test_queries:

        result = classify_query(query)

        print()
        print(f"Question: {query}")
        print(f"Route: {result['route']}")
        print(f"Search document: {result['search_document']}")
        print(f"Search statute: {result['search_statute']}")

    print()
    print("=" * 60)
    print("ROUTER TEST COMPLETE")
    print("=" * 60)